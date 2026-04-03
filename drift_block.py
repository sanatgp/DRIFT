# DRIFT Block
# Pipeline:
#   Local transforms → Local partial DFT (Bᵢ) → AllReduce →
#   Local slice (rank p takes modes Sₚ) → Sharded spectral conv (R_θ^(p)) →
#   AllGather → Local partial iDFT (Bᵢ⁻¹) → Local inverse transforms
#
# Communication:
#   AllReduce: 2M·c·(P-1)/P bytes  (same as ReduceScatter + AllGather)
#   AllGather: M·c·(P-1)/P bytes
#   Total: 3M·c·(P-1)/P bytes
#
# Model parallelism: W_spec sharded across GPUs (S_local = S/P per GPU)
# Data parallelism: spatial domain partitioned along x-dimension
# When P=1: collapses to local-only (no communication)

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mpi4py import MPI
import cupy as cp


def torch_to_cupy(t: torch.Tensor) -> cp.ndarray:
    assert t.is_cuda and t.is_contiguous()
    return cp.from_dlpack(torch.utils.dlpack.to_dlpack(t))

def cupy_to_torch(a: cp.ndarray) -> torch.Tensor:
    return torch.utils.dlpack.from_dlpack(a.toDlpack())



def _raw_allreduce_gpu(x_complex: torch.Tensor, comm: MPI.Comm):
    """
    AllReduce SUM on complex tensor. Every rank gets the full reduced result.
    Input:  x_complex of any shape, complex, contiguous
    Output: tensor of same shape with elementwise sum across ranks
    """
    send_real = torch.view_as_real(x_complex.contiguous()).contiguous()
    recv_real = torch.empty_like(send_real)

    torch.cuda.current_stream().synchronize()
    t0 = time.perf_counter()
    comm.Allreduce(torch_to_cupy(send_real.reshape(-1)),
                   torch_to_cupy(recv_real.reshape(-1)), op=MPI.SUM)
    torch.cuda.current_stream().synchronize()
    dt_mpi = time.perf_counter() - t0

    out_complex = torch.view_as_complex(recv_real)
    return out_complex, dt_mpi


def _raw_allgather_gpu(x_complex: torch.Tensor, comm: MPI.Comm, shard_dim: int = 0):
    """
    AllGather: each rank contributes its local slice, all ranks get the full tensor.
    Input:  x_complex of shape (..., S_local, ...)
    Output: tensor of shape (..., S_local * P, ...)
    """
    world_size = comm.Get_size()
    shape = x_complex.shape
    S_local = shape[shard_dim]
    S_global = S_local * world_size

    perm = list(range(x_complex.ndim))
    perm[0], perm[shard_dim] = perm[shard_dim], perm[0]
    x_perm = x_complex.permute(perm).contiguous()

    out_shape = (S_global, *x_perm.shape[1:])
    out_complex = torch.empty(out_shape, dtype=x_complex.dtype, device=x_complex.device)

    send_real = torch.view_as_real(x_perm.contiguous()).reshape(-1).contiguous()
    recv_real = torch.view_as_real(out_complex.contiguous()).reshape(-1).contiguous()

    torch.cuda.current_stream().synchronize()
    t0 = time.perf_counter()
    comm.Allgather(torch_to_cupy(send_real), torch_to_cupy(recv_real))
    torch.cuda.current_stream().synchronize()
    dt_mpi = time.perf_counter() - t0

    out_complex = torch.view_as_complex(
        recv_real.reshape(torch.view_as_real(out_complex).shape)
    ).reshape(out_shape)

    inv_perm = list(range(len(perm)))
    for i, p_i in enumerate(perm):
        inv_perm[p_i] = i
    out_complex = out_complex.permute(inv_perm).contiguous()

    return out_complex, dt_mpi

class _AllReduceFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, comm):
        ctx.comm = comm
        if comm.Get_size() <= 1:
            _AllReduceFunc._dt_mpi = 0.0
            return x
        result, dt_mpi = _raw_allreduce_gpu(x, comm)
        _AllReduceFunc._dt_mpi = dt_mpi
        return result

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.comm.Get_size() <= 1:
            return grad_output, None
        # Adjoint of AllReduce(SUM) is AllReduce(SUM)
        result, _ = _raw_allreduce_gpu(grad_output, ctx.comm)
        return result, None


class _AllGatherFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, comm, shard_dim):
        ctx.comm = comm
        ctx.shard_dim = shard_dim
        if comm.Get_size() <= 1:
            _AllGatherFunc._dt_mpi = 0.0
            return x
        result, dt_mpi = _raw_allgather_gpu(x, comm, shard_dim)
        _AllGatherFunc._dt_mpi = dt_mpi
        return result

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.comm.Get_size() <= 1:
            return grad_output, None, None
        # Adjoint of AllGather is ReduceScatter
        # Implement as AllReduce + local slice
        comm = ctx.comm
        shard_dim = ctx.shard_dim
        world_size = comm.Get_size()
        rank = comm.Get_rank()

        result, _ = _raw_allreduce_gpu(grad_output, comm)
        S_global = result.shape[shard_dim]
        S_local = S_global // world_size
        start = rank * S_local
        out = result.narrow(shard_dim, start, S_local).contiguous()
        return out, None, None


class PartialDFTFNOBlock(nn.Module):
    """
    Communication: AllReduce → local slice → sharded spectral conv → AllGather
    W_spec: (S_local, width, width) where S_local = S // world_size
    Each GPU holds 1/P of the spectral modes.
    """

    def __init__(self, P_x, in_shape, modes, device=None, dtype=torch.float32,
                 shard_weights=True):
        super().__init__()

        from dfno import BroadcastedLinear

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert device.type == "cuda", "GPU-only implementation."
        self.device = device

        self.P_x = P_x
        self.in_shape = in_shape
        self.width = in_shape[1]
        self.modes = tuple(modes)
        self.dtype = dtype
        self.dtype_c = torch.complex64 if dtype == torch.float32 else torch.complex128

        self.comm = P_x._comm
        self.world_size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.Px = P_x.shape[2]
        self.Py = P_x.shape[3]

        if len(in_shape) == 5:
            local_x, local_y, Z, T = in_shape[2], in_shape[3], 1, in_shape[4]
            self._is_2d = True
            self.modes_x, self.modes_y = self.modes[0], self.modes[1]
            self.modes_z = 0
            self.modes_t = self.modes[2]
        else:
            local_x, local_y, Z, T = in_shape[2], in_shape[3], in_shape[4], in_shape[5]
            self._is_2d = False
            self.modes_x, self.modes_y, self.modes_z = self.modes[0], self.modes[1], self.modes[2]
            self.modes_t = self.modes[3] if len(self.modes) > 3 else self.modes[2]

        self.local_x = local_x
        self.local_y = local_y
        global_X = local_x * self.Px
        global_Y = local_y * self.Py
        self.X, self.Y, self.Z, self.T = global_X, global_Y, Z, T

        coord_x = P_x.index[2]
        coord_y = P_x.index[3]

        def kept_freqs(N, m):
            if m <= 0:
                return []
            return list(range(m)) + list(range(N - m, N))

        kept_kx = kept_freqs(global_X, self.modes_x)
        kept_ky = kept_freqs(global_Y, self.modes_y)

        #X, Y basis
        def make_basis(N, local_n, offset, kept_k):
            j = torch.arange(local_n, device=device, dtype=torch.float64)
            k = torch.tensor(kept_k, device=device, dtype=torch.float64)
            ang = -2.0 * math.pi * k[:, None] * (offset + j[None, :]) / float(N)
            basis = torch.complex(torch.cos(ang), torch.sin(ang)).to(self.dtype_c)
            ibasis = torch.complex(torch.cos(-ang), torch.sin(-ang)).to(self.dtype_c) / float(N)
            return basis, ibasis

        bx, ibx = make_basis(global_X, self.local_x, coord_x * self.local_x, kept_kx)
        by, iby = make_basis(global_Y, self.local_y, coord_y * self.local_y, kept_ky)

        self.register_buffer("bx_T", bx.t().contiguous(), persistent=False)
        self.register_buffer("by_T", by.t().contiguous(), persistent=False)
        self.register_buffer("ibx", ibx.contiguous(), persistent=False)
        self.register_buffer("iby", iby.contiguous(), persistent=False)

        self.Kt = self.modes_t

        n_t = torch.arange(T, device=device, dtype=torch.float64)
        k_t = torch.arange(self.Kt, device=device, dtype=torch.float64)

        ang_t = -2.0 * math.pi * k_t[:, None] * n_t[None, :] / float(T)
        bt_fwd = torch.complex(torch.cos(ang_t), torch.sin(ang_t)).to(self.dtype_c)
        self.register_buffer("bt_fwd_T", bt_fwd.t().contiguous(), persistent=False)

        inv_ang_t = 2.0 * math.pi * k_t[:, None] * n_t[None, :] / float(T)
        bt_inv = torch.complex(torch.cos(inv_ang_t), torch.sin(inv_ang_t)).to(self.dtype_c)
        scale_t = torch.ones(self.Kt, 1, device=device, dtype=torch.float64) * 2.0 / float(T)
        scale_t[0] = 1.0 / float(T)
        if T % 2 == 0 and self.Kt > T // 2:
            scale_t[T // 2] = 1.0 / float(T)
        elif T % 2 == 0 and self.Kt == T // 2:
            scale_t[-1] = 1.0 / float(T)
        bt_inv = (bt_inv.to(torch.complex128) * scale_t.to(torch.complex128)).to(self.dtype_c)
        self.register_buffer("bt_inv", bt_inv.contiguous(), persistent=False)

        #Z basis
        self.Kz = (2 * self.modes_z) if (Z > 1 and self.modes_z > 0) else 1
        self._has_z = (Z > 1 and self.modes_z > 0)

        if self._has_z:
            kept_kz = kept_freqs(Z, self.modes_z)
            n_z = torch.arange(Z, device=device, dtype=torch.float64)
            k_z = torch.tensor(kept_kz, device=device, dtype=torch.float64)

            ang_z = -2.0 * math.pi * k_z[:, None] * n_z[None, :] / float(Z)
            bz_fwd = torch.complex(torch.cos(ang_z), torch.sin(ang_z)).to(self.dtype_c)
            inv_ang_z = 2.0 * math.pi * k_z[:, None] * n_z[None, :] / float(Z)
            bz_inv = torch.complex(torch.cos(inv_ang_z), torch.sin(inv_ang_z)).to(self.dtype_c) / float(Z)

            self.register_buffer("bz_fwd_T", bz_fwd.t().contiguous(), persistent=False)
            self.register_buffer("bz_inv", bz_inv.contiguous(), persistent=False)

        #spectral shape and sharded weights
        self.Kx = 2 * self.modes_x
        self.Ky = 2 * self.modes_y
        self.spec_spatial = (self.Kx, self.Ky, self.Kz, self.Kt)
        self.S = self.Kx * self.Ky * self.Kz * self.Kt

        if self.world_size > 1:
            assert self.S % self.world_size == 0, \
                f"Total spectral modes S={self.S} must be divisible by world_size={self.world_size}"
            self.S_local = self.S // self.world_size
        else:
            self.S_local = self.S

        scale = 1.0 / (self.width * self.width)
        self.W_spec = nn.Parameter(
            scale * torch.randn(self.S_local, self.width, self.width,
                                device=device, dtype=self.dtype_c)
        )

        self.linear = BroadcastedLinear(P_x, self.width, self.width, bias=False, dim=1,
                                        device=device, dtype=dtype)
        self.dt_comm = 0.0

    def forward(self, x):
        assert x.is_cuda
        self.dt_comm = 0.0

        # Skip connection
        y0 = self.linear(x)
        self.dt_comm += getattr(self.linear, "dt_comm", 0.0)

        squeeze_z = False
        if x.ndim == 5:
            x = x.unsqueeze(4)
            squeeze_z = True

        B = x.shape[0]

        #pDFT on T
        orig_shape = x.shape
        T_dim = orig_shape[-1]
        x = torch.mm(x.reshape(-1, T_dim).to(self.dtype_c), self.bt_fwd_T)
        x = x.view(*orig_shape[:-1], self.Kt)

        # pDFT on Z
        if self._has_z:
            x = x.permute(0, 1, 2, 3, 5, 4).contiguous()
            shape_pre = x.shape
            Z_dim = shape_pre[-1]
            x = torch.mm(x.reshape(-1, Z_dim), self.bz_fwd_T)
            x = x.view(*shape_pre[:-1], self.Kz)
            x = x.permute(0, 1, 2, 3, 5, 4).contiguous()

        # x: (B, W, lx, ly, Kz, Kt) complex
        Kz = x.shape[4]
        Kt = x.shape[5]
        batch = B * self.width * Kz * Kt

        # pDFT Y then X 
        x_y = x.permute(0, 1, 4, 5, 2, 3).reshape(batch * self.local_x, self.local_y)
        s_y = torch.mm(x_y, self.by_T)

        s_y = s_y.view(batch, self.local_x, self.Ky).permute(0, 2, 1).reshape(
            batch * self.Ky, self.local_x)
        s_xy = torch.mm(s_y, self.bx_T)

        spectral = s_xy.view(batch, self.Ky, self.Kx).permute(0, 2, 1)
        spectral = spectral.view(B, self.width, Kz, Kt, self.Kx, self.Ky).permute(
            0, 1, 4, 5, 2, 3
        ).contiguous()
        # spectral: (B, width, Kx, Ky, Kz, Kt)

        #AllReduce + local slice 
        spectral_flat = spectral.view(B, self.width, self.S)

        t0 = time.time()

        if self.world_size > 1:
            spectral_full = _AllReduceFunc.apply(spectral_flat, self.comm)
            # Local slice: each GPU takes its partition of the spectrum
            start = self.rank * self.S_local
            spectral_local = spectral_full.narrow(2, start, self.S_local).contiguous()
            self.dt_comm += _AllReduceFunc._dt_mpi
        else:
            spectral_local = spectral_flat

        # W_spec: (S_local, width, width), spectral_local: (B, width, S_local)
        x_bat = spectral_local.permute(2, 1, 0).contiguous()  # (S_local, width, B)
        y_bat = torch.bmm(self.W_spec, x_bat)                 # (S_local, width, B)
        y_local = y_bat.permute(2, 1, 0).contiguous()         # (B, width, S_local)

        t1 = time.time()

        if self.world_size > 1:
            y_full = _AllGatherFunc.apply(y_local, self.comm, 2)
            self.dt_comm += _AllGatherFunc._dt_mpi
        else:
            y_full = y_local

        y = y_full.view(B, self.width, *self.spec_spatial)

        #inverse pDFT X then Y
        batch_inv = B * self.width * Kz * Kt
        y_inv = y.permute(0, 1, 4, 5, 3, 2).reshape(batch_inv * self.Ky, self.Kx)
        s_x = torch.mm(y_inv, self.ibx)

        s_x = s_x.view(batch_inv, self.Ky, self.local_x).permute(0, 2, 1).reshape(
            batch_inv * self.local_x, self.Ky)
        s_xy = torch.mm(s_x, self.iby)

        spatial = s_xy.view(batch_inv, self.local_x, self.local_y)
        spatial = spatial.view(B, self.width, Kz, Kt, self.local_x, self.local_y).permute(
            0, 1, 4, 5, 2, 3
        ).contiguous()

        # === Phase 8: inverse pDFT on Z ===
        if self._has_z:
            spatial = spatial.permute(0, 1, 2, 3, 5, 4).contiguous()
            shape_pre = spatial.shape
            spatial = torch.mm(spatial.reshape(-1, self.Kz), self.bz_inv)
            spatial = spatial.view(*shape_pre[:-1], self.Z)
            spatial = spatial.permute(0, 1, 2, 3, 5, 4).contiguous()

        # inverse pDFT on T 
        sp_shape = spatial.shape
        sp_flat = spatial.reshape(-1, self.Kt)
        y = torch.mm(sp_flat, self.bt_inv).real
        y = y.view(*sp_shape[:-1], self.T)

        if y.dtype != self.dtype:
            y = y.to(self.dtype)

        if squeeze_z:
            y = y.squeeze(4)

        return F.gelu(y0 + y)

    def cleanup(self):
        pass