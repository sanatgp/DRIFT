"""
Profiling script for DRIFT.

Usage:
  mpirun -np 4  python profile_drift.py
  mpirun -np 32 python profile_drift.py
"""

import json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpi4py import MPI

SHAPE         = [128, 128, 64]
MODES         = [8, 8, 8, 16]
WIDTH         = 20
NUM_BLOCKS    = 4
OUT_STEPS     = 30
WARMUP        = 5
TRIALS        = 20


def setup_device():
    rank = MPI.COMM_WORLD.Get_rank()
    local_rank = rank % torch.cuda.device_count()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


class Profiler:
    def __init__(self, device):
        self.use_cuda = device.type == "cuda"
        self.records = {}
        self._stack = []

    def _sync(self):
        if self.use_cuda: torch.cuda.synchronize()

    def start(self, name):
        self._sync(); self._stack.append((name, time.perf_counter()))

    def stop(self):
        self._sync(); name, t0 = self._stack.pop()
        self.records.setdefault(name, []).append(time.perf_counter() - t0)

    def record(self, name, val):
        self.records.setdefault(name, []).append(val)

    def summary(self):
        return {n: dict(mean_ms=round(float(np.mean(v))*1000, 4),
                        std_ms=round(float(np.std(v))*1000, 4))
                for n, v in self.records.items()}

    def reset(self): self.records = {}


def profiled_block(block, x, prof, idx):
    p = f"blk{idx}"
    prof.start(f"{p}/linear"); y0 = block.linear(x); prof.stop()
    prof.record(f"{p}/linear_comm", getattr(block.linear, "dt_comm", 0.0))
    squeeze_z = (x.ndim == 5)
    if squeeze_z: x = x.unsqueeze(4)
    B = x.shape[0]

    prof.start(f"{p}/pdft_T")
    orig = x.shape
    x = torch.mm(x.reshape(-1, orig[-1]).to(block.dtype_c), block.bt_fwd_T)
    x = x.view(*orig[:-1], block.Kt)
    prof.stop()

    if block._has_z:
        prof.start(f"{p}/pdft_Z")
        x = x.permute(0,1,2,3,5,4).contiguous(); sp = x.shape
        x = torch.mm(x.reshape(-1, sp[-1]), block.bz_fwd_T)
        x = x.view(*sp[:-1], block.Kz).permute(0,1,2,3,5,4).contiguous()
        prof.stop()

    Kz, Kt = x.shape[4], x.shape[5]
    batch = B * block.width * Kz * Kt

    prof.start(f"{p}/pdft_YX")
    x_y = x.permute(0,1,4,5,2,3).reshape(batch * block.local_x, block.local_y)
    s_y = torch.mm(x_y, block.by_T)
    s_y = s_y.view(batch, block.local_x, block.Ky).permute(0,2,1).reshape(batch * block.Ky, block.local_x)
    s_xy = torch.mm(s_y, block.bx_T)
    spectral = s_xy.view(batch, block.Ky, block.Kx).permute(0,2,1)
    spectral = spectral.view(B, block.width, Kz, Kt, block.Kx, block.Ky).permute(0,1,4,5,2,3).contiguous()
    prof.stop()

    spectral_flat = spectral.view(B, block.width, block.S)
    if block.world_size > 1:
        prof.start(f"{p}/allreduce")
        from drift_block import _raw_allreduce_gpu
        spectral_full, _ = _raw_allreduce_gpu(spectral_flat, block.comm)
        prof.stop()
        spectral_local = spectral_full.narrow(2, block.rank * block.S_local, block.S_local).contiguous()
    else:
        spectral_local = spectral_flat

    prof.start(f"{p}/spec_conv")
    x_bat = spectral_local.permute(2,1,0).contiguous()
    y_bat = torch.bmm(block.W_spec, x_bat)
    y_local = y_bat.permute(2,1,0).contiguous()
    prof.stop()

    if block.world_size > 1:
        prof.start(f"{p}/allgather")
        from drift_block import _raw_allgather_gpu
        y_full, _ = _raw_allgather_gpu(y_local, block.comm, shard_dim=2)
        prof.stop()
    else:
        y_full = y_local

    y = y_full.view(B, block.width, *block.spec_spatial)
    batch_inv = B * block.width * Kz * Kt

    prof.start(f"{p}/ipdft_XY")
    y_inv = y.permute(0,1,4,5,3,2).reshape(batch_inv * block.Ky, block.Kx)
    s_x = torch.mm(y_inv, block.ibx)
    s_x = s_x.view(batch_inv, block.Ky, block.local_x).permute(0,2,1).reshape(batch_inv * block.local_x, block.Ky)
    s_xy = torch.mm(s_x, block.iby)
    spatial = s_xy.view(batch_inv, block.local_x, block.local_y)
    spatial = spatial.view(B, block.width, Kz, Kt, block.local_x, block.local_y).permute(0,1,4,5,2,3).contiguous()
    prof.stop()

    if block._has_z:
        prof.start(f"{p}/ipdft_Z")
        spatial = spatial.permute(0,1,2,3,5,4).contiguous(); sp = spatial.shape
        spatial = torch.mm(spatial.reshape(-1, block.Kz), block.bz_inv)
        spatial = spatial.view(*sp[:-1], block.Z).permute(0,1,2,3,5,4).contiguous()
        prof.stop()

    prof.start(f"{p}/ipdft_T")
    sp_shape = spatial.shape
    y = torch.mm(spatial.reshape(-1, block.Kt), block.bt_inv).real
    y = y.view(*sp_shape[:-1], block.T)
    if y.dtype != block.dtype: y = y.to(block.dtype)
    prof.stop()

    if squeeze_z: y = y.squeeze(4)
    prof.start(f"{p}/gelu"); out = F.gelu(y0 + y); prof.stop()
    return out


def profiled_forward(lift1, lift2, blocks, proj3, proj4, x, prof):
    prof.start("total_fwd")
    prof.start("lift1"); out = F.gelu(lift1(x)); prof.stop()
    prof.record("lift1_comm", getattr(lift1, "dt_comm", 0.0))
    prof.start("lift2"); out = F.gelu(lift2(out)); prof.stop()
    prof.record("lift2_comm", getattr(lift2, "dt_comm", 0.0))
    for i, blk in enumerate(blocks): out = profiled_block(blk, out, prof, i)
    prof.start("proj1"); out = F.gelu(proj3(out)); prof.stop()
    prof.record("proj1_comm", getattr(proj3, "dt_comm", 0.0))
    prof.start("proj2"); out = proj4(out); prof.stop()
    prof.record("proj2_comm", getattr(proj4, "dt_comm", 0.0))
    prof.stop()
    return out


def main():
    from dfno import BroadcastedLinear
    from drift_block import PartialDFTFNOBlock
    from utils import create_standard_partitions

    device = setup_device()
    ws = MPI.COMM_WORLD.Get_size()
    P_shape = (1, 1, ws, 1, 1, 1)
    _, P_x, _ = create_standard_partitions(P_shape)
    rank = P_x.rank
    local_shape = [SHAPE[0]//ws, SHAPE[1], SHAPE[2]]

    if rank == 0:
        print(f"DRIFT Profiler | P={ws} | Grid={SHAPE} | Modes={MODES} | Width={WIDTH}")

    x = torch.rand(1, 1, *local_shape, 1, device=device, dtype=torch.float32)
    block_shape = (1, WIDTH, *local_shape, OUT_STEPS)

    lift1 = BroadcastedLinear(P_x, 1, OUT_STEPS, dim=-1, device=device)
    lift2 = BroadcastedLinear(P_x, 1, WIDTH, dim=1, device=device)
    blocks = nn.ModuleList([
        PartialDFTFNOBlock(P_x, block_shape, MODES, device=device) for _ in range(NUM_BLOCKS)
    ])
    proj3 = BroadcastedLinear(P_x, WIDTH, 128, dim=1, device=device)
    proj4 = BroadcastedLinear(P_x, 128, 1, dim=1, device=device)
    all_modules = [lift1, lift2, proj3, proj4, *blocks]

    prof = Profiler(device)
    for _ in range(WARMUP):
        y = profiled_forward(lift1, lift2, blocks, proj3, proj4, x, prof)
        y.sum().backward()
        for m in all_modules: m.zero_grad()
        P_x._comm.Barrier()
    prof.reset()

    for _ in range(TRIALS):
        P_x._comm.Barrier()
        y = profiled_forward(lift1, lift2, blocks, proj3, proj4, x, prof)
        P_x._comm.Barrier()
        prof.start("total_bwd"); y.sum().backward()
        for m in all_modules: m.zero_grad()
        prof.stop(); P_x._comm.Barrier()

    summary = prof.summary()
    COMM = ("/allreduce", "/allgather", "_comm")
    fwd = summary["total_fwd"]["mean_ms"]
    bwd = summary["total_bwd"]["mean_ms"]
    comm = sum(summary[k]["mean_ms"] for k in summary if any(t in k for t in COMM))

    if rank == 0:
        print(f"\n  Forward: {fwd:.1f}ms | Backward: {bwd:.1f}ms | Comm: {comm:.1f}ms ({100*comm/fwd:.1f}%)")
        for i in range(NUM_BLOCKS):
            ar = summary.get(f"blk{i}/allreduce", {}).get("mean_ms", 0)
            ag = summary.get(f"blk{i}/allgather", {}).get("mean_ms", 0)
            print(f"  blk{i}: AR={ar:.1f}ms, AG={ag:.1f}ms")

#        os.makedirs("results", exist_ok=True)
#        fn = f"results/drift_P{ws}.json"
#        with open(fn, 'w') as f:
#            json.dump({"fwd_ms": round(fwd,1), "bwd_ms": round(bwd,1),
#                       "comm_ms": round(comm,1), "phases": summary}, f, indent=2)

if __name__ == "__main__":
    main()