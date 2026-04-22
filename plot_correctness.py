"""
DRIFT correctness verification figure.

(a) Spectral coefficient comparison: FFT+truncate vs partial DFT
(b) Distributed full-model comparison: DFNO vs DRIFT, matched weights

Usage:
  mpirun -np 16 python plot_correctness.py \
      --data-file ./data/ns3d_128x128x128_tin5_tout16.pt \
      --modes 8 8 8 16 --sample 4 --channel 1
"""

import argparse, os, math
import numpy as np
import torch
import torch.nn.functional as F
from mpi4py import MPI

from utils import create_standard_partitions
from dfno import DistributedFNO, BroadcastedLinear
from drift_block import PartialDFTFNOBlock


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", type=str, required=True)
    p.add_argument("--modes", type=int, nargs=4, default=[8, 8, 8, 16])
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--sample", type=int, default=4,
                   help="Which x_test sample to use (Fig. 7 uses sample 4)")
    p.add_argument("--channel", type=int, default=1,
                   help="Channel to visualize (1=vx)")
    p.add_argument("--input-vmax", type=float, default=5.0,
                   help="Input vx colorbar range: [-vmax, +vmax]")
    p.add_argument("--output-vmax", type=float, default=0.04,
                   help="Model output colorbar range: [-vmax, +vmax]")
    p.add_argument("--spec-vmax", type=float, default=0.1,
                   help="Spectral magnitude colorbar upper bound (normalized)")
    p.add_argument("--err-vmin", type=float, default=1e-18,
                   help="Error panel LogNorm lower bound")
    p.add_argument("--err-vmax", type=float, default=1e-15,
                   help="Error panel LogNorm upper bound")
    p.add_argument("--out", type=str, default="fig_correctness.pdf")
    return p.parse_args()


def setup():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ws = comm.Get_size()
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return comm, rank, ws, device


def make_partition(ws, shape, modes_z):
    max_z_ranks = 2 * modes_z
    if ws <= max_z_ranks:
        return (1, 1, ws, 1, 1, 1)
    best = None
    for px in range(max_z_ranks, 0, -1):
        if ws % px != 0:
            continue
        py = ws // px
        if shape[0] % px != 0 or shape[1] % py != 0:
            continue
        best = (px, py)
        break
    if best is None:
        raise ValueError(f"No valid 2D partition for P={ws}, shape={shape}")
    px, py = best
    return (1, 1, px, py, 1, 1)


def copy_blinear(dst, src):
    with torch.no_grad():
        if src.P_root.active and dst.P_root.active:
            dst.W.data.copy_(src.W.data)
            dst.b.data.copy_(src.b.data)


def reassemble_2d(chunks, Px, Py, nx, ny):
    if Py == 1:
        return np.concatenate(chunks, axis=1)
    rows = []
    for cx in range(Px):
        row_chunks = [chunks[cx * Py + cy] for cy in range(Py)]
        rows.append(np.concatenate(row_chunks, axis=2))
    return np.concatenate(rows, axis=1)


def rel_l2(a, b, comm):
    d = comm.allreduce(((a - b) ** 2).sum().item(), op=MPI.SUM)
    n = comm.allreduce((b ** 2).sum().item(), op=MPI.SUM)
    return math.sqrt(d / (n + 1e-12))


def spectral_comparison(x_full, modes, channel, z_mid):
    """
    Compare FFT+truncate vs partial DFT on a 2D slice.
    Uses 2*modes coefficients to match DFNO's rfft/fft truncation.
    """
    kx, ky = 2 * modes[0], 2 * modes[1]
    field = x_full[0, channel, :, :, z_mid, 0]  # [nx, ny]
    nx, ny_dim = field.shape

    # FFT + truncate (what DFNO does)
    X_fft_full = torch.fft.fft2(field.double())
    X_fft = X_fft_full[:kx, :ky]

    # Partial DFT (what DRIFT does)
    bx = torch.exp(-2j * torch.pi * torch.arange(kx, dtype=torch.float64).unsqueeze(1)
                   * torch.arange(nx, dtype=torch.float64).unsqueeze(0) / nx)
    by = torch.exp(-2j * torch.pi * torch.arange(ky, dtype=torch.float64).unsqueeze(1)
                   * torch.arange(ny_dim, dtype=torch.float64).unsqueeze(0) / ny_dim)
    X_pdft = bx @ field.double().to(torch.complex128) @ by.T

    error = (X_fft - X_pdft).abs()
    rel_err = error.norm() / X_fft.abs().norm()

    return (field.numpy(),
            X_fft.abs().numpy(),
            X_pdft.abs().numpy(),
            error.numpy(),
            rel_err.item(),
            kx, ky)


def main():
    args = parse_args()
    comm, rank, ws, device = setup()

    data = torch.load(args.data_file, weights_only=False, map_location='cpu')
    nx, ny, nz = data['grid']
    t_in, t_out = data['t_in'], data['t_out']
    x_full = data['x_test'][args.sample:args.sample+1].contiguous()
    n_channels = x_full.shape[1]
    z_mid = nz // 2

    if rank == 0:
        input_slice, fft_coeffs, pdft_coeffs, spec_error, rel_frob, kx_viz, ky_viz = \
            spectral_comparison(x_full, args.modes, args.channel, z_mid)
        print(f"Spectral coefficient error: {rel_frob:.1e} (relative Frobenius)")

    P_shape = make_partition(ws, [nx, ny, nz], args.modes[2])
    Px, Py = P_shape[2], P_shape[3]
    lx, ly = nx // Px, ny // Py

    if Py == 1:
        x_local = x_full[:, :, rank * lx:(rank + 1) * lx].contiguous().to(device)
    else:
        cx, cy = rank // Py, rank % Py
        x_local = x_full[:, :, cx * lx:(cx + 1) * lx, cy * ly:(cy + 1) * ly].contiguous().to(device)

    _, P_x, _ = create_standard_partitions(P_shape)
    in_shape = (1, n_channels, nx, ny, nz, t_in)

    # DFNO
    torch.manual_seed(12345)
    dfno = DistributedFNO(P_x, in_shape, t_out, args.width, tuple(args.modes),
                          num_blocks=args.blocks, device=device, dtype=torch.float32)
    dfno.eval()

    # DRIFT
    block_shape = (1, args.width, lx, ly, nz, t_out)

    class DRIFTModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lift_t = BroadcastedLinear(P_x, t_in, t_out, dim=-1, device=device)
            self.lift_c = BroadcastedLinear(P_x, n_channels, args.width, dim=1, device=device)
            self.blocks = torch.nn.ModuleList([
                PartialDFTFNOBlock(P_x, block_shape, tuple(args.modes),
                                   device=device, shard_weights=True)
                for _ in range(args.blocks)
            ])
            self.proj1 = BroadcastedLinear(P_x, args.width, 128, dim=1, device=device)
            self.proj2 = BroadcastedLinear(P_x, 128, n_channels, dim=1, device=device)

        def forward(self, x):
            x = F.gelu(self.lift_t(x))
            x = F.gelu(self.lift_c(x))
            for blk in self.blocks:
                x = blk(x)
            x = F.gelu(self.proj1(x))
            x = self.proj2(x)
            return x

    drift = DRIFTModel()
    drift.eval()

    #Sync shared weights
    with torch.no_grad():
        copy_blinear(drift.lift_t, dfno.linear1)
        copy_blinear(drift.lift_c, dfno.linear2)
        copy_blinear(drift.proj1, dfno.linear3)
        copy_blinear(drift.proj2, dfno.linear4)
        for i in range(len(dfno.blocks)):
            copy_blinear(drift.blocks[i].linear, dfno.blocks[i].linear)

    dfno.linear4 = BroadcastedLinear(P_x, 128, n_channels, dim=1, device=device)
    drift.proj2 = BroadcastedLinear(P_x, 128, n_channels, dim=1, device=device)
    with torch.no_grad():
        copy_blinear(drift.proj2, dfno.linear4)

    with torch.no_grad():
        for blk in dfno.blocks:
            for w in blk.weights:
                w.data.zero_()
        for blk in drift.blocks:
            blk.W_spec.data.zero_()

    with torch.no_grad():
        y_dfno = dfno(x_local)
        y_drift = drift(x_local)

    err = rel_l2(y_drift, y_dfno, comm)

    x_in_list = comm.gather(x_local[0].cpu().numpy(), root=0)
    y_d_list = comm.gather(y_dfno[0].cpu().numpy(), root=0)
    y_r_list = comm.gather(y_drift[0].cpu().numpy(), root=0)

    if rank == 0:
        x_in_full = reassemble_2d(x_in_list, Px, Py, nx, ny)
        y_d_full = reassemble_2d(y_d_list, Px, Py, nx, ny)
        y_r_full = reassemble_2d(y_r_list, Px, Py, nx, ny)

        err_str = "bitwise identical" if err == 0.0 else f"{err:.1e}"
        print(f"Distributed model error: {err_str} (relative L2)")

        # Plot
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        ch = args.channel
        modes_str = f"({args.modes[0]}, {args.modes[1]}, {args.modes[2]}, {args.modes[3]})"

        fft_norm = fft_coeffs / fft_coeffs.max() * args.spec_vmax
        pdft_norm = pdft_coeffs / pdft_coeffs.max() * args.spec_vmax

        fig = plt.figure(figsize=(16, 9))

        def add_cbar(ax, im):
            div = make_axes_locatable(ax)
            cax = div.append_axes("bottom", size="5%", pad=0.25)
            plt.colorbar(im, cax=cax, orientation="horizontal")

        gs_a = fig.add_gridspec(1, 4, left=0.05, right=0.95, top=0.82, bottom=0.52,
                                wspace=0.35)

        fig.text(0.05, 0.85, "(a) Spectral coefficient comparison",
                 fontsize=12, fontweight="bold")

        spec_extent = [0, kx_viz - 1, 0, ky_viz - 1]

        ax = fig.add_subplot(gs_a[0, 0])
        ax.set_title(f"PDEBench input ($v_x$, z={z_mid})", fontsize=10)
        im = ax.imshow(input_slice.T, origin="lower", cmap="RdBu_r",
                       vmin=-args.input_vmax, vmax=args.input_vmax,
                       extent=[0, nx, 0, ny])
        ax.set_xlabel("x"); ax.set_ylabel("y")
        add_cbar(ax, im)

        ax = fig.add_subplot(gs_a[0, 1])
        ax.set_title("FFT + truncate (DFNO)", fontsize=10)
        im = ax.imshow(fft_norm.T, origin="lower", cmap="viridis",
                       vmin=0, vmax=args.spec_vmax,
                       extent=spec_extent)
        ax.set_xlabel("$k_x$ index"); ax.set_ylabel("$k_y$ index")
        add_cbar(ax, im)

        ax = fig.add_subplot(gs_a[0, 2])
        ax.set_title("Partial DFT (DRIFT)", fontsize=10)
        im = ax.imshow(pdft_norm.T, origin="lower", cmap="viridis",
                       vmin=0, vmax=args.spec_vmax,
                       extent=spec_extent)
        ax.set_xlabel("$k_x$ index")
        add_cbar(ax, im)

        ax = fig.add_subplot(gs_a[0, 3])
        ax.set_title(r"$|\hat{X}_{\mathrm{FFT}} - \hat{X}_{\mathrm{pDFT}}|$",
                     fontsize=10)
        im = ax.imshow(spec_error.T, origin="lower", cmap="hot",
                       norm=LogNorm(vmin=args.err_vmin, vmax=args.err_vmax),
                       extent=spec_extent)
        ax.set_xlabel("$k_x$ index")
        add_cbar(ax, im)

        gs_b = fig.add_gridspec(1, 3, left=0.05, right=0.95, top=0.42, bottom=0.08,
                                wspace=0.3)

        fig.text(0.05, 0.45,
                 f"(b) Distributed full-model comparison "
                 f"(P = {ws} GPUs, matched weights, {err_str})",
                 fontsize=12, fontweight="bold")

        inp_slice = x_in_full[ch, :, :, z_mid, 0]
        ax = fig.add_subplot(gs_b[0, 0])
        ax.set_title(f"Input ($v_x$)", fontsize=10)
        im = ax.imshow(inp_slice.T, origin="lower", cmap="RdBu_r",
                       vmin=-args.input_vmax, vmax=args.input_vmax,
                       extent=[0, nx, 0, ny])
        ax.set_xlabel("x"); ax.set_ylabel("y")
        add_cbar(ax, im)

        dfno_slice = y_d_full[ch, :, :, z_mid, 0]
        drift_slice = y_r_full[ch, :, :, z_mid, 0]

        ax = fig.add_subplot(gs_b[0, 1])
        ax.set_title(f"DFNO output ($v_x$)", fontsize=10)
        im = ax.imshow(dfno_slice.T, origin="lower", cmap="RdBu_r",
                       vmin=-args.output_vmax, vmax=args.output_vmax,
                       extent=[0, nx, 0, ny])
        ax.set_xlabel("x")
        add_cbar(ax, im)

        ax = fig.add_subplot(gs_b[0, 2])
        ax.set_title(f"DRIFT output ($v_x$)", fontsize=10)
        im = ax.imshow(drift_slice.T, origin="lower", cmap="RdBu_r",
                       vmin=-args.output_vmax, vmax=args.output_vmax,
                       extent=[0, nx, 0, ny])
        ax.set_xlabel("x")
        add_cbar(ax, im)

        os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
        fig.savefig(args.out, bbox_inches="tight", dpi=200)
        print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()