"""
DRIFT vs DFNO — Training Convergence Sanity Check

Runs 100 training iterations on PDEBench 3D NS with both DFNO and DRIFT,
logs per-iteration loss, and saves the loss curves for plotting.

Both models use identical architecture (lift, proj, width, modes, blocks)
and the same optimizer/LR. The ONLY difference is the spectral transform.
If DRIFT is mathematically exact, the loss curves should overlap.

Usage:
  mpirun -np 4 python train_convergence.py --data-file ./data/ns3d_128x128x128_tin5_tout16.pt
  mpirun -np 4 python train_convergence.py --data-file ./data/ns3d_128x128x128_tin5_tout16.pt --iters 200 --lr 1e-3
"""

import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from mpi4py import MPI

from utils import create_standard_partitions
from dfno import DistributedFNO, BroadcastedLinear
from drift_block import PartialDFTFNOBlock
import distdl.nn as dnn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", type=str, required=True)
    p.add_argument("--modes", type=int, nargs=4, default=[8, 8, 8, 8])
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
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


def main():
    args = parse_args()
    comm, rank, ws, device = setup()

    data = torch.load(args.data_file, weights_only=False, map_location='cpu')
    nx, ny, nz = data['grid']
    t_in, t_out = data['t_in'], data['t_out']
    n_channels = data['x_train'].shape[1]

    n_samples = min(10, data['x_train'].shape[0])
    x_all = data['x_train'][:n_samples].contiguous()  # [N, V, X, Y, Z, t_in]
    y_all = data['y_train'][:n_samples].contiguous()  # [N, V, X, Y, Z, t_out]

    P_shape = make_partition(ws, [nx, ny, nz], args.modes[2])
    Px, Py = P_shape[2], P_shape[3]
    lx, ly = nx // Px, ny // Py

    if rank == 0:
        print("=" * 65)
        print("  DRIFT vs DFNO — Training Convergence Check")
        print(f"  Ranks: {ws}   Device: {device}")
        print(f"  Grid: ({nx},{ny},{nz})  Channels: {n_channels}")
        print(f"  Modes: {args.modes}  Width: {args.width}  Blocks: {args.blocks}")
        print(f"  Iters: {args.iters}  LR: {args.lr}  Seed: {args.seed}")
        print(f"  Training samples: {n_samples}")
        print(f"  Data: {args.data_file}")
        print("=" * 65)

    _, P_x, _ = create_standard_partitions(P_shape)

    def slice_local(tensor, idx):
        """Slice one sample and extract local spatial partition."""
        s = tensor[idx:idx + 1]  # [1, V, X, Y, Z, T]
        if Py == 1:
            return s[:, :, rank * lx:(rank + 1) * lx].contiguous().to(device)
        else:
            cx, cy = rank // Py, rank % Py
            return s[:, :, cx * lx:(cx + 1) * lx, cy * ly:(cy + 1) * ly].contiguous().to(device)

    torch.manual_seed(args.seed)
    in_shape = (1, n_channels, nx, ny, nz, t_in)
    dfno = DistributedFNO(P_x, in_shape, t_out, args.width, tuple(args.modes),
                          num_blocks=args.blocks, device=device, dtype=torch.float32)
    dfno.train()
    opt_dfno = torch.optim.Adam(dfno.parameters(), lr=args.lr)
    loss_fn_dfno = dnn.DistributedMSELoss(P_x)

    torch.manual_seed(args.seed)  # same init
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
            self.proj2 = BroadcastedLinear(P_x, 128, 1, dim=1, device=device)
            self.dt_comm = 0.0

        def forward(self, x):
            self.dt_comm = 0.0
            x = F.gelu(self.lift_t(x)); self.dt_comm += self.lift_t.dt_comm
            x = F.gelu(self.lift_c(x)); self.dt_comm += self.lift_c.dt_comm
            for blk in self.blocks:
                x = blk(x); self.dt_comm += blk.dt_comm
            x = F.gelu(self.proj1(x)); self.dt_comm += self.proj1.dt_comm
            x = self.proj2(x); self.dt_comm += self.proj2.dt_comm
            return x

    drift = DRIFTModel()
    drift.train()
    opt_drift = torch.optim.Adam(drift.parameters(), lr=args.lr)
    loss_fn_drift = dnn.DistributedMSELoss(P_x)

    dfno_p = sum(p.numel() for p in dfno.parameters())
    drift_p = sum(p.numel() for p in drift.parameters())
    if rank == 0:
        print(f"    DFNO params/GPU:  {dfno_p:,}")
        print(f"    DRIFT params/GPU: {drift_p:,}")

    if rank == 0: print(f"\n[3] Training ({args.iters} iterations)...")

    dfno_losses = []
    drift_losses = []
    dfno_times = []
    drift_times = []

    for it in range(args.iters):
        idx = it % n_samples
        x_local = slice_local(x_all, idx)
        y_local = slice_local(y_all, idx)

        # Target: take first channel of y to match DFNO's 1-channel output
        # DFNO outputs (1, 1, lx, ly, nz, t_out), y_local is (1, V, lx, ly, nz, t_out)
        y_target = y_local[:, 0:1].contiguous()

        comm.Barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        opt_dfno.zero_grad()
        pred_dfno = dfno(x_local)
        loss_dfno = loss_fn_dfno(pred_dfno, y_target)
        loss_dfno.backward()
        opt_dfno.step()

        torch.cuda.synchronize()
        comm.Barrier()
        dt_dfno = time.perf_counter() - t0

        comm.Barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        opt_drift.zero_grad()
        pred_drift = drift(x_local)
        loss_drift = loss_fn_drift(pred_drift, y_target)
        loss_drift.backward()
        opt_drift.step()

        torch.cuda.synchronize()
        comm.Barrier()
        dt_drift = time.perf_counter() - t0

        l_dfno = loss_dfno.item()
        l_drift = loss_drift.item()
        dfno_losses.append(l_dfno)
        drift_losses.append(l_drift)
        dfno_times.append(dt_dfno)
        drift_times.append(dt_drift)

        if rank == 0 and (it % 10 == 0 or it == args.iters - 1):
            print(f"  iter {it:4d}  |  DFNO loss: {l_dfno:.6f} ({dt_dfno*1000:.0f}ms)  "
                  f"|  DRIFT loss: {l_drift:.6f} ({dt_drift*1000:.0f}ms)")

    dfno.eval()
    drift.eval()

    n_test = min(data['x_test'].shape[0], 10)
    dfno_test_l2 = []
    drift_test_l2 = []

    with torch.no_grad():
        for ti in range(n_test):
            x_local = slice_local(data['x_test'], ti)
            y_local = slice_local(data['y_test'], ti)
            y_target = y_local[:, 0:1].contiguous()

            pred_dfno = dfno(x_local)
            pred_drift = drift(x_local)

            def rel_l2_dist(pred, target):
                d = comm.allreduce(((pred - target) ** 2).sum().item(), op=MPI.SUM)
                n = comm.allreduce((target ** 2).sum().item(), op=MPI.SUM)
                return (d / (n + 1e-12)) ** 0.5

            dfno_test_l2.append(rel_l2_dist(pred_dfno, y_target))
            drift_test_l2.append(rel_l2_dist(pred_drift, y_target))

    dfno_test_l2 = np.array(dfno_test_l2)
    drift_test_l2 = np.array(drift_test_l2)

    if rank == 0:
        print(f"    DFNO  test rel L2: {dfno_test_l2.mean():.4f} ± {dfno_test_l2.std():.4f}")
        print(f"    DRIFT test rel L2: {drift_test_l2.mean():.4f} ± {drift_test_l2.std():.4f}")

    if rank == 0:
        dfno_losses = np.array(dfno_losses)
        drift_losses = np.array(drift_losses)
        dfno_times = np.array(dfno_times)
        drift_times = np.array(drift_times)

        print(f"\n{'=' * 65}")
        print(f"  TRAINING CONVERGENCE RESULTS ({args.iters} iterations)")
        print(f"{'=' * 65}")
        print(f"  {'':25s} {'DFNO':>12s} {'DRIFT':>12s}")
        print(f"  {'-' * 50}")
        print(f"  {'Final loss':25s} {dfno_losses[-1]:>12.6f} {drift_losses[-1]:>12.6f}")
        print(f"  {'Min loss':25s} {dfno_losses.min():>12.6f} {drift_losses.min():>12.6f}")
        print(f"  {'Test rel L2 (mean)':25s} {dfno_test_l2.mean():>12.4f} {drift_test_l2.mean():>12.4f}")
        print(f"  {'Test rel L2 (std)':25s} {dfno_test_l2.std():>12.4f} {drift_test_l2.std():>12.4f}")
        print(f"  {'Mean iter time (ms)':25s} {dfno_times.mean()*1000:>10.1f}   {drift_times.mean()*1000:>10.1f}")
        print(f"  {'Training speedup':25s} {'':>12s} {dfno_times.mean()/drift_times.mean():>11.1f}x")
        print(f"{'=' * 65}")

        os.makedirs("results", exist_ok=True)
        fn = f"results/convergence_P{ws}_{nx}x{ny}x{nz}.json"
        with open(fn, 'w') as f:
            json.dump({
                "config": {
                    "world_size": ws, "grid": [nx, ny, nz],
                    "modes": args.modes, "width": args.width, "blocks": args.blocks,
                    "iters": args.iters, "lr": args.lr, "seed": args.seed,
                    "n_samples": n_samples,
                },
                "dfno_losses": dfno_losses.tolist(),
                "drift_losses": drift_losses.tolist(),
                "dfno_times_ms": (dfno_times * 1000).tolist(),
                "drift_times_ms": (drift_times * 1000).tolist(),
                "dfno_test_rel_l2": dfno_test_l2.tolist(),
                "drift_test_rel_l2": drift_test_l2.tolist(),
            }, f, indent=2)
        print(f"  Saved: {fn}")

        # Save numpy for easy plotting
        np.savez(f"results/convergence_P{ws}_{nx}x{ny}x{nz}.npz",
                 dfno_losses=dfno_losses, drift_losses=drift_losses,
                 dfno_times=dfno_times, drift_times=drift_times,
                 dfno_test_l2=dfno_test_l2, drift_test_l2=drift_test_l2)
        print(f"  Saved: results/convergence_P{ws}_{nx}x{ny}x{nz}.npz")


if __name__ == "__main__":
    main()