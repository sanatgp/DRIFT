"""
DRIFT vs DFNO — Epoch-based Training Convergence

Runs 100 epochs on PDEBench 3D NS with both DFNO and DRIFT.
Each epoch iterates over all training samples. After each epoch,
evaluates on the test set and logs:
  - Epoch-averaged train MSE loss
  - Test relative L2 error
  - Per-epoch wall-clock time

Usage:
  mpirun -np 16 python train_epochs.py --data-file ./data/ns3d_128x128x128_tin5_tout16.pt
  mpirun -np 16 python train_epochs.py --data-file ./data/ns3d_128x128x128_tin5_tout16.pt --epochs 100 --lr 1e-3
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
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scheduler", action="store_true", help="Use cosine annealing LR scheduler")
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


def rel_l2_dist(pred, target, comm):
    d = comm.allreduce(((pred - target) ** 2).sum().item(), op=MPI.SUM)
    n = comm.allreduce((target ** 2).sum().item(), op=MPI.SUM)
    return (d / (n + 1e-12)) ** 0.5


def main():
    args = parse_args()
    comm, rank, ws, device = setup()

    data = torch.load(args.data_file, weights_only=False, map_location='cpu')
    nx, ny, nz = data['grid']
    t_in, t_out = data['t_in'], data['t_out']
    n_channels = data['x_train'].shape[1]
    
    
    n_train = min(40, data['x_train'].shape[0])
  #  n_train = data['x_train'].shape[0]
    n_test = data['x_test'].shape[0]
    x_train = data['x_train'].contiguous()
    y_train = data['y_train'].contiguous()
    x_test = data['x_test'].contiguous()
    y_test = data['y_test'].contiguous()

    P_shape = make_partition(ws, [nx, ny, nz], args.modes[2])
    Px, Py = P_shape[2], P_shape[3]
    lx, ly = nx // Px, ny // Py

    if rank == 0:
        print("  DRIFT vs DFNO — Epoch-based Training")
        print(f"  GPUs: {ws}   Device: {device}")
        print(f"  Grid: ({nx},{ny},{nz})  Channels: {n_channels}")
        print(f"  Modes: {args.modes}  Width: {args.width}  Blocks: {args.blocks}")
        print(f"  Epochs: {args.epochs}  LR: {args.lr}  Seed: {args.seed}")
        print(f"  Train samples: {n_train}  Test samples: {n_test}")
        print(f"  Partition: Px={Px}, Py={Py}, lx={lx}, ly={ly}")


    _, P_x, _ = create_standard_partitions(P_shape)

    def slice_local(tensor, idx):
        s = tensor[idx:idx + 1]
        if Py == 1:
            return s[:, :, rank * lx:(rank + 1) * lx].contiguous().to(device)
        else:
            cx, cy = rank // Py, rank % Py
            return s[:, :, cx * lx:(cx + 1) * lx, cy * ly:(cy + 1) * ly].contiguous().to(device)

    torch.manual_seed(args.seed)
    in_shape = (1, n_channels, nx, ny, nz, t_in)
    dfno = DistributedFNO(P_x, in_shape, t_out, args.width, tuple(args.modes),
                          num_blocks=args.blocks, device=device, dtype=torch.float32)

    torch.manual_seed(args.seed)
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

    opt_dfno = torch.optim.Adam(dfno.parameters(), lr=args.lr)
    opt_drift = torch.optim.Adam(drift.parameters(), lr=args.lr)
    loss_fn_dfno = dnn.DistributedMSELoss(P_x)
    loss_fn_drift = dnn.DistributedMSELoss(P_x)

    if args.scheduler:
        sched_dfno = torch.optim.lr_scheduler.CosineAnnealingLR(opt_dfno, T_max=args.epochs)
        sched_drift = torch.optim.lr_scheduler.CosineAnnealingLR(opt_drift, T_max=args.epochs)

    if rank == 0:
        dfno_p = sum(p.numel() for p in dfno.parameters())
        drift_p = sum(p.numel() for p in drift.parameters())
        print(f"  DFNO params/GPU:  {dfno_p:,}")
        print(f"  DRIFT params/GPU: {drift_p:,}")

    rng = np.random.RandomState(args.seed)

    log = {
        'dfno_train_loss': [], 'drift_train_loss': [],
        'dfno_val_l2': [], 'drift_val_l2': [],
        'dfno_epoch_time': [], 'drift_epoch_time': [],
    }

    if rank == 0:
        print(f"\n{'Epoch':>6} | {'DFNO loss':>10} {'DFNO L2':>9} {'DFNO t':>8} | "
              f"{'DRIFT loss':>11} {'DRIFT L2':>9} {'DRIFT t':>8} | {'Speedup':>8}")
        print("-" * 90)

    for epoch in range(args.epochs):
        order = rng.permutation(n_train)

        dfno.train()
        dfno_epoch_loss = 0.0
        comm.Barrier(); torch.cuda.synchronize()
        t0 = time.perf_counter()

        for i in range(n_train):
            idx = int(order[i])
            x_local = slice_local(x_train, idx)
            y_local = slice_local(y_train, idx)
            y_target = y_local[:, 0:1].contiguous()

            opt_dfno.zero_grad()
            pred = dfno(x_local)
            loss = loss_fn_dfno(pred, y_target)
            loss.backward()
            opt_dfno.step()
            dfno_epoch_loss += loss.item()

        torch.cuda.synchronize(); comm.Barrier()
        dt_dfno = time.perf_counter() - t0
        dfno_epoch_loss /= n_train

        drift.train()
        drift_epoch_loss = 0.0
        comm.Barrier(); torch.cuda.synchronize()
        t0 = time.perf_counter()

        for i in range(n_train):
            idx = int(order[i])
            x_local = slice_local(x_train, idx)
            y_local = slice_local(y_train, idx)
            y_target = y_local[:, 0:1].contiguous()

            opt_drift.zero_grad()
            pred = drift(x_local)
            loss = loss_fn_drift(pred, y_target)
            loss.backward()
            opt_drift.step()
            drift_epoch_loss += loss.item()

        torch.cuda.synchronize(); comm.Barrier()
        dt_drift = time.perf_counter() - t0
        drift_epoch_loss /= n_train

        if args.scheduler:
            sched_dfno.step()
            sched_drift.step()

        dfno.eval()
        drift.eval()
        dfno_l2_list = []
        drift_l2_list = []

        with torch.no_grad():
            for ti in range(n_test):
                x_local = slice_local(x_test, ti)
                y_local = slice_local(y_test, ti)
                y_target = y_local[:, 0:1].contiguous()

                pred_dfno = dfno(x_local)
                pred_drift = drift(x_local)

                dfno_l2_list.append(rel_l2_dist(pred_dfno, y_target, comm))
                drift_l2_list.append(rel_l2_dist(pred_drift, y_target, comm))

        dfno_val_l2 = np.mean(dfno_l2_list)
        drift_val_l2 = np.mean(drift_l2_list)

        log['dfno_train_loss'].append(dfno_epoch_loss)
        log['drift_train_loss'].append(drift_epoch_loss)
        log['dfno_val_l2'].append(dfno_val_l2)
        log['drift_val_l2'].append(drift_val_l2)
        log['dfno_epoch_time'].append(dt_dfno)
        log['drift_epoch_time'].append(dt_drift)

        speedup = dt_dfno / dt_drift

        if rank == 0:
            print(f"  {epoch:4d}  | {dfno_epoch_loss:>10.4f} {dfno_val_l2:>9.4f} {dt_dfno:>7.1f}s | "
                  f"{drift_epoch_loss:>11.4f} {drift_val_l2:>9.4f} {dt_drift:>7.1f}s | {speedup:>7.1f}x")

    if rank == 0:
        dfno_times = np.array(log['dfno_epoch_time'])
        drift_times = np.array(log['drift_epoch_time'])
        total_speedup = dfno_times.sum() / drift_times.sum()


        print(f"  FINAL RESULTS ({args.epochs} epochs, P={ws})")
        print(f"  {'':28s} {'DFNO':>12s} {'DRIFT':>12s}")
        print(f"  {'-' * 55}")
        print(f"  {'Final train loss':28s} {log['dfno_train_loss'][-1]:>12.4f} {log['drift_train_loss'][-1]:>12.4f}")
        print(f"  {'Final test rel L2':28s} {log['dfno_val_l2'][-1]:>12.4f} {log['drift_val_l2'][-1]:>12.4f}")
        print(f"  {'Best test rel L2':28s} {min(log['dfno_val_l2']):>12.4f} {min(log['drift_val_l2']):>12.4f}")
        print(f"  {'Mean epoch time (s)':28s} {dfno_times.mean():>12.1f} {drift_times.mean():>12.1f}")
        print(f"  {'Total training time':28s} {dfno_times.sum():>11.1f}s {drift_times.sum():>11.1f}s")
        print(f"  {'Training speedup':28s} {'':>12s} {total_speedup:>11.1f}x")

        os.makedirs("results", exist_ok=True)
        tag = f"P{ws}_{nx}x{ny}x{nz}_{args.epochs}ep"

        fn_json = f"results/epochs_{tag}.json"
        with open(fn_json, 'w') as f:
            json.dump({
                "config": {
                    "world_size": ws, "grid": [nx, ny, nz],
                    "modes": args.modes, "width": args.width, "blocks": args.blocks,
                    "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
                    "n_train": n_train, "n_test": n_test,
                    "scheduler": args.scheduler,
                },
                **{k: v if not isinstance(v, np.ndarray) else v.tolist() for k, v in log.items()},
            }, f, indent=2)
        print(f"  Saved: {fn_json}")

        fn_npz = f"results/epochs_{tag}.npz"
        np.savez(fn_npz, **{k: np.array(v) for k, v in log.items()})
        print(f"  Saved: {fn_npz}")


if __name__ == "__main__":
    main()