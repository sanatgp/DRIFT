"""
DRIFT vs DFNO — PDEBench 3D Navier-Stokes Evaluation

Reports per-phase timings with named sections:
  DRIFT: pDFT(T), pDFT(Z), pDFT(YX), AllReduce, SpecConv, AllGather,
         iPDFT(XY), iPDFT(Z), iPDFT(T), Lift/Proj, Linear bypass, GeLU
  DFNO:  R1 (all-to-all), FFT1, R2 (all-to-all), FFT2, SpecConv,
         iFFT1, R3 (all-to-all), iFFT2, R4 (all-to-all), Lift/Proj

Saves results/eval_P{ws}.json with timings consumed by ab_validate.py.

Usage:
  mpirun -np 4  python eval_drift_vs_dfno.py --data-file ./data/ns3d_128x128x128_tin5_tout16.pt 
  mpirun -np 16 python eval_drift_vs_dfno.py --data-file ./data/ns3d_128x128x128_tin5_tout16.pt
"""

import argparse, json, os, time, math
import numpy as np
import torch
import torch.nn.functional as F
from mpi4py import MPI

from utils import create_standard_partitions
from dfno import DistributedFNO, BroadcastedLinear
from drift_block import PartialDFTFNOBlock


class Profiler:
    def __init__(self, device):
        self.use_cuda = device.type == "cuda"
        self.records = {}
        self._stack = []

    def _sync(self):
        if self.use_cuda:
            torch.cuda.synchronize()

    def start(self, name):
        self._sync()
        self._stack.append((name, time.perf_counter()))

    def stop(self):
        self._sync()
        name, t0 = self._stack.pop()
        dt = time.perf_counter() - t0
        self.records.setdefault(name, []).append(dt)
        return dt

    def record(self, name, value_sec):
        self.records.setdefault(name, []).append(value_sec)

    def summary(self):
        out = {}
        for name, vals in self.records.items():
            arr = np.array(vals)
            out[name] = dict(mean_ms=round(float(np.mean(arr)) * 1000, 4),
                             std_ms=round(float(np.std(arr)) * 1000, 4),
                             min_ms=round(float(np.min(arr)) * 1000, 4),
                             max_ms=round(float(np.max(arr)) * 1000, 4))
        return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", type=str, required=True)
    p.add_argument("--modes", type=int, nargs=4, default=[8, 8, 8, 8])
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--n-channels", type=int, default=None)
    p.add_argument("--save-viz", action="store_true",
                   help="Save z-midplane numpy slices for plot_correctness.py")
    p.add_argument("--no-save", action="store_true",
                   help="Skip saving results JSON (saved by default)")
    p.add_argument("--results-dir", default="results")
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


def profiled_dfno_block(block, x, prof, idx):
    p = f"dfno/blk{idx}"
    prof.start(f"{p}/linear"); y0 = block.linear(x); prof.stop()
    prof.record(f"{p}/linear_comm", block.linear.dt_comm)
    prof.start(f"{p}/R1"); x = block.R1(x); prof.stop()
    prof.start(f"{p}/fft1")
    saved = {}
    outermost = block.dim_m[-1]
    x = torch.fft.rfft(x, dim=outermost)
    saved[outermost] = list(x.shape)
    x = block.restrict(x, outermost)
    for dim in reversed(block.dim_m[:-1]):
        x = torch.fft.fft(x, dim=dim); saved[dim] = list(x.shape); x = block.restrict(x, dim)
    prof.stop()
    prof.start(f"{p}/R2"); x = block.R2(x); prof.stop()
    prof.start(f"{p}/fft2")
    for dim in reversed(block.dim_y):
        x = torch.fft.fft(x, dim=dim); saved[dim] = list(x.shape); x = block.restrict(x, dim)
    prof.stop()
    prof.start(f"{p}/spec_conv")
    y = 0 * x.clone()
    for w, sl in zip(block.weights, block.slices):
        y[sl] = torch.einsum(block.eqn, x[sl], w)
    prof.stop()
    prof.start(f"{p}/ifft1")
    for dim in block.dim_y:
        y = block.zeropad(y, dim, saved[dim]); y = torch.fft.ifft(y, dim=dim)
    prof.stop()
    prof.start(f"{p}/R3"); y = block.R3(y); prof.stop()
    prof.start(f"{p}/ifft2")
    for dim in block.dim_m[:-1]:
        y = block.zeropad(y, dim, saved[dim]); y = torch.fft.ifft(y, dim=dim)
    y = block.zeropad(y, outermost, saved[outermost]); y = torch.fft.irfft(y, dim=outermost)
    prof.stop()
    prof.start(f"{p}/R4"); y = block.R4(y); prof.stop()
    prof.start(f"{p}/gelu"); out = F.gelu(y0 + y); prof.stop()
    return out


def profiled_dfno_forward(model, x, prof):
    prof.start("dfno/total_fwd")
    prof.start("dfno/lift1"); out = F.gelu(model.linear1(x)); prof.stop()
    prof.record("dfno/lift1_comm", model.linear1.dt_comm)
    prof.start("dfno/lift2"); out = F.gelu(model.linear2(out)); prof.stop()
    prof.record("dfno/lift2_comm", model.linear2.dt_comm)
    for i, block in enumerate(model.blocks):
        out = profiled_dfno_block(block, out, prof, i)
    prof.start("dfno/proj1"); out = F.gelu(model.linear3(out)); prof.stop()
    prof.record("dfno/proj1_comm", model.linear3.dt_comm)
    prof.start("dfno/proj2"); out = model.linear4(out); prof.stop()
    prof.record("dfno/proj2_comm", model.linear4.dt_comm)
    prof.stop()
    return out


def profiled_drift_block(block, x, prof, idx):
    p = f"drift/blk{idx}"
    prof.start(f"{p}/linear"); y0 = block.linear(x); prof.stop()
    prof.record(f"{p}/linear_comm", block.linear.dt_comm)
    squeeze_z = False
    if x.ndim == 5: x = x.unsqueeze(4); squeeze_z = True
    B = x.shape[0]
    prof.start(f"{p}/pdft_t")
    orig_shape = x.shape
    x = torch.mm(x.reshape(-1, orig_shape[-1]).to(block.dtype_c), block.bt_fwd_T)
    x = x.view(*orig_shape[:-1], block.Kt)
    prof.stop()
    if block._has_z:
        prof.start(f"{p}/pdft_z")
        x = x.permute(0, 1, 2, 3, 5, 4).contiguous()
        sp = x.shape
        x = torch.mm(x.reshape(-1, sp[-1]), block.bz_fwd_T)
        x = x.view(*sp[:-1], block.Kz).permute(0, 1, 2, 3, 5, 4).contiguous()
        prof.stop()
    Kz, Kt = x.shape[4], x.shape[5]
    batch = B * block.width * Kz * Kt
    prof.start(f"{p}/pdft_yx")
    x_y = x.permute(0, 1, 4, 5, 2, 3).reshape(batch * block.local_x, block.local_y)
    s_y = torch.mm(x_y, block.by_T)
    s_y = s_y.view(batch, block.local_x, block.Ky).permute(0, 2, 1).reshape(batch * block.Ky, block.local_x)
    s_xy = torch.mm(s_y, block.bx_T)
    spectral = s_xy.view(batch, block.Ky, block.Kx).permute(0, 2, 1)
    spectral = spectral.view(B, block.width, Kz, Kt, block.Kx, block.Ky).permute(0, 1, 4, 5, 2, 3).contiguous()
    prof.stop()
    prof.start(f"{p}/allreduce")
    spectral_flat = spectral.view(B, block.width, block.S)
    if block.world_size > 1:
        from drift_block import _AllReduceFunc
        spectral_full = _AllReduceFunc.apply(spectral_flat, block.comm)
        spectral_local = spectral_full.narrow(2, block.rank * block.S_local, block.S_local).contiguous()
        prof.stop()
        prof.record(f"{p}/allreduce_comm", _AllReduceFunc._dt_mpi)
    else:
        spectral_local = spectral_flat; prof.stop()
    prof.start(f"{p}/spec_conv")
    x_bat = spectral_local.permute(2, 1, 0).contiguous()
    y_bat = torch.bmm(block.W_spec, x_bat)
    y_local = y_bat.permute(2, 1, 0).contiguous()
    prof.stop()
    prof.start(f"{p}/allgather")
    if block.world_size > 1:
        from drift_block import _AllGatherFunc
        y_full = _AllGatherFunc.apply(y_local, block.comm, 2)
        prof.stop()
        prof.record(f"{p}/allgather_comm", _AllGatherFunc._dt_mpi)
    else:
        y_full = y_local; prof.stop()
    y = y_full.view(B, block.width, *block.spec_spatial)
    prof.start(f"{p}/ipdft_xy")
    batch_inv = B * block.width * Kz * Kt
    y_inv = y.permute(0, 1, 4, 5, 3, 2).reshape(batch_inv * block.Ky, block.Kx)
    s_x = torch.mm(y_inv, block.ibx)
    s_x = s_x.view(batch_inv, block.Ky, block.local_x).permute(0, 2, 1).reshape(batch_inv * block.local_x, block.Ky)
    s_xy = torch.mm(s_x, block.iby)
    spatial = s_xy.view(batch_inv, block.local_x, block.local_y)
    spatial = spatial.view(B, block.width, Kz, Kt, block.local_x, block.local_y).permute(0, 1, 4, 5, 2, 3).contiguous()
    prof.stop()
    if block._has_z:
        prof.start(f"{p}/ipdft_z")
        spatial = spatial.permute(0, 1, 2, 3, 5, 4).contiguous()
        sp = spatial.shape
        spatial = torch.mm(spatial.reshape(-1, block.Kz), block.bz_inv)
        spatial = spatial.view(*sp[:-1], block.Z).permute(0, 1, 2, 3, 5, 4).contiguous()
        prof.stop()
    prof.start(f"{p}/ipdft_t")
    sp_shape = spatial.shape
    y_out = torch.mm(spatial.reshape(-1, block.Kt), block.bt_inv).real
    y_out = y_out.view(*sp_shape[:-1], block.T)
    if y_out.dtype != block.dtype: y_out = y_out.to(block.dtype)
    prof.stop()
    if squeeze_z: y_out = y_out.squeeze(4)
    prof.start(f"{p}/gelu"); out = F.gelu(y0 + y_out); prof.stop()
    return out


def profiled_drift_forward(model, x, prof):
    prof.start("drift/total_fwd")
    prof.start("drift/lift1"); out = F.gelu(model.lift_t(x)); prof.stop()
    prof.record("drift/lift1_comm", model.lift_t.dt_comm)
    prof.start("drift/lift2"); out = F.gelu(model.lift_c(out)); prof.stop()
    prof.record("drift/lift2_comm", model.lift_c.dt_comm)
    for i, block in enumerate(model.blocks):
        out = profiled_drift_block(block, out, prof, i)
    prof.start("drift/proj1"); out = F.gelu(model.proj1(out)); prof.stop()
    prof.record("drift/proj1_comm", model.proj1.dt_comm)
    prof.start("drift/proj2"); out = model.proj2(out); prof.stop()
    prof.record("drift/proj2_comm", model.proj2.dt_comm)
    prof.stop()
    return out


def rel_l2(a, b, comm):
    d = comm.allreduce(((a - b) ** 2).sum().item(), op=MPI.SUM)
    n = comm.allreduce((b ** 2).sum().item(), op=MPI.SUM)
    return math.sqrt(d / (n + 1e-12))


def reassemble_2d(chunks, Px, Py, nx, ny):
    if Py == 1:
        return np.concatenate(chunks, axis=1)
    lx = nx // Px
    ly = ny // Py
    rows = []
    for cx in range(Px):
        row_chunks = [chunks[cx * Py + cy] for cy in range(Py)]
        rows.append(np.concatenate(row_chunks, axis=2))
    return np.concatenate(rows, axis=1)


def aggregate_phases(summary, prefix, phase_tags, num_blocks):
    """Sum mean_ms across blocks for each listed phase tag."""
    out = {}
    for tag in phase_tags:
        total = 0.0
        for i in range(num_blocks):
            key = f"{prefix}/blk{i}/{tag}"
            if key in summary:
                total += summary[key]["mean_ms"]
        out[tag] = round(total, 2)
    return out


def main():
    args = parse_args()
    comm, rank, ws, device = setup()

    data = torch.load(args.data_file, weights_only=False, map_location='cpu')
    nx, ny, nz = data['grid']
    t_in, t_out = data['t_in'], data['t_out']
    x_full = data['x_test'][0:1].contiguous()
    if args.n_channels is not None and args.n_channels < x_full.shape[1]:
        x_full = x_full[:, :args.n_channels].contiguous()
    n_channels = x_full.shape[1]

    P_shape = make_partition(ws, [nx, ny, nz], args.modes[2])
    Px, Py = P_shape[2], P_shape[3]
    lx, ly = nx // Px, ny // Py

    if Py == 1:
        x_local = x_full[:, :, rank * lx:(rank + 1) * lx].contiguous().to(device)
    else:
        cx, cy = rank // Py, rank % Py
        x_local = x_full[:, :, cx * lx:(cx + 1) * lx, cy * ly:(cy + 1) * ly].contiguous().to(device)

    if rank == 0:
        print(f"DRIFT vs DFNO | P={ws} | Grid=({nx},{ny},{nz}) | "
              f"Modes={args.modes} | Width={args.width} | Blocks={args.blocks}")
        print(f"Partition: Px={Px}, Py={Py}, lx={lx}, ly={ly}")

    _, P_x, _ = create_standard_partitions(P_shape)
    in_shape = (1, n_channels, nx, ny, nz, t_in)

    torch.manual_seed(12345)
    dfno = DistributedFNO(P_x, in_shape, t_out, args.width, tuple(args.modes),
                          num_blocks=args.blocks, device=device, dtype=torch.float32)
    dfno.eval()

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
    drift.eval()

    with torch.no_grad():
        copy_blinear(drift.lift_t, dfno.linear1)
        copy_blinear(drift.lift_c, dfno.linear2)
        copy_blinear(drift.proj1, dfno.linear3)
        copy_blinear(drift.proj2, dfno.linear4)
        for i in range(len(dfno.blocks)):
            copy_blinear(drift.blocks[i].linear, dfno.blocks[i].linear)

    if args.save_viz:
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
        y_dfno = dfno(x_local)
        y_drift = drift(x_local)

    err = rel_l2(y_drift, y_dfno, comm)
    if rank == 0:
        print(f"Correctness: rel L2 = {err:.2e} ({'PASS' if err < 1e-5 else 'FAIL'})")

    if args.save_viz:
        y_d_list = comm.gather(y_dfno[0].cpu().numpy(), root=0)
        y_r_list = comm.gather(y_drift[0].cpu().numpy(), root=0)
        x_in_list = comm.gather(x_local[0].cpu().numpy(), root=0)
        if rank == 0:
            y_d_full = reassemble_2d(y_d_list, Px, Py, nx, ny)
            y_r_full = reassemble_2d(y_r_list, Px, Py, nx, ny)
            x_in_full = reassemble_2d(x_in_list, Px, Py, nx, ny)
            os.makedirs(args.results_dir, exist_ok=True)
            viz_fn = f"{args.results_dir}/viz_P{ws}_{nx}x{ny}x{nz}.npz"
            z_mid = nz // 2
            np.savez(viz_fn,
                     input=x_in_full[:, :, :, z_mid, 0],
                     dfno=y_d_full[:, :, :, z_mid, 0],
                     drift=y_r_full[:, :, :, z_mid, 0],
                     error=np.abs(y_d_full[:, :, :, z_mid, 0] - y_r_full[:, :, :, z_mid, 0]),
                     grid=[nx, ny, nz], rel_l2=err, Px=Px, Py=Py,
                     var_names=['density', 'vx', 'vy', 'vz', 'pressure'])
            print(f"Saved: {viz_fn}")
        dfno.linear4 = BroadcastedLinear(P_x, 128, 1, dim=1, device=device)
        drift.proj2 = BroadcastedLinear(P_x, 128, 1, dim=1, device=device)

    for _ in range(args.warmup):
        with torch.no_grad():
            _ = dfno(x_local)
            _ = drift(x_local)
    torch.cuda.synchronize(); comm.Barrier()

    prof = Profiler(device)
    for _ in range(args.trials):
        comm.Barrier()
        with torch.no_grad():
            profiled_dfno_forward(dfno, x_local, prof)
        y = dfno(x_local)
        comm.Barrier()
        prof.start("dfno/total_bwd"); y.sum().backward(); dfno.zero_grad(); prof.stop()
        comm.Barrier()
        with torch.no_grad():
            profiled_drift_forward(drift, x_local, prof)
        y = drift(x_local)
        comm.Barrier()
        prof.start("drift/total_bwd"); y.sum().backward(); drift.zero_grad(); prof.stop()
        comm.Barrier()

    summary = prof.summary()
    DFNO_COMM = ("/R1", "/R2", "/R3", "/R4", "_comm")
    DRIFT_COMM = ("/allreduce", "/allgather", "_comm")

    dfno_comm_keys = [k for k in summary if k.startswith("dfno/") and any(t in k for t in DFNO_COMM)]
    drift_comm_keys = [k for k in summary if k.startswith("drift/") and any(t in k for t in DRIFT_COMM)]

    dfno_fwd = summary["dfno/total_fwd"]["mean_ms"]
    dfno_bwd = summary["dfno/total_bwd"]["mean_ms"]
    dfno_comm = sum(summary[k]["mean_ms"] for k in dfno_comm_keys)
    drift_fwd = summary["drift/total_fwd"]["mean_ms"]
    drift_bwd = summary["drift/total_bwd"]["mean_ms"]
    drift_comm = sum(summary[k]["mean_ms"] for k in drift_comm_keys)
    sp_fwd = dfno_fwd / max(drift_fwd, 1e-9)
    sp_total = (dfno_fwd + dfno_bwd) / max(drift_fwd + drift_bwd, 1e-9)

    DRIFT_PHASES = [
        "linear", "pdft_t", "pdft_z", "pdft_yx",
        "allreduce", "spec_conv", "allgather",
        "ipdft_xy", "ipdft_z", "ipdft_t", "gelu",
    ]
    DFNO_PHASES = [
        "linear", "R1", "fft1", "R2", "fft2",
        "spec_conv", "ifft1", "R3", "ifft2", "R4", "gelu",
    ]
    drift_phases = aggregate_phases(summary, "drift", DRIFT_PHASES, args.blocks)
    dfno_phases = aggregate_phases(summary, "dfno", DFNO_PHASES, args.blocks)

    drift_liftproj = round(
        summary["drift/lift1"]["mean_ms"] + summary["drift/lift2"]["mean_ms"]
        + summary["drift/proj1"]["mean_ms"] + summary["drift/proj2"]["mean_ms"], 2)
    dfno_liftproj = round(
        summary["dfno/lift1"]["mean_ms"] + summary["dfno/lift2"]["mean_ms"]
        + summary["dfno/proj1"]["mean_ms"] + summary["dfno/proj2"]["mean_ms"], 2)

    if rank == 0:
        g = f"{nx}x{ny}x{nz}"
        print(f"\n{'='*60}")
        print(f"  RESULTS — P={ws}  Grid={g}  Modes={args.modes}")
        print(f"{'='*60}")
        print(f"  {'':28s} {'DFNO':>10s}   {'DRIFT':>10s}")
        print(f"  {'-'*52}")
        print(f"  {'Forward (ms)':28s} {dfno_fwd:>10.1f}   {drift_fwd:>10.1f}")
        print(f"  {'Backward (ms)':28s} {dfno_bwd:>10.1f}   {drift_bwd:>10.1f}")
        print(f"  {'Comm (ms)':28s} {dfno_comm:>10.1f}   {drift_comm:>10.1f}")
        print(f"  {'Comm %':28s} {100*dfno_comm/dfno_fwd:>9.1f}%   {100*drift_comm/drift_fwd:>9.1f}%")
        print(f"  {'-'*52}")
        print(f"  Forward speedup:  {sp_fwd:.1f}x")
        print(f"  Total speedup:    {sp_total:.1f}x")
        print(f"{'='*60}")

        print(f"\n  DRIFT per-phase (summed across {args.blocks} blocks, ms):")
        phase_labels_drift = {
            "pdft_t":    "pDFT (T)",
            "pdft_z":    "pDFT (Z)",
            "pdft_yx":   "pDFT (Y,X)",
            "allreduce": "AllReduce",
            "spec_conv": "Spectral conv",
            "allgather": "AllGather",
            "ipdft_xy":  "iPDFT (X,Y)",
            "ipdft_z":   "iPDFT (Z)",
            "ipdft_t":   "iPDFT (T)",
            "linear":    "Linear bypass",
            "gelu":      "GeLU",
        }
        for tag in DRIFT_PHASES:
            if drift_phases[tag] > 0:
                print(f"    {phase_labels_drift[tag]:<18s} {drift_phases[tag]:>8.2f}")
        print(f"    {'Lift + Proj':<18s} {drift_liftproj:>8.2f}")

        print(f"\n  DFNO per-phase (summed across {args.blocks} blocks, ms):")
        phase_labels_dfno = {
            "R1":        "R1 (all-to-all)",
            "fft1":      "FFT (m-dims)",
            "R2":        "R2 (all-to-all)",
            "fft2":      "FFT (y-dims)",
            "spec_conv": "Spectral conv",
            "ifft1":     "iFFT (y-dims)",
            "R3":        "R3 (all-to-all)",
            "ifft2":     "iFFT (m-dims)",
            "R4":        "R4 (all-to-all)",
            "linear":    "Linear bypass",
            "gelu":      "GeLU",
        }
        for tag in DFNO_PHASES:
            if dfno_phases[tag] > 0:
                print(f"    {phase_labels_dfno[tag]:<18s} {dfno_phases[tag]:>8.2f}")
        print(f"    {'Lift + Proj':<18s} {dfno_liftproj:>8.2f}")

        print(f"\n  Per-block comm/compute:")
        for i in range(args.blocks):
            for mname, ctags in [("DFNO", DFNO_COMM), ("DRIFT", DRIFT_COMM)]:
                pfx = f"{mname.lower()}/blk{i}"
                bk = [k for k in summary if k.startswith(pfx)]
                bc = sum(summary[k]["mean_ms"] for k in bk if any(t in k for t in ctags))
                bp = sum(summary[k]["mean_ms"] for k in bk if not any(t in k for t in ctags))
                print(f"    blk{i} {mname:<6} comm={bc:>7.1f}ms  compute={bp:>7.1f}ms")

        if not args.no_save:
            os.makedirs(args.results_dir, exist_ok=True)
            out_fn = f"{args.results_dir}/eval_P{ws}.json"
            payload = {
                "world_size": ws,
                "grid": [nx, ny, nz],
                "modes": args.modes,
                "width": args.width,
                "blocks": args.blocks,
                "Px": Px,
                "Py": Py,
                "trials": args.trials,
                "warmup": args.warmup,
                "rel_l2": err,
                "dfno_fwd_ms": dfno_fwd,
                "dfno_bwd_ms": dfno_bwd,
                "dfno_comm_ms": dfno_comm,
                "drift_fwd_ms": drift_fwd,
                "drift_bwd_ms": drift_bwd,
                "drift_comm_ms": drift_comm,
                "fwd_speedup": round(sp_fwd, 2),
                "total_speedup": round(sp_total, 2),
                "drift_phases": {**drift_phases, "lift_proj": drift_liftproj},
                "dfno_phases": {**dfno_phases, "lift_proj": dfno_liftproj},
                "raw_summary": summary,
            }
            with open(out_fn, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"\n  Saved: {out_fn}")


if __name__ == "__main__":
    main()