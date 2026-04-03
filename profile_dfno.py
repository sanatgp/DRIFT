"""
Profiling script for Baseline Distributed FNO (DFNO).

Usage:
  mpirun -np 4  python profile_dfno.py
  mpirun -np 32 python profile_dfno.py
"""

import json, os, time
import numpy as np
import torch
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


def make_partition(world_size, modes_z):
    max_z = 2 * modes_z
    if world_size <= max_z:
        return (1, 1, world_size, 1, 1, 1)
    for px in range(max_z, 0, -1):
        if world_size % px != 0: continue
        py = world_size // px
        if SHAPE[0] % px != 0 or SHAPE[1] % py != 0: continue
        return (1, 1, px, py, 1, 1)
    raise ValueError(f"No valid partition for P={world_size}")


def profiled_block(block, x, prof, idx):
    p = f"blk{idx}"
    prof.start(f"{p}/linear"); y0 = block.linear(x); prof.stop()
    prof.record(f"{p}/linear_comm", block.linear.dt_comm)
    prof.start(f"{p}/R1"); x = block.R1(x); prof.stop()
    prof.start(f"{p}/fft1")
    saved = {}; outermost = block.dim_m[-1]
    x = torch.fft.rfft(x, dim=outermost); saved[outermost] = list(x.shape)
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
    for w, sl in zip(block.weights, block.slices): y[sl] = torch.einsum(block.eqn, x[sl], w)
    prof.stop()
    prof.start(f"{p}/ifft1")
    for dim in block.dim_y: y = block.zeropad(y, dim, saved[dim]); y = torch.fft.ifft(y, dim=dim)
    prof.stop()
    prof.start(f"{p}/R3"); y = block.R3(y); prof.stop()
    prof.start(f"{p}/ifft2")
    for dim in block.dim_m[:-1]: y = block.zeropad(y, dim, saved[dim]); y = torch.fft.ifft(y, dim=dim)
    y = block.zeropad(y, outermost, saved[outermost]); y = torch.fft.irfft(y, dim=outermost)
    prof.stop()
    prof.start(f"{p}/R4"); y = block.R4(y); prof.stop()
    prof.start(f"{p}/gelu"); out = F.gelu(y0 + y); prof.stop()
    return out


def profiled_forward(model, x, prof):
    prof.start("total_fwd")
    prof.start("lift1"); out = F.gelu(model.linear1(x)); prof.stop()
    prof.record("lift1_comm", model.linear1.dt_comm)
    prof.start("lift2"); out = F.gelu(model.linear2(out)); prof.stop()
    prof.record("lift2_comm", model.linear2.dt_comm)
    for i, block in enumerate(model.blocks): out = profiled_block(block, out, prof, i)
    prof.start("proj1"); out = F.gelu(model.linear3(out)); prof.stop()
    prof.record("proj1_comm", model.linear3.dt_comm)
    prof.start("proj2"); out = model.linear4(out); prof.stop()
    prof.record("proj2_comm", model.linear4.dt_comm)
    prof.stop()
    return out


def main():
    from dfno import DistributedFNO
    from utils import create_standard_partitions

    device = setup_device()
    ws = MPI.COMM_WORLD.Get_size()
    P_shape = make_partition(ws, MODES[2])
    _, P_x, _ = create_standard_partitions(P_shape)
    rank = P_x.rank
    Px, Py = P_shape[2], P_shape[3]
    local_shape = [SHAPE[0]//Px, SHAPE[1]//Py, SHAPE[2]]

    if rank == 0:
        print(f"DFNO Profiler | P={ws} | Grid={SHAPE} | Modes={MODES} | Width={WIDTH}")

    x = torch.rand(1, 1, *local_shape, 1, device=device, dtype=torch.float32)
    model = DistributedFNO(P_x, (1, 1, *SHAPE, 1), OUT_STEPS, WIDTH, tuple(MODES),
                           num_blocks=NUM_BLOCKS, device=device, dtype=x.dtype)
    prof = Profiler(device)

    for _ in range(WARMUP):
        y = model(x); y.sum().backward(); model.zero_grad(); P_x._comm.Barrier()

    for _ in range(TRIALS):
        P_x._comm.Barrier()
        y = profiled_forward(model, x, prof)
        P_x._comm.Barrier()
        prof.start("total_bwd"); y.sum().backward(); model.zero_grad(); prof.stop()
        P_x._comm.Barrier()

    summary = prof.summary()
    COMM = ("/R1", "/R2", "/R3", "/R4", "_comm")
    fwd = summary["total_fwd"]["mean_ms"]
    bwd = summary["total_bwd"]["mean_ms"]
    comm = sum(summary[k]["mean_ms"] for k in summary if any(t in k for t in COMM))

    if rank == 0:
        print(f"\n  Forward: {fwd:.1f}ms | Backward: {bwd:.1f}ms | Comm: {comm:.1f}ms ({100*comm/fwd:.1f}%)")
        for i in range(NUM_BLOCKS):
            rs = [f"{r}={summary[f'blk{i}/{r}']['mean_ms']:.1f}" for r in ["R1","R2","R3","R4"]]
            print(f"  blk{i}: {', '.join(rs)}")

#        os.makedirs("results", exist_ok=True)
#        fn = f"results/dfno_P{ws}.json"
#        with open(fn, 'w') as f:
#            json.dump({"fwd_ms": round(fwd,1), "bwd_ms": round(bwd,1),
#                       "comm_ms": round(comm,1), "phases": summary}, f, indent=2)

if __name__ == "__main__":
    main()