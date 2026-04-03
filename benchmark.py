"""
Simple timing + correctness: DFNO vs DRIFT forward pass.

Usage:
  mpirun -np 4  python benchmark.py
  mpirun -np 32 python benchmark.py
"""

import time, math, torch, torch.nn as nn, torch.nn.functional as F
from mpi4py import MPI
from dfno import DistributedFNO, BroadcastedLinear
from drift_block import PartialDFTFNOBlock
from utils import create_standard_partitions

SHAPE      = [128, 128, 64]
MODES      = [8, 8, 8, 16]
WIDTH      = 20
BLOCKS     = 4
OUT_STEPS  = 30
WARMUP     = 5
TRIALS     = 20

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
ws = comm.Get_size()
local_rank = rank % torch.cuda.device_count()
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

max_z = 2 * MODES[2]
if ws <= max_z:
    P_shape = (1, 1, ws, 1, 1, 1)
else:
    for px in range(max_z, 0, -1):
        if ws % px != 0: continue
        py = ws // px
        if SHAPE[0] % px != 0 or SHAPE[1] % py != 0: continue
        P_shape = (1, 1, px, py, 1, 1); break

_, P_x, _ = create_standard_partitions(P_shape)
Px, Py = P_shape[2], P_shape[3]
lx, ly = SHAPE[0]//Px, SHAPE[1]//Py
local_shape = [lx, ly, SHAPE[2]]

x = torch.rand(1, 1, *local_shape, 1, device=device)

dfno = DistributedFNO(P_x, (1, 1, *SHAPE, 1), OUT_STEPS, WIDTH, tuple(MODES),
                      num_blocks=BLOCKS, device=device, dtype=torch.float32)

class DRIFTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = BroadcastedLinear(P_x, 1, OUT_STEPS, dim=-1, device=device)
        self.l2 = BroadcastedLinear(P_x, 1, WIDTH, dim=1, device=device)
        self.blocks = nn.ModuleList([
            PartialDFTFNOBlock(P_x, (1, WIDTH, *local_shape, OUT_STEPS), MODES, device=device)
            for _ in range(BLOCKS)
        ])
        self.l3 = BroadcastedLinear(P_x, WIDTH, 128, dim=1, device=device)
        self.l4 = BroadcastedLinear(P_x, 128, 1, dim=1, device=device)

    def forward(self, x):
        x = F.gelu(self.l1(x))
        x = F.gelu(self.l2(x))
        for b in self.blocks: x = b(x)
        x = F.gelu(self.l3(x))
        return self.l4(x)

drift = DRIFTModel()

with torch.no_grad():
    def cp(dst, src):
        if src.P_root.active and dst.P_root.active:
            dst.W.data.copy_(src.W.data); dst.b.data.copy_(src.b.data)
    cp(drift.l1, dfno.linear1); cp(drift.l2, dfno.linear2)
    cp(drift.l3, dfno.linear3); cp(drift.l4, dfno.linear4)
    for i in range(BLOCKS): cp(drift.blocks[i].linear, dfno.blocks[i].linear)
    for blk in dfno.blocks:
        for w in blk.weights: w.data.zero_()
    for blk in drift.blocks: blk.W_spec.data.zero_()

with torch.no_grad():
    y_dfno = dfno(x); y_drift = drift(x)
d = comm.allreduce(((y_dfno - y_drift) ** 2).sum().item(), op=MPI.SUM)
n = comm.allreduce((y_dfno ** 2).sum().item(), op=MPI.SUM)
rel_l2 = math.sqrt(d / (n + 1e-12))
if rank == 0:
    print(f"Correctness: rel L2 = {rel_l2:.2e} ({'PASS' if rel_l2 < 1e-5 else 'FAIL'})")

for _ in range(WARMUP):
    with torch.no_grad(): dfno(x); drift(x)
torch.cuda.synchronize(); comm.Barrier()

def time_model(model, x, n):
    times = []
    for _ in range(n):
        comm.Barrier(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(): model(x)
        torch.cuda.synchronize(); comm.Barrier()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times) * 1000

dfno_ms = time_model(dfno, x, TRIALS)
drift_ms = time_model(drift, x, TRIALS)

if rank == 0:
    print(f"P={ws} | DFNO: {dfno_ms:.1f}ms | DRIFT: {drift_ms:.1f}ms | Speedup: {dfno_ms/drift_ms:.1f}x")
