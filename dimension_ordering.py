"""
Per-stage gemm FLOPs comparing current order (T,Z,Y,X) vs FLOP-optimal
order (Y,Z,X,T) at P=4. Grid 128 x 128 x 64, T=30, modes (8,8,8,16),
width dv=20. At P=4 with slab decomposition along x, local x = 128/4 = 32.

Usage:
  python dimension_ordering.py
"""

NX = 32     # local x at P=4 ("X:32->16")
NY = 128
NZ = 64
NT = 30
P = 4    

# K values (2*kmax for spatial, kt for time)
KX, KY, KZ, KT = 16, 16, 16, 16

WIDTH = 20

# Index: 0=X, 1=Y, 2=Z, 3=T
NAMES = ["X", "Y", "Z", "T"]


def gemm_flops_for_order(order):
    """Return list of (stage, dim_name, N_before, K, GFLOPs aggregate over P GPUs) for given order."""
    shape = [NX, NY, NZ, NT]
    K = [KX, KY, KZ, KT]

    stages = []
    for stage_num, idx in enumerate(order, 1):
        # Other dims (still at current size)
        other = WIDTH
        for j in range(4):
            if j != idx:
                other *= shape[j]
        # Per-GPU FLOPs * P GPUs = aggregate
        flops = 2 * K[idx] * shape[idx] * other * P
        stages.append((stage_num, NAMES[idx], shape[idx], K[idx], flops))
        # Contract this dim
        shape[idx] = K[idx]
    return stages


def main():
    # Current order: T -> Z -> Y -> X
    current = gemm_flops_for_order([3, 2, 1, 0])
    # Optimal order: Y -> Z -> X -> T
    optimal = gemm_flops_for_order([1, 2, 0, 3])

    print(f"Per-GPU local: X={NX}, Y={NY}, Z={NZ}, T={NT}  (P={P}, slab along x)")
    print(f"Modes: ({KX//2},{KY//2},{KZ//2},{KT}), width={WIDTH}")
    print(f"GFLOPs shown are aggregate over all {P} GPUs.")
    print()
    print(f"{'Stage':>5}  {'Current':<20} {'GFLOPs':>8}    {'Optimal':<20} {'GFLOPs':>8}")
    print("-" * 75)
    for (s, n_c, N_c, K_c, f_c), (_, n_o, N_o, K_o, f_o) in zip(current, optimal):
        print(f"{s:>5}  {n_c}:{N_c}->{K_c:<13} {f_c/1e9:>8.1f}    "
              f"{n_o}:{N_o}->{K_o:<13} {f_o/1e9:>8.1f}")
    print("-" * 75)
    total_c = sum(s[4] for s in current) / 1e9
    total_o = sum(s[4] for s in optimal) / 1e9
    print(f"{'':>5}  {'Total':<20} {total_c:>8.1f}    {'Total':<20} {total_o:>8.1f}")
    print()
    print(f"Savings: {(1 - total_o/total_c)*100:.1f}% fewer FLOPs with optimal order")


if __name__ == "__main__":
    main()
