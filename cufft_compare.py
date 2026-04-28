"""
Computes per-stage gemm FLOPs with progressive compression for the
GLOBAL (undistributed) grid, plus the equivalent total cuFFT FLOPs,
and prints the gemm/cuFFT ratio.

Default config matches Table II:
  Grid 128 x 128 x 64, T=30, modes (8,8,8,16), width dv=20

Usage:
  python cufft_compare.py
"""

import math

NX, NY, NZ, NT = 128, 128, 64, 30
KX, KY, KZ, KT = 16, 16, 16, 16   # 2*kmax for spatial; kt for time
WIDTH = 20

# Per-stage FLOP factor: 2 * K * N * (other dims) * width
def gemm_stage_flops(N, K, other_dims_product, width):
    return 2 * K * N * other_dims_product * width

def main():
    # Stage order matches paper Table II: T -> Z -> Y -> X
    print(f"Config: {NX}x{NY}x{NZ}x{NT}  modes ({KX//2},{KY//2},{KZ//2},{KT})  width={WIDTH}")
    print()
    print(f"{'Stage':<7}{'Contracts':<14}{'Tensor shape':<28}{'gemm FLOPs':>12}{'Share':>10}")
    print("-" * 71)

    shape = [NX, NY, NZ, NT]   # X, Y, Z, T
    K = [KX, KY, KZ, KT]
    names = ["X", "Y", "Z", "T"]

    # Order: contract T first, then Z, then Y, then X
    order = [3, 2, 1, 0]

    stages = []
    for i, idx in enumerate(order, 1):
        # other dims (still at their current size)
        other = 1
        for j in range(4):
            if j != idx:
                other *= shape[j]
        flops = gemm_stage_flops(shape[idx], K[idx], other, WIDTH)
        tensor_str = f"{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}"
        contract_str = f"{shape[idx]} -> {K[idx]}"
        stages.append((i, names[idx], contract_str, tensor_str, flops))
        # Update shape: this dim is now contracted to K
        shape[idx] = K[idx]

    total_gemm = sum(s[4] for s in stages)
    for i, name, contract_str, tensor_str, flops in stages:
        share = 100 * flops / total_gemm
        print(f"{i} ({name}){'':<3}{contract_str:<14}{tensor_str:<28}"
              f"{flops/1e9:>10.1f} G{share:>9.0f}%")

    print("-" * 71)
    print(f"{'Total gemm':<49}{total_gemm/1e9:>10.1f} G")

    # cuFFT FLOPs: (5/2) N log2 N per 1D FFT, applied independently per dim
    cufft_total = 0
    full_shape = [NX, NY, NZ, NT]
    for d, N in enumerate(full_shape):
        other = 1
        for j in range(4):
            if j != d:
                other *= full_shape[j]
        cufft_total += 2.5 * N * math.log2(N) * other * WIDTH

    print(f"{'Total cuFFT (no compression)':<49}{cufft_total/1e9:>10.1f} G")

    ratio = total_gemm / cufft_total
    print(f"{'gemm / cuFFT ratio':<49}{ratio:>10.2f}x ({(1-ratio)*100:.0f}% fewer)")


if __name__ == "__main__":
    main()
