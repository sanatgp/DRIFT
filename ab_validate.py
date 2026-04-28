"""
α-β Model Validation for DRIFT

Fits SEPARATE α, β for DFNO and DRIFT from measured comm times.

Usage:
  python ab_validate.py
  python ab_validate.py --results-dir results
"""

import argparse, json, os, math
import numpy as np

# Default config
NX, NY, NZ, NT = 128, 128, 128, 16
MODES = [8, 8, 8, 8]
WIDTH = 20
BATCH = 1
BLOCKS = 4
SIZEOF_COMPLEX = 8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--nx", type=int, default=NX)
    p.add_argument("--ny", type=int, default=NY)
    p.add_argument("--nz", type=int, default=NZ)
    p.add_argument("--nt", type=int, default=NT)
    p.add_argument("--modes", type=int, nargs=4, default=MODES)
    p.add_argument("--width", type=int, default=WIDTH)
    p.add_argument("--blocks", type=int, default=BLOCKS)
    p.add_argument("--output", default="results/ab_validation.pdf")
    return p.parse_args()


def load_results(d):
    r = {}
    if not os.path.exists(d):
        return r
    for fn in sorted(os.listdir(d)):
        if fn.startswith("eval_P") and fn.endswith(".json"):
            with open(os.path.join(d, fn)) as f:
                data = json.load(f)
                r[data["world_size"]] = data
    return r


# DFNO cost model: 4 all-to-all repartitions per block
# each: alpha*(P-1) + beta*N*c/P
def dfno_model(P, args, alpha, beta):
    N = args.nx * args.ny * args.nz * args.nt
    c = args.width * BATCH * SIZEOF_COMPLEX
    n = 4 * args.blocks
    T = n * (alpha * (P - 1) + beta * N * c / P)
    return T * 1000  # ms


def fit_dfno(meas, args):
    Ps = sorted(meas.keys())
    t_sec = np.array([meas[P]["dfno_comm_ms"] / 1000 for P in Ps])
    N = args.nx * args.ny * args.nz * args.nt
    c = args.width * BATCH * SIZEOF_COMPLEX
    n = 4 * args.blocks
    A = np.column_stack([
        [n * (P - 1) for P in Ps],
        [n * N * c / P for P in Ps]
    ])
    p, _, _, _ = np.linalg.lstsq(A, t_sec, rcond=None)
    return max(p[0], 1e-7), max(p[1], 1e-12)


# DRIFT cost model: Per block: ReduceScatter + AllGather
# RS:  floor(log2 P)*alpha + (P-1)/P * M*c * beta
# AG:  floor(log2 P)*alpha + (P-1)/P * M*c * beta
# Plus broadcasts for skip/lift/proj. each: floor(log2 P)*alpha + W_size * beta

def drift_model(P, args, alpha, beta):
    M = 1
    for m in args.modes:
        M *= 2 * m
    c = args.width * BATCH * SIZEOF_COMPLEX
    payload = M * c
    logP = math.floor(math.log2(max(P, 2)))

    # blocks * (RS + AG)
    T_blocks = args.blocks * (2 * logP * alpha + 2 * (P-1)/P * payload * beta)

    # broadcasts: 1 skip/block + 4 lift/proj
    n_bc = args.blocks + 4
    W = args.width * args.width * 4
    T_bc = n_bc * (logP * alpha + W * beta)

    return (T_blocks + T_bc) * 1000  # ms


def fit_drift(meas, args):
    Ps = sorted(meas.keys())
    t_sec = np.array([meas[P]["drift_comm_ms"] / 1000 for P in Ps])

    M = 1
    for m in args.modes:
        M *= 2 * m
    c = args.width * BATCH * SIZEOF_COMPLEX
    payload = M * c
    n_bc = args.blocks + 4
    W = args.width * args.width * 4

    # T = alpha * col1 + beta * col2
    col_alpha = []
    col_beta = []
    for P in Ps:
        logP = math.floor(math.log2(max(P, 2)))
        ca = args.blocks * 2 * logP + n_bc * logP
        cb = args.blocks * 2 * (P-1)/P * payload + n_bc * W
        col_alpha.append(ca)
        col_beta.append(cb)

    A = np.column_stack([col_alpha, col_beta])
    p, _, _, _ = np.linalg.lstsq(A, t_sec, rcond=None)
    return max(p[0], 1e-7), max(p[1], 1e-12)


def main():
    args = parse_args()
    meas = load_results(args.results_dir)

    if not meas:
        print(f"No results in {args.results_dir}/. Run eval_drift_vs_dfno.py first.")
        return
    if len(meas) < 2:
        print("Need >= 2 GPU counts to fit. Run eval at more P values.")
        return

    Ps = sorted(meas.keys())
    print(f"Loaded: P = {Ps}")

    # Fit separately
    ad, bd = fit_dfno(meas, args)
    ar, br = fit_drift(meas, args)

    print(f"\nDFNO fit:   α = {ad:.2e} s    β = {bd:.2e} s/byte   BW ≈ {1/bd/1e9:.1f} GB/s")
    print(f"DRIFT fit:  α = {ar:.2e} s    β = {br:.2e} s/byte   BW ≈ {1/br/1e9:.1f} GB/s")

    all_P = [4, 8, 16, 32, 64, 128, 256]
    print(f"\n{'='*80}")
    print(f"{'P':>5}  {'DFNO pred':>10} {'DFNO meas':>10}  "
          f"{'DRIFT pred':>11} {'DRIFT meas':>11}  {'Speedup':>8}")
    print(f"{'-'*72}")

    for P in all_P:
        dp = dfno_model(P, args, ad, bd)
        rp = drift_model(P, args, ar, br)
        sp = dp / max(rp, 0.01)

        dm = f"{meas[P]['dfno_comm_ms']:.1f}" if P in meas else "—"
        rm = f"{meas[P]['drift_comm_ms']:.1f}" if P in meas else "—"
        tag = "←" if P in meas else "(proj)"

        print(f"{P:>5}  {dp:>9.1f}ms {dm:>10}  "
              f"{rp:>10.1f}ms {rm:>11}  {sp:>7.1f}×  {tag}")

if __name__ == "__main__":
    main()
