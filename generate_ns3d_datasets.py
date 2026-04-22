"""
Generate FNO datasets from PDEBench 3D Compressible Navier-Stokes.

Usage:
  python generate_ns3d_datasets.py --input 3D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5
  python generate_ns3d_datasets.py --input data.hdf5 --t-in 5 --t-out 16
"""

import argparse, os
import h5py, numpy as np, torch


def load_pdebench_3dcfd(path):
    with h5py.File(path, 'r') as f:
        arrays = [f[v][:] for v in ['density', 'Vx', 'Vy', 'Vz', 'pressure']]
    data = np.stack(arrays, axis=-1)
    N, T, nx, ny, nz, nv = data.shape
    print(f"Loaded: [N={N}, T={T}, {nx}x{ny}x{nz}, V={nv}]")
    return data


def make_fno_splits(data, t_in, t_out, train_frac=0.9):
    N, T, nx, ny, nz, nv = data.shape
    assert t_in + t_out <= T, f"t_in({t_in}) + t_out({t_out}) > T({T})"
    inp = np.transpose(data[:, :t_in], (0, 5, 2, 3, 4, 1))       # [N, V, nx, ny, nz, t_in]
    out = np.transpose(data[:, t_in:t_in+t_out], (0, 5, 2, 3, 4, 1))  # [N, V, nx, ny, nz, t_out]
    n_train = int(N * train_frac)
    return {
        'x_train': torch.from_numpy(inp[:n_train].copy()).float(),
        'y_train': torch.from_numpy(out[:n_train].copy()).float(),
        'x_test':  torch.from_numpy(inp[n_train:].copy()).float(),
        'y_test':  torch.from_numpy(out[n_train:].copy()).float(),
        'grid': (nx, ny, nz), 't_in': t_in, 't_out': t_out,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--t-in", type=int, default=5)
    p.add_argument("--t-out", type=int, default=16)
    p.add_argument("--train-frac", type=float, default=0.9)
    p.add_argument("--out-dir", type=str, default="./data")
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    assert os.path.exists(args.input), f"Not found: {args.input}"
    os.makedirs(args.out_dir, exist_ok=True)

    data = load_pdebench_3dcfd(args.input)
    if args.max_samples and args.max_samples < data.shape[0]:
        data = data[:args.max_samples]
        print(f"Using first {args.max_samples} samples")

    _, _, nx, ny, nz, _ = data.shape
    splits = make_fno_splits(data, args.t_in, args.t_out, args.train_frac)

    out_path = os.path.join(args.out_dir, f"ns3d_{nx}x{ny}x{nz}_tin{args.t_in}_tout{args.t_out}.pt")
    torch.save(splits, out_path)

    print(f"Saved: {out_path} ({os.path.getsize(out_path)/1e9:.2f} GB)")
    print(f"  x_train: {list(splits['x_train'].shape)}")
    print(f"  x_test:  {list(splits['x_test'].shape)}")


if __name__ == "__main__":
    main()