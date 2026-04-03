"""
Generate FNO datasets from PDEBench 3D Compressible Navier-Stokes.

Download data from: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986
Or use PDEBench download script:
  cd PDEBench/pdebench/data_download && python download_direct.py --pde_name 3D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_Train

Usage:
  python generate_ns3d_datasets.py --input 3D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5
  python generate_ns3d_datasets.py --input 3D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5 --t-in 5 --t-out 16
"""

import argparse, os, time
import h5py, numpy as np, torch, torch.fft


def spectral_upsample_3d(x, target_shape):
    nx, ny, nz = x.shape[-3:]
    tx, ty, tz = target_shape
    if (tx, ty, tz) == (nx, ny, nz):
        return x
    X = torch.fft.fftn(x.float(), dim=(-3, -2, -1))
    hx, hy, hz = (nx+1)//2, (ny+1)//2, (nz+1)//2
    s = list(X.shape); s[-3] = tx
    tmp = torch.zeros(s, dtype=X.dtype, device=x.device)
    tmp[..., :hx, :, :] = X[..., :hx, :, :]
    if nx - hx > 0: tmp[..., tx-(nx-hx):, :, :] = X[..., hx:, :, :]
    s[-2] = ty
    tmp2 = torch.zeros(s, dtype=X.dtype, device=x.device)
    tmp2[..., :hy, :] = tmp[..., :hy, :]
    if ny - hy > 0: tmp2[..., ty-(ny-hy):, :] = tmp[..., hy:, :]
    s[-1] = tz
    Xp = torch.zeros(s, dtype=X.dtype, device=x.device)
    Xp[..., :hz] = tmp2[..., :hz]
    if nz - hz > 0: Xp[..., tz-(nz-hz):] = tmp2[..., hz:]
    return torch.fft.ifftn(Xp, dim=(-3, -2, -1)).real * ((tx*ty*tz) / (nx*ny*nz))


def load_pdebench_3dcfd(path):
    with h5py.File(path, 'r') as f:
        arrays = [f[v][:] for v in ['density', 'Vx', 'Vy', 'Vz', 'pressure']]
    data = np.stack(arrays, axis=-1)
    N, T, nx, ny, nz, nv = data.shape
    print(f"Loaded: [N={N}, T={T}, {nx}x{ny}x{nz}, V={nv}]")
    return data


def make_fno_splits(data, t_in, t_out, train_frac=0.9):
    N, T, nx, ny, nz, nv = data.shape
    assert t_in + t_out <= T
    inp = np.transpose(data[:, :t_in], (0, 5, 2, 3, 4, 1))
    out = np.transpose(data[:, t_in:t_in+t_out], (0, 5, 2, 3, 4, 1))
    n_train = int(N * train_frac)
    return {
        'x_train': torch.from_numpy(inp[:n_train].copy()).float(),
        'y_train': torch.from_numpy(out[:n_train].copy()).float(),
        'x_test':  torch.from_numpy(inp[n_train:].copy()).float(),
        'y_test':  torch.from_numpy(out[n_train:].copy()).float(),
        'grid': (nx, ny, nz), 't_in': t_in, 't_out': t_out,
    }


def upsample_tensor(tensor, target, chunk_size=16):
    N, V, nx, ny, nz, T = tensor.shape
    tx, ty, tz = target
    if (nx, ny, nz) == (tx, ty, tz):
        return tensor
    results = []
    for i in range(0, N, chunk_size):
        batch = tensor[i:min(i+chunk_size, N)]
        B = batch.shape[0]
        flat = batch.reshape(B*V*T, nx, ny, nz)
        flat_up = spectral_upsample_3d(flat, target)
        batch_up = flat_up.reshape(B, V, T, tx, ty, tz).permute(0, 1, 3, 4, 5, 2)
        results.append(batch_up)
    return torch.cat(results, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--t-in", type=int, default=5)
    p.add_argument("--t-out", type=int, default=16)
    p.add_argument("--train-frac", type=float, default=0.9)
    p.add_argument("--out-dir", type=str, default="./data")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=16)
    p.add_argument("--resolutions", type=int, nargs='+', default=None,
                   help="Target resolutions (default: native only)")
    args = p.parse_args()

    assert os.path.exists(args.input), f"Not found: {args.input}"
    os.makedirs(args.out_dir, exist_ok=True)

    data = load_pdebench_3dcfd(args.input)
    if args.max_samples and args.max_samples < data.shape[0]:
        data = data[:args.max_samples]

    _, _, nx, ny, nz, _ = data.shape
    base = make_fno_splits(data, args.t_in, args.t_out, args.train_frac)

    resolutions = args.resolutions or [nx]

    for res in resolutions:
        target = (res, res, res)
        tag = f"{res}x{res}x{res}"
        out_path = os.path.join(args.out_dir, f"ns3d_{tag}_tin{args.t_in}_tout{args.t_out}.pt")

        if res == nx:
            save_dict = dict(base)
        else:
            t0 = time.time()
            save_dict = {
                'x_train': upsample_tensor(base['x_train'], target, args.chunk_size),
                'y_train': upsample_tensor(base['y_train'], target, args.chunk_size),
                'x_test':  upsample_tensor(base['x_test'], target, args.chunk_size),
                'y_test':  upsample_tensor(base['y_test'], target, args.chunk_size),
                'grid': target, 't_in': args.t_in, 't_out': args.t_out,
                'upsampled_from': (nx, ny, nz),
            }
            print(f"  Upsampled to {tag} in {time.time()-t0:.1f}s")

        torch.save(save_dict, out_path)
        print(f"Saved: {out_path} ({os.path.getsize(out_path)/1e9:.2f} GB)")
        print(f"  x_train: {list(save_dict['x_train'].shape)}")


if __name__ == "__main__":
    main()