# DRIFT

**Direct Reduced Fourier Transforms for Distributed Spectral Neural Operators**

A communication-avoiding algorithm for distributed Fourier Neural Operators (FNOs). 
## Requirements

- NVIDIA GPUs with CUDA 12.x
- CUDA-aware MPI (HPC-X 2.19 / OpenMPI with UCX, built with CUDA support)
- Python 3.10+

Install Python dependencies:

```bash
pip install -r requirements.txt
```


## Data

All experiments use the PDEBench 3D compressible Navier-Stokes dataset.

**Download** the trajectory file (83 GB) from [DaRUS](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986):

```bash
bash download_data.sh ./data
```

**Generate FNO tensors** (loads the HDF5 file, stacks variables, writes train/test splits as a `.pt` file):

```bash
python generate_ns3d_datasets.py \
    --input ./data/3D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5 \
    --t-in 5 --t-out 16
```

This writes `./data/ns3d_128x128x128_tin5_tout16.pt`. Smaller grid variants used for strong/weak scaling are obtained by slicing or downsampling the native $128^3$ data at load time; no additional downloads are required.

## Experiments

### Correctness validation (Fig. 7)

Compares DRIFT's partial-DFT spectral coefficients against FFT+truncation, and the full DFNO and DRIFT models with matched non-spectral weights:

```bash
mpirun -np 16 python plot_correctness.py \
    --data-file ./data/ns3d_128x128x128_tin5_tout16.pt \
    --sample 4 --channel 1
```

Produces `fig_correctness.pdf` with the spectral coefficient comparison (relative Frobenius error) and the distributed full-model comparison.

### Performance benchmarks (Fig. 2, Table V, Figs. 8--11, Table I)

Runs both DFNO and DRIFT on the PDEBench data with 5 warm-up and 20 timed iterations per configuration, recording per-phase timings for each model (partial DFT, AllReduce, AllGather, spectral convolution, linear bypass, lift/projection for DRIFT; repartitions $R_1$--$R_4$, FFT, iFFT, spectral convolution, lift/projection for DFNO):

```bash
mpirun -np 4 python eval_drift_vs_dfno.py \
    --data-file ./data/ns3d_128x128x128_tin5_tout16.pt

mpirun -np 32 python eval_drift_vs_dfno.py \
    --data-file ./data/ns3d_128x128x128_tin5_tout16.pt 
```

Pass `--save-json` to write the per-phase breakdown to `results/phases_P{ws}_{grid}.json`.

### Training convergence (Fig. 12)

Trains DFNO and DRIFT for 100 epochs with matched Adam hyperparameters, matched random initialization, and identical per-epoch sample ordering. Logs per-epoch loss, test relative $L_2$ error, and wall-clock time:

```bash
mpirun -np 16 python train_convergence.py \
    --data-file ./data/ns3d_128x128x128_tin5_tout16.pt \
    --epochs 100 --lr 1e-3
```

Results are saved to `results/epochs_P{ws}_{grid}_{epochs}ep.{json,npz}`. Render Fig. 12 with:

```bash
python plot_training.py --input results/epochs_P16_128x128x128_100ep.npz \
    --save fig_training.pdf
```

## License

Released for research and educational use. See `LICENSE`.
