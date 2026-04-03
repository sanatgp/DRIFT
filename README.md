# DRIFT

**Direct Reduced Fourier Transforms for Distributed Spectral Neural Operators**

A communication-avoiding algorithm for distributed Fourier Neural Operators (FNOs).

## Requirements

- NVIDIA GPUs with CUDA
- CUDA-aware MPI (e.g., HPC-X, OpenMPI with UCX)
- Python 3.10+, PyTorch 2.0+, mpi4py, CuPy, distdl

Install dependencies:

    pip install -r requirements.txt

## Quick Start

Profile on synthetic data (no dataset needed):

    mpirun -np 4 python profile_dfno.py
    mpirun -np 4 python profile_drift.py

Evaluate on PDEBench 3D compressible Navier-Stokes:

    mpirun -np 4 python eval_drift_vs_dfno.py \
        --data-file ./data/ns3d_128x128x128_tin5_tout16.pt \
        --modes 8 8 8 8

Training convergence comparison:

    mpirun -np 4 python train_convergence.py \
        --data-file ./data/ns3d_128x128x128_tin5_tout16.pt \
        --iters 100 --lr 1e-3

## Data

Download PDEBench 3D compressible Navier-Stokes (83 GB):

    bash download_data.sh ./data

Or download `3D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5` manually from [DaRUS](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986).

Then generate FNO tensors:

    python generate_ns3d_datasets.py \
        --input ./data/3D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5
