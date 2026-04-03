#!/bin/bash
# Download PDEBench 3D Compressible Navier-Stokes dataset from DaRUS
# File: 3D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5 (83 GB)
# Source: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986

set -e
OUTDIR="${1:-./data}"
mkdir -p "$OUTDIR"
OUTFILE="$OUTDIR/3D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5"

if [ -f "$OUTFILE" ]; then
    echo "Already exists: $OUTFILE"
    exit 0
fi

FILE_URL=$(python3 -c "
import json, urllib.request
api = 'https://darus.uni-stuttgart.de/api/datasets/:persistentId/?persistentId=doi:10.18419/darus-2986'
data = json.loads(urllib.request.urlopen(api).read())
for f in data['data']['latestVersion']['files']:
    if '3D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_Train' in f['dataFile'].get('filename', ''):
        fid = f['dataFile']['id']
        print(f'https://darus.uni-stuttgart.de/api/access/datafile/{fid}')
        break
")

if [ -z "$FILE_URL" ]; then
    echo "ERROR: Could not find file ID. Download manually from:"
    echo "  https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986"
    exit 1
fi

echo "URL: $FILE_URL"
wget -c -O "$OUTFILE" "$FILE_URL"
