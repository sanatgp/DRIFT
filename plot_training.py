"""
Plot DRIFT vs DFNO training convergence — 2-panel layout.
Left:  Training loss vs epoch
Right: Training loss vs cumulative wall-clock time

Usage:
  python plot_training.py --input results/epochs_P16_128x128x128_100ep.npz
  python plot_training.py --input results/epochs_P16_128x128x128_100ep.json --save fig_training.pdf
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.linewidth'] = 0.8
import matplotlib.pyplot as plt


def load_data(path):
    if path.endswith('.npz'):
        data = np.load(path)
        return {k: data[k] for k in data.files}
    elif path.endswith('.json'):
        with open(path) as f:
            raw = json.load(f)
        out = {}
        for k in ['dfno_train_loss', 'drift_train_loss',
                   'dfno_val_l2', 'drift_val_l2',
                   'dfno_epoch_time', 'drift_epoch_time']:
            if k in raw:
                out[k] = np.array(raw[k])
        if 'config' in raw:
            out['config'] = raw['config']
        return out
    else:
        raise ValueError(f"Unknown format: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    d = load_data(args.input)
    dfno_loss = d['dfno_train_loss']
    drift_loss = d['drift_train_loss']
    dfno_times = d['dfno_epoch_time']
    drift_times = d['drift_epoch_time']

    n_epochs = len(dfno_loss)
    epochs = np.arange(1, n_epochs + 1)

    dfno_cumtime_s = np.cumsum(dfno_times)
    drift_cumtime_s = np.cumsum(drift_times)

    dfno_cumtime_h = dfno_cumtime_s / 3600.0
    drift_cumtime_h = drift_cumtime_s / 3600.0

    dfno_total_h = dfno_cumtime_h[-1]
    drift_total_min = drift_cumtime_s[-1] / 60.0

    color_dfno = '#6B6B6B'  
    color_drift = '#1B3A5C'  

    fig, (ax_epoch, ax_time) = plt.subplots(1, 2, figsize=(10, 4.2),
                                             constrained_layout=True)

    ax_epoch.semilogy(epochs, dfno_loss, color=color_dfno, linewidth=1.5,
                      solid_capstyle='round', label='DFNO')
    ax_epoch.semilogy(epochs, drift_loss, color=color_drift, linewidth=1.5,
                      linestyle='--', dash_capstyle='round', label='DRIFT')
    ax_epoch.set_xlabel('Epoch', fontsize=15)
    ax_epoch.set_ylabel('MSE Loss', fontsize=15)
    ax_epoch.tick_params(axis='both', labelsize=13)
    ax_epoch.legend(frameon=False, fontsize=14, loc='upper right')
    ax_epoch.grid(True, alpha=0.2, linewidth=0.5)
    ax_epoch.set_xlim(1, n_epochs)
    ax_epoch.spines['top'].set_visible(False)
    ax_epoch.spines['right'].set_visible(False)

    ax_time.semilogy(dfno_cumtime_h, dfno_loss, color=color_dfno, linewidth=1.5,
                     solid_capstyle='round', label='DFNO')
    ax_time.semilogy(drift_cumtime_h, drift_loss, color=color_drift, linewidth=1.5,
                     linestyle='--', dash_capstyle='round', label='DRIFT')

    ax_time.set_xlabel('Wall-clock time (hours)', fontsize=15)
    ax_time.set_ylabel('MSE Loss', fontsize=15)
    ax_time.tick_params(axis='both', labelsize=13)
    ax_time.legend(frameon=False, fontsize=14, loc='upper right')
    ax_time.grid(True, alpha=0.2, linewidth=0.5)
    ax_time.spines['top'].set_visible(False)
    ax_time.spines['right'].set_visible(False)

    ax_time.annotate(
        f'DFNO: {dfno_total_h:.1f}h',
        xy=(dfno_cumtime_h[-1], dfno_loss[-1]),
        xytext=(dfno_cumtime_h[-1] * 0.55, dfno_loss[-1] * 8),
        fontsize=14, color=color_dfno,
        arrowprops=dict(arrowstyle='->', color=color_dfno, lw=1.0),
    )

    ax_time.annotate(
        f'DRIFT: {drift_total_min:.0f}min',
        xy=(drift_cumtime_h[-1], drift_loss[-1]),
        xytext=(dfno_cumtime_h[-1] * 0.25, drift_loss[-1] * 15),
        fontsize=14, color=color_drift,
        arrowprops=dict(arrowstyle='->', color=color_drift, lw=1.0),
    )

    if args.save:
        fig.savefig(args.save, dpi=args.dpi, bbox_inches='tight')
        print(f"Saved: {args.save}")
    else:
        plt.show()

if __name__ == "__main__":
    main()