# %%
# !%matplotlib inline
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.time_frequency import psd_array_welch
from mne.viz.topomap import plot_topomap

from utils import get_colormap

# %%
# Configuration
preprocessed_dir = Path("results/preprocessed")
output_dir = Path("results/plots")
output_dir.mkdir(parents=True, exist_ok=True)

# Frequency bands to analyze
freq_bands = [
    # {"name": "Theta", "fmin": 6, "fmax": 8},
    {"name": "Alpha", "fmin": 8, "fmax": 12},
]

# Plot settings
use_zscore = True
vlim = (None, None)
vlim = (-4, 4) if use_zscore else (0, 1.8)
n_fft_multiplier = 2.5  # seconds for FFT window
cmap = get_colormap("parula")


# %%
def create_topomap_plots(raw, run_name, use_zscore=False):
    """Create topographic plots for frequency bands."""
    fig = plt.figure()
    fs = raw.info["sfreq"]

    for i, band in enumerate(freq_bands):
        ax = fig.add_subplot(1, len(freq_bands), i + 1)

        pxx = raw.compute_psd(
            method="welch",
            fmin=band["fmin"],
            fmax=band["fmax"],
            n_fft=int(fs * n_fft_multiplier),
        ).get_data()

        # Get mean power across frequencies
        power_values = pxx.mean(-1)

        # Apply z-score normalization if requested
        if use_zscore:
            power_values = (power_values - np.mean(power_values)) / np.std(power_values)
            plot_title_suffix = " (Z-score)"
        else:
            plot_title_suffix = " (Raw Power)"

        # Create topographic map
        im, cm = plot_topomap(
            power_values,
            raw.info,
            cmap=cmap,
            contours=False,
            show=False,
            vlim=vlim,
            axes=ax,
        )

        # Add colorbar for each subplot
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
        if use_zscore:
            cbar.set_label("Z-score", rotation=270, labelpad=15)
        else:
            cbar.set_label("Power (μV²/Hz)", rotation=270, labelpad=15)

        # Set title
        title = f"{band['name']} ({band['fmin']}-{band['fmax']} Hz){plot_title_suffix}"
        ax.set_title(title)

    fig.suptitle(run_name)

    # Save plot
    suffix = "_zscore" if use_zscore else "_power"
    plot_path = output_dir / f"{run_name}_topomap{suffix}.png"
    plt.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {plot_path}")

    return fig


# %%
# Find all preprocessed files
# fif_files = list(preprocessed_dir.glob("*_raw.fif.gz"))
fif_files = [
    Path("results/preprocessed/run0_raw.fif.gz"),
    Path("results/preprocessed/run1_raw.fif.gz"),
    Path("results/preprocessed/run2_raw.fif.gz"),
    Path("results/preprocessed/run3_raw.fif.gz"),
]

raws = []
for fif_file in fif_files:
    raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)
    raws.append(raw)

# %%
# Process each file
figures = []

for i, raw in enumerate(raws):
    # Extract run name from the file
    run_name = fif_files[i].stem.replace("_raw.fif", "")

    # Create plots
    fig = create_topomap_plots(raw, run_name, use_zscore)
    figures.append(fig)

print(f"\nProcessing complete! All plots saved to {output_dir}")

# %%
# Optional: Create a summary comparison plot
if len(figures) > 1:
    print("\nCreating summary comparison...")

    # Create a large figure with all runs
    n_runs = len(figures)
    n_bands = len(freq_bands)

    fig_summary = plt.figure(figsize=(n_bands * 4, n_runs * 3))

    plot_idx = 1
    for i, run_name in enumerate(figures):
        raw = raws[i]
        fs = raw.info["sfreq"]

        for j, band in enumerate(freq_bands):
            ax = fig_summary.add_subplot(n_runs, n_bands, plot_idx)

            # Calculate PSD
            pxx, f = psd_array_welch(
                raw._data,
                fmin=band["fmin"],
                fmax=band["fmax"],
                n_fft=int(fs * n_fft_multiplier),
                sfreq=fs,
            )

            # Get mean power and apply z-score if requested
            power_values = pxx.mean(-1)
            if use_zscore:
                power_values = (power_values - np.mean(power_values)) / np.std(
                    power_values
                )

            colormap = get_colormap("parula")

            # Create topomap
            im, cm = plot_topomap(
                power_values,
                raw.info,
                cmap=colormap,
                contours=False,
                show=False,
                vlim=vlim,
                axes=ax,
            )

            # Title
            ax.set_title(f"Run {i + 1}", fontsize=10)

            plot_idx += 1

    # Add a single colorbar
    fig_summary.subplots_adjust(right=0.85)
    cbar_ax = fig_summary.add_axes([0.87, 0.15, 0.02, 0.7])
    fig_summary.colorbar(im, cax=cbar_ax)

    plot_type = "Z-score" if use_zscore else "Power"
    cbar_ax.set_ylabel(f"{band['name']} ({plot_type})", rotation=270, labelpad=15)

    plot_type = "Z-score" if use_zscore else "Power"

    # Save summary
    suffix = "_zscore" if use_zscore else "_power"
    summary_path = output_dir / f"all_runs_comparison{suffix}.png"
    fig_summary.savefig(summary_path, dpi=300, bbox_inches="tight")
    print(f"Saved summary plot: {summary_path}")

    plt.show()

# %%
