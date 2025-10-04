# %%
# !%matplotlib inline
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.time_frequency import psd_array_welch
from mne.viz.topomap import plot_topomap

# Import palettable
try:
    import palettable

    print("Palettable imported successfully!")
except ImportError:
    print("Installing palettable...")
    import subprocess

    subprocess.check_call(["pip", "install", "palettable"])
    import palettable

# %%
# Configuration
preprocessed_dir = Path("results/preprocessed")
output_dir = Path("results/colormap_tests")
output_dir.mkdir(parents=True, exist_ok=True)

# Frequency bands to analyze
freq_bands = [
    {"name": "Alpha", "fmin": 8, "fmax": 12},
]

# Plot settings
use_zscore = True
vlim = (-2, 2) if use_zscore else (0, 1.2)
n_fft_multiplier = 2.5


# %%
def discover_all_palettable_colormaps():
    """Automatically discover ALL available colormaps in palettable."""
    import inspect

    all_colormaps = {}

    # Get all modules in palettable
    for module_name in dir(palettable):
        if module_name.startswith("_"):
            continue

        module = getattr(palettable, module_name)
        if not inspect.ismodule(module):
            continue

        print(f"Exploring {module_name}...")

        # Check for submodules (e.g., sequential, diverging, qualitative)
        for submodule_name in dir(module):
            if submodule_name.startswith("_"):
                continue

            submodule = getattr(module, submodule_name)
            if not inspect.ismodule(submodule):
                continue

            category_name = f"{module_name}_{submodule_name}"
            all_colormaps[category_name] = []

            print(f"  Found submodule: {submodule_name}")

            # Get all colormap objects in this submodule
            for attr_name in dir(submodule):
                if attr_name.startswith("_"):
                    continue

                attr = getattr(submodule, attr_name)

                # Check if it has mpl_colormap attribute (indicating it's a colormap)
                if hasattr(attr, "mpl_colormap"):
                    try:
                        colormap = attr.mpl_colormap
                        all_colormaps[category_name].append((attr_name, colormap))
                        print(f"    Found colormap: {attr_name}")
                    except Exception as e:
                        print(f"    Error with {attr_name}: {e}")
                        continue

            # Remove empty categories
            if not all_colormaps[category_name]:
                del all_colormaps[category_name]

    return all_colormaps


# %%
def create_topomap_with_colormap(
    raw, run_name, colormap_name, colormap, use_zscore=False
):
    """Create topographic plot with a specific colormap."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fs = raw.info["sfreq"]

    band = freq_bands[0]  # Use first frequency band

    # Compute PSD
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
        vlim_use = vlim
    else:
        plot_title_suffix = " (Raw Power)"
        vlim_use = (
            0,
            np.percentile(power_values, 95),
        )  # Use 95th percentile for better scaling

    # Create topographic map
    im, cm = plot_topomap(
        power_values,
        raw.info,
        cmap=colormap,
        contours=False,
        show=False,
        vlim=vlim_use,
        axes=ax,
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    if use_zscore:
        cbar.set_label("Z-score", rotation=270, labelpad=15)
    else:
        cbar.set_label("Power (μV²/Hz)", rotation=270, labelpad=15)

    # Set title
    title = f"{band['name']} ({band['fmin']}-{band['fmax']} Hz){plot_title_suffix}\\n{colormap_name}"
    ax.set_title(title)

    plt.tight_layout()
    return fig


# %%
# Load data (using just one file for colormap testing)
fif_file = Path("results/preprocessed/run1_raw.fif.gz")
raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)
run_name = fif_file.stem.replace("_raw.fif", "")

print(f"Loaded {run_name} with {len(raw.ch_names)} channels")

# %%
# Get all available colormaps
print("Discovering all palettable colormaps...")
colormap_dict = discover_all_palettable_colormaps()

print(f"\nFound {len(colormap_dict)} categories:")
for category, colormaps in colormap_dict.items():
    print(f"  {category}: {len(colormaps)} colormaps")

# Create plots for each colormap category
for category, colormaps in colormap_dict.items():
    print(f"\\nProcessing {category} colormaps...")

    # Create subfolder for this category
    category_dir = output_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)

    for colormap_name, colormap in colormaps:
        print(f"  Creating plot with {colormap_name}...")

        try:
            fig = create_topomap_with_colormap(
                raw, run_name, colormap_name, colormap, use_zscore
            )

            # Save plot
            suffix = "_zscore" if use_zscore else "_power"
            plot_path = category_dir / f"{run_name}_{colormap_name}{suffix}.png"
            fig.savefig(
                plot_path, dpi=150, bbox_inches="tight"
            )  # Lower DPI for faster generation
            plt.close(fig)  # Close to save memory

        except Exception as e:
            print(f"    Error with {colormap_name}: {e}")
            continue

print(f"\\nColormap testing complete! All plots saved to {output_dir}")
print(
    f"\\nGenerated plots for {sum(len(cmaps) for cmaps in colormap_dict.values())} colormaps"
)

# %%
# Create a more comprehensive overview
print("\\nCreating comprehensive overview...")

# Count total colormaps
total_colormaps = sum(len(cmaps) for cmaps in colormap_dict.values())
print(f"Total colormaps found: {total_colormaps}")

# Create a larger grid to accommodate all colormaps
cols = 8
rows = (total_colormaps + cols - 1) // cols  # Ceiling division

fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
if rows == 1:
    axes = axes.reshape(1, -1)
axes = axes.flatten()

plot_idx = 0
for category, colormaps in colormap_dict.items():
    for colormap_name, colormap in colormaps:
        if plot_idx >= len(axes):
            break

        ax = axes[plot_idx]

        try:
            # Compute data (simplified)
            band = freq_bands[0]
            pxx = raw.compute_psd(
                method="welch",
                fmin=band["fmin"],
                fmax=band["fmax"],
                n_fft=int(raw.info["sfreq"] * n_fft_multiplier),
            ).get_data()
            power_values = pxx.mean(-1)

            if use_zscore:
                power_values = (power_values - np.mean(power_values)) / np.std(
                    power_values
                )

            # Create small topomap
            im, cm = plot_topomap(
                power_values,
                raw.info,
                cmap=colormap,
                contours=False,
                show=False,
                vlim=vlim if use_zscore else None,
                axes=ax,
            )

            # Add category info to title
            ax.set_title(f"{colormap_name}\\n({category})", fontsize=6)
            plot_idx += 1

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error\\n{colormap_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=6,
            )
            ax.set_title(f"{colormap_name}\\n({category})", fontsize=6)
            plot_idx += 1

# Hide unused subplots
for i in range(plot_idx, 36):
    axes[i].set_visible(False)

plt.tight_layout()
overview_path = output_dir / "colormap_overview.png"
fig.savefig(overview_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Overview saved to {overview_path}")

# %%
