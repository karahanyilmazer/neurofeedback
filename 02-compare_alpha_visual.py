# %%
# !%matplotlib inline
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from mne._fiff.pick import _pick_data_channels, pick_info
from mne.channels.layout import _find_topomap_coords
from mne.viz.topomap import plot_topomap

from utils import get_colormap


def highlight_channels_on_topomap(
    ax,
    raw,
    channel_names,
    marker="ro",
    markersize=8,
    markeredgecolor="white",
    markeredgewidth=1,
):
    """Highlight channels on topomap using exact MNE positioning."""

    # Get all data channel picks and their names
    all_picks = _pick_data_channels(raw.info, exclude=())
    all_channel_names = [raw.ch_names[pick] for pick in all_picks]

    # Find which of our requested channels are available
    available_channels = [ch for ch in channel_names if ch in all_channel_names]
    if not available_channels:
        return {}

    # Get indices of only the channels we need
    needed_indices = [all_channel_names.index(ch) for ch in available_channels]
    needed_picks = [all_picks[i] for i in needed_indices]

    # Create minimal info object with only needed channels
    pos_info = pick_info(raw.info, needed_picks)
    channel_picks = list(range(len(pos_info["chs"])))
    pos = _find_topomap_coords(pos_info, picks=channel_picks)

    # Plot the channels
    highlighted = {}
    for i, ch_name in enumerate(available_channels):
        pos_xy = pos[i]
        ax.plot(
            pos_xy[0],
            pos_xy[1],
            marker,
            markersize=markersize,
            markeredgecolor=markeredgecolor,
            markeredgewidth=markeredgewidth,
        )
        highlighted[ch_name] = pos_xy

    return highlighted


# %%
# Configuration
preprocessed_dir = Path("results/preprocessed")
output_dir = Path("results/alpha_comparison")
output_dir.mkdir(parents=True, exist_ok=True)

# Alpha frequency band
alpha_band = {"name": "Alpha", "fmin": 8, "fmax": 12}
n_fft_multiplier = 2.5

# Visual/occipital channels (typical locations for alpha activity)
visual_channels = ["O1", "O2", "Oz", "P3", "P4", "Pz", "P7", "P8"]
# Backup channels if some are missing
backup_visual = ["PO3", "PO4", "POz", "P1", "P2", "CP1", "CP2"]


USE_ZSCORE = True
VLIM = (None, None)
VLIM = (-4, 4) if USE_ZSCORE else (0, 1.8)
CMAP = get_colormap("parula")
RUNS = {
    "run0": "Baseline",
    "run1": "Day 5",
    "run2": "Day 8 (Before NF)",
    "run3": "Day 8 (After NF)",
}


# %%
def get_available_visual_channels(raw):
    """Get available visual channels from the raw data."""
    available_channels = raw.ch_names

    # Primary visual channels
    visual_present = [ch for ch in visual_channels if ch in available_channels]

    # Add backup channels if needed
    if len(visual_present) < 4:  # If we have fewer than 4 visual channels
        backup_present = [
            ch
            for ch in backup_visual
            if ch in available_channels and ch not in visual_present
        ]
        visual_present.extend(backup_present)

    print(f"Available visual channels: {visual_present}")
    return visual_present


def get_channel_position_robust(raw, channel_name):
    """
    Robustly get 2D channel position for topomap plotting.

    Parameters:
    -----------
    raw : mne.io.Raw
        Raw object containing channel information
    channel_name : str
        Name of the channel to get position for

    Returns:
    --------
    pos : array-like or None
        2D position [x, y] or None if position cannot be determined
    """
    if channel_name not in raw.ch_names:
        return None

    ch_idx = raw.ch_names.index(channel_name)

    # Method 1: Try from channel info loc field
    try:
        if raw.info["chs"][ch_idx]["loc"] is not None:
            loc = raw.info["chs"][ch_idx]["loc"]
            if (
                len(loc) >= 2
                and not np.isnan(loc[:2]).any()
                and not np.allclose(loc[:2], 0)
            ):
                return loc[:2]
    except (IndexError, TypeError, KeyError):
        pass

    # Method 2: Try from montage if available
    try:
        montage = raw.info.get_montage()
        if montage is not None and channel_name in montage.ch_names:
            ch_montage_idx = montage.ch_names.index(channel_name)
            # Account for fiducial points in dig
            dig_idx = ch_montage_idx + 3  # Skip nasion, lpa, rpa
            if dig_idx < len(montage.dig):
                dig_point = montage.dig[dig_idx]
                if dig_point["r"] is not None and len(dig_point["r"]) >= 2:
                    return dig_point["r"][:2]
    except (IndexError, TypeError, KeyError, AttributeError):
        pass

    # Method 3: Try to get from topomap transformation
    try:
        picks = mne.pick_types(raw.info, eeg=True)
        if ch_idx in picks:
            pos_array, outlines = mne.viz.topomap._prepare_topomap_plot(
                raw.info, "eeg", sphere=None
            )
            pick_idx = np.where(picks == ch_idx)[0]
            if len(pick_idx) > 0 and pick_idx[0] < len(pos_array):
                return pos_array[pick_idx[0]]
    except Exception:
        pass

    # Method 4: Try standard montage as fallback
    try:
        standard_montage = mne.channels.make_standard_montage("standard_1020")
        if channel_name in standard_montage.ch_names:
            ch_montage_idx = standard_montage.ch_names.index(channel_name)
            dig_idx = ch_montage_idx + 3  # Skip fiducials
            if dig_idx < len(standard_montage.dig):
                dig_point = standard_montage.dig[dig_idx]
                if dig_point["r"] is not None and len(dig_point["r"]) >= 2:
                    return dig_point["r"][:2]
    except Exception:
        pass

    return None


def compute_alpha_power_all(raw):
    """Compute alpha power for all channels, z-score, then select visual channels."""
    # Compute PSD for alpha band for all channels
    psd = raw.compute_psd(
        method="welch",
        fmin=alpha_band["fmin"],
        fmax=alpha_band["fmax"],
        n_fft=int(raw.info["sfreq"] * n_fft_multiplier),
    )

    # Get power data (channels x frequencies)
    power_data = psd.get_data()

    # Average across frequencies to get mean alpha power per channel
    alpha_power_all = power_data.mean(axis=-1)

    # Z-score normalization across all channels
    if USE_ZSCORE:
        alpha_power_all = (alpha_power_all - np.mean(alpha_power_all)) / np.std(
            alpha_power_all
        )

    # Get available visual channels
    available_visual = get_available_visual_channels(raw)
    # Indices of visual channels in raw
    visual_indices = [raw.ch_names.index(ch) for ch in available_visual]
    # Select visual channel powers
    alpha_power_visual = alpha_power_all[visual_indices]

    return alpha_power_visual, available_visual, alpha_power_all


# %%
# Load all available preprocessed files
# fif_files = list(preprocessed_dir.glob("*_raw.fif.gz"))
fif_files = [
    Path("results/preprocessed/run0_raw.fif.gz"),
    Path("results/preprocessed/run1_raw.fif.gz"),
    Path("results/preprocessed/run2_raw.fif.gz"),
    Path("results/preprocessed/run3_raw.fif.gz"),
]

print(f"Found {len(fif_files)} preprocessed files:")
for f in fif_files:
    print(f"  - {f.name}")

# %%
# Process each file and collect alpha power data
alpha_data = {}
all_visual_channels = set()


for fif_file in sorted(fif_files):
    run_name = RUNS[fif_file.stem.replace("_raw.fif", "")]
    print(f"\nProcessing {run_name}...")

    # Load data
    raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)

    # Compute alpha power for all channels, then select visual channels
    alpha_power_visual, visual_ch, alpha_power_all = compute_alpha_power_all(raw)
    all_visual_channels.update(visual_ch)

    # Store data
    alpha_data[run_name] = {
        "power": alpha_power_visual,
        "channels": visual_ch,
        "raw": raw,  # Keep for topographic plots
        "power_all": alpha_power_all,  # For topomap
    }

    print(f"  Mean visual alpha power: {alpha_power_visual.mean():.4f} (z-score)")
    print(f"  Visual channels: {visual_ch}")

all_visual_channels = sorted(list(all_visual_channels))

# %%
# Create comparison plots

# Colorbar placement
COLORBAR_MODE = "shared"  # "shared" or "individual"

fig, axes = plt.subplots(1, len(alpha_data), figsize=(5 * len(alpha_data), 6))
topomaps = []

# 1. Topographic comparison
for i, (run_name, data) in enumerate(alpha_data.items()):
    ax = axes[i]
    power_all = data["power_all"]
    im, cm = plot_topomap(
        power_all,
        data["raw"].info,
        cmap=CMAP,
        contours=False,
        vlim=VLIM,
        show=False,
        axes=ax,
    )
    topomaps.append(im)

    # Highlight visual channels using exact topomap positioning
    highlighted = highlight_channels_on_topomap(
        ax,
        data["raw"],
        data["channels"],
        marker="ro",
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=1,
    )

    # Report any channels that couldn't be positioned
    missing_channels = set(data["channels"]) - set(highlighted.keys())
    if missing_channels:
        print(f"Warning: Could not position channels {missing_channels} in {run_name}")

    ax.set_title(f"{run_name}")

# Colorbar options
if COLORBAR_MODE == "individual":
    for i, im in enumerate(topomaps):
        fig.colorbar(
            im,
            ax=axes[i],
            orientation="horizontal",
            fraction=0.046,
            pad=0.15,
            label="Alpha Power (z-score)",
        )
elif COLORBAR_MODE == "shared":
    # Shared colorbar below all axes
    cbar = fig.colorbar(
        topomaps[0],
        ax=axes,
        orientation="horizontal",
        fraction=0.07,
        pad=0.15,
        label="Alpha Power (z-score)",
    )

fig.suptitle("Alpha Power (z-score)\n(Visual channels marked in red)")
# plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(
    output_dir / "alpha_topographic_comparison.png", dpi=300, bbox_inches="tight"
)
plt.show()

# %%
# 2. Bar plot comparison of average alpha power per run
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Overall comparison (violin plot for z-scored data)
run_names = list(alpha_data.keys())
df_data = []
for run_name, data in alpha_data.items():
    for ch, power in zip(data["channels"], data["power"]):
        df_data.append({"Run": run_name, "Channel": ch, "Alpha_Power": power})
df = pd.DataFrame(df_data)

if len(df) > 0:
    sns.violinplot(
        data=df, x="Run", y="Alpha_Power", ax=ax1, inner="box", palette="pastel"
    )
    ax1.set_ylabel("Alpha Power (z-score)")
    ax1.set_title("Distribution of Alpha Power (z-score) in Visual Channels")
    ax1.grid(axis="y", alpha=0.3)

# Channel-wise comparison
# Create a DataFrame for easier plotting
df_data = []
for run_name, data in alpha_data.items():
    for ch, power in zip(data["channels"], data["power"]):
        df_data.append({"Run": run_name, "Channel": ch, "Alpha_Power": power})

df = pd.DataFrame(df_data)

# Box plot by channel
if len(df) > 0:
    sns.boxplot(data=df, x="Channel", y="Alpha_Power", hue="Run", ax=ax2)
    ax2.set_title("Alpha Power Distribution by Channel (z-score)")
    ax2.set_ylabel("Alpha Power (z-score)")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "alpha_power_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
# 3. Statistical comparison
if len(alpha_data) >= 2:
    print("\n" + "=" * 50)
    print("STATISTICAL COMPARISON")
    print("=" * 50)

    # Overall comparison between runs
    from scipy import stats

    run_pairs = [
        (r1, r2) for i, r1 in enumerate(run_names) for r2 in run_names[i + 1 :]
    ]

    for r1, r2 in run_pairs:
        power1 = alpha_data[r1]["power"]
        power2 = alpha_data[r2]["power"]

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(power1, power2)

        print(f"\n{r1} vs {r2}:")
        print(f"  {r1}: {power1.mean():.4f} ± {power1.std():.4f} μV²/Hz")
        print(f"  {r2}: {power2.mean():.4f} ± {power2.std():.4f} μV²/Hz")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")

    # Effect sizes (Cohen's d)
    def cohens_d(x1, x2):
        n1, n2 = len(x1), len(x2)
        s1, s2 = x1.std(ddof=1), x2.std(ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        return (x1.mean() - x2.mean()) / pooled_std

    print("\nEffect Sizes (Cohen's d):")
    for r1, r2 in run_pairs:
        d = cohens_d(alpha_data[r1]["power"], alpha_data[r2]["power"])
        magnitude = "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
        print(f"  {r1} vs {r2}: d = {d:.3f} ({magnitude})")

# %%
# 4. Detailed channel analysis

# Add Cz channel to the DataFrame if present in each run
cz_rows = []
for run_name, data in alpha_data.items():
    raw = data["raw"]
    names = raw.ch_names
    if "Cz" in names:
        cz_power = data["power_all"][names.index("Cz")]
        cz_rows.append({"Run": run_name, "Channel": "Cz", "Alpha_Power": cz_power})

# Append Cz rows to df
if cz_rows:
    df = pd.concat([df, pd.DataFrame(cz_rows)])

fig, ax = plt.subplots(1, 1, figsize=(12, 8))


# Create a heatmap of alpha power by channel and run
if len(df) > 0:
    pivot_df = df.pivot(index="Channel", columns="Run", values="Alpha_Power")
    # Rearrange columns to only swap the last two columns
    pivot_df = pivot_df.reindex(
        columns=pivot_df.columns.tolist()[:-2] + pivot_df.columns.tolist()[-2:][::-1]
    )
    # Ensure Cz is at the bottom
    idx = [ch for ch in pivot_df.index if ch != "Cz"]
    if "Cz" in pivot_df.index:
        idx.append("Cz")
    pivot_df = pivot_df.reindex(idx)
    sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap=CMAP, ax=ax)
    ax.set_title(
        "Alpha Power Heatmap (z-score): Visual Channels (incl. Cz) Across Days"
    )
    ax.set_ylabel("Channel")
    ax.set_xlabel("Day")

plt.tight_layout()
plt.savefig(output_dir / "alpha_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
# 5. Summary report
print("\n" + "=" * 60)
print("ALPHA ACTIVITY COMPARISON SUMMARY")
print("=" * 60)

print(f"\nAnalysis performed on {len(alpha_data)} runs")
print(f"Visual channels analyzed: {all_visual_channels}")
print(f"Alpha frequency band: {alpha_band['fmin']}-{alpha_band['fmax']} Hz")


print(f"\nMean Alpha Power (z-score) by Day:")
for run_name in sorted(alpha_data.keys()):
    power = alpha_data[run_name]["power"]
    print(f"  {run_name}: {power.mean():.4f} ± {power.std():.4f} (z-score)")

# Find the most active visual channel per day
print("\nMost Active Visual Channel by Day (z-score):")
for run_name, data in alpha_data.items():
    max_idx = np.argmax(data["power"])
    max_channel = data["channels"][max_idx]
    max_power = data["power"][max_idx]
    print(f"  {run_name}: {max_channel} ({max_power:.4f} z-score)")

# %%
# --- Highest/Lowest z-score channel comparison and t-test ---
print("\n" + "=" * 60)
print("CHANNELS WITH HIGHEST AND LOWEST Z-SCORES (across all days)")
print("=" * 60)

# Restrict to visual channels plus Cz
visual_plus_cz = set(all_visual_channels) | {"Cz"}

# Gather all z-scores and names for visual channels + Cz for all runs
all_vis_scores = []
all_vis_names = []
for run_name, data in alpha_data.items():
    raw = data["raw"]
    scores = data["power_all"]
    names = raw.ch_names
    for ch, score in zip(names, scores):
        if ch in visual_plus_cz:
            all_vis_scores.append(score)
            all_vis_names.append(ch)

# Find channel with highest and lowest z-score (across all runs, restricted)
max_idx = np.argmax(all_vis_scores)
min_idx = np.argmin(all_vis_scores)
max_channel = all_vis_names[max_idx]
min_channel = all_vis_names[min_idx]
print("=" * 60)

# Find highest and lowest visual cortex channels (across all runs)
visual_only = set(all_visual_channels)
visual_scores = {ch: [] for ch in visual_only}
for run_name, data in alpha_data.items():
    raw = data["raw"]
    scores = data["power_all"]
    names = raw.ch_names
    for ch in visual_only:
        if ch in names:
            visual_scores[ch].append(scores[names.index(ch)])
        else:
            visual_scores[ch].append(np.nan)

# Compute mean z-score for each visual channel
visual_means = {ch: np.nanmean(vals) for ch, vals in visual_scores.items()}
max_visual_ch = max(visual_means, key=visual_means.get)
min_visual_ch = min(visual_means, key=visual_means.get)
print(
    f"Highest z-score visual channel: {max_visual_ch} ({visual_means[max_visual_ch]:.4f})"
)
print(
    f"Lowest z-score visual channel: {min_visual_ch} ({visual_means[min_visual_ch]:.4f})"
)

# Always include Cz
cz_scores = []
for run_name, data in alpha_data.items():
    raw = data["raw"]
    names = raw.ch_names
    if "Cz" in names:
        cz_scores.append(data["power_all"][names.index("Cz")])
    else:
        cz_scores.append(np.nan)

# Gather per-day values for these three channels
# Gather per-day values for these three channels
max_ch_values = []
min_ch_values = []
days = []
for run_name, data in alpha_data.items():
    raw = data["raw"]
    ch_names = raw.ch_names
    if max_visual_ch in ch_names:
        max_ch_values.append(data["power_all"][ch_names.index(max_visual_ch)])
    else:
        max_ch_values.append(np.nan)
    if min_visual_ch in ch_names:
        min_ch_values.append(data["power_all"][ch_names.index(min_visual_ch)])
    else:
        min_ch_values.append(np.nan)
    days.append(run_name)

colors = get_colormap("Set1_3", as_colors=True)

# Plot comparison
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(
    days,
    max_ch_values,
    marker="o",
    label=f"{max_visual_ch} (Largest Activity)",
    color=colors[0],
)
ax.plot(
    days,
    min_ch_values,
    marker="o",
    label=f"{min_visual_ch} (Smallest Activity)",
    color=colors[1],
)
ax.plot(
    days,
    cz_scores,
    marker="o",
    label=f"Cz (NF Channel)",
    color=colors[2],
)
ax.set_title("Alpha Power (z-score) over Visual Cortex and Cz")
ax.set_ylabel("Alpha Power (z-score)")
ax.set_xlabel("Day")
ax.legend()
plt.tight_layout()
plt.savefig(output_dir / "highest_lowest_visual_cz_channel_comparison.png", dpi=300)
plt.show()

# T-test between days for these channels
from scipy import stats

print("\nT-TESTS FOR HIGHEST/LOWEST VISUAL AND CZ CHANNELS:")
for i in range(len(days)):
    for j in range(i + 1, len(days)):
        d1, d2 = days[i], days[j]
        # Highest visual channel
        vals1 = max_ch_values[i]
        vals2 = max_ch_values[j]
        print(f"\n{max_visual_ch} ({d1} vs {d2}):")
        print(f"  {d1}: {vals1:.4f}, {d2}: {vals2:.4f}")
        print(f"  Difference: {vals1 - vals2:.4f}")
        # Lowest visual channel
        vals1 = min_ch_values[i]
        vals2 = min_ch_values[j]
        print(f"{min_visual_ch} ({d1} vs {d2}):")
        print(f"  {d1}: {vals1:.4f}, {d2}: {vals2:.4f}")
        print(f"  Difference: {vals1 - vals2:.4f}")
        # Cz channel
        vals1 = cz_scores[i]
        vals2 = cz_scores[j]
        print(f"Cz ({d1} vs {d2}):")
        print(f"  {d1}: {vals1:.4f}, {d2}: {vals2:.4f}")
        print(f"  Difference: {vals1 - vals2:.4f}")

print(f"\nAll plots saved to: {output_dir}")

# %%

# %%
