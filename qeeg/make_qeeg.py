# %%
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# from mne_connectivity.viz import plot_connectivity_circle
import mne
import numpy as np
import pandas as pd
import pyxdf
import seaborn as sns
from mne.time_frequency import psd_array_welch
from mne.viz import plot_sensors, plot_topomap
from tqdm import tqdm

# Add the parent directory to path to import your utils
sys.path.append(str(Path(__file__).parent.parent))
from utils import apply_notch_filter, drop_channels, load_config, load_data_as_raw


# %%
def get_raw_with_your_method(run_name, fmin, fmax, common_channels):
    """
    Load raw data using your preprocessing pipeline approach.

    Parameters:
    -----------
    run_name : str
        Name of the run (e.g., 'run0')
    fmin, fmax : float
        Frequency band limits for filtering
    common_channels : list
        List of common channel names to keep

    Returns:
    --------
    mne.io.Raw : Preprocessed raw object
    """
    # Load config for the specified run
    config_path = Path(__file__).parent.parent / f"configs/config_{run_name}.yaml"
    config = load_config(config_path)

    # Load raw data using your method
    raw = load_data_as_raw(Path(config["dataset"]["raw_file"]), config=config)

    # Drop unwanted channels
    raw = drop_channels(raw, config["channels"].get("drop", []))

    # Set bad channels from config
    raw.info["bads"] = config["channels"]["bads"]

    # Apply filtering
    raw.filter(fmin, fmax)
    raw = apply_notch_filter(raw, config["filtering"].get("notch"))

    # Keep only common channels that exist in the data
    existing_common_channels = [ch for ch in common_channels if ch in raw.ch_names]
    if existing_common_channels:
        raw.pick_channels(existing_common_channels)
        print(
            f"Kept {len(existing_common_channels)} common channels: {existing_common_channels}"
        )
    else:
        print(f"Warning: No common channels found. Available channels: {raw.ch_names}")

    # Interpolate bad channels
    raw.interpolate_bads()

    # Set average reference
    raw.set_eeg_reference(ref_channels="average")

    return raw


def remove_ref_name(ch_name):
    if "-" in ch_name:
        return ch_name[: ch_name.index("-")]
    else:
        return ch_name


def read_xdf(path):
    streams, header = pyxdf.load_xdf(path)
    data = streams[0]["time_series"].T
    n_chs = data.shape[0]
    ch_names = [
        list(streams[0]["info"]["desc"][0]["channels"][0].items())[0][1][ix]["label"][0]
        for ix in range(n_chs)
    ]
    sfreq = float(streams[0]["info"]["nominal_srate"][0])
    info = mne.create_info(ch_names, sfreq, "eeg")
    raw = mne.io.RawArray(data, info)
    return raw


def get_raw(run_name, fmin, fmax):
    # Load config for the specified run
    config_path = Path(__file__).parent.parent / f"configs/config_{run_name}.yaml"
    config = load_config(config_path)

    # Load raw data using your method
    raw = load_data_as_raw(Path(config["dataset"]["raw_file"]), config=config)

    mne.rename_channels(raw.info, lambda x: x.removeprefix("EEG "))
    mne.rename_channels(raw.info, remove_ref_name)
    raw.pick_channels(common_channels)
    raw.set_montage("easycap-M1")
    raw.filter(fmin, fmax)

    raw.info["bads"] = ["Fp1", "Fp2", "F7"]
    raw.interpolate_bads()

    raw.set_eeg_reference(ref_channels="average")
    return raw


# %%
print("Current working dir:", os.getcwd())

# Configuration
subject = "run3"  # Using your run0 configuration
base_path = "."  # Adjust this to your data path if needed

df_healthy = pd.read_pickle("df_nice_healthy")
info = np.load("info.npy", allow_pickle=True).item()
ch_names = info.ch_names
freqs = np.arange(1, 46)
common_channels = [
    "Fp2",
    "Fp1",
    "F4",
    "F3",
    "C4",
    "C3",
    "P4",
    "P3",
    "O2",
    "O1",
    "F8",
    "F7",
    "Fz",
    "Cz",
    "Pz",
]

psds_healthy = np.array([x for x in df_healthy["psd"].to_numpy()])
psds_healthy /= psds_healthy.sum(-1, keepdims=True)

u_psd_healthy = psds_healthy.mean(0)
std_psd_healthy = psds_healthy.std(0)

out_path = "../results/patient_qeeg_results/{}".format(subject)
if not os.path.isdir(out_path):
    os.makedirs(out_path)

# Use your preprocessing method for run0
print(f"Loading data using your preprocessing method for {subject}...")


out_path = "../results/patient_qeeg_results/{}".format(subject)
if not os.path.isdir(out_path):
    os.makedirs(out_path)

raw = get_raw(subject, 1, 45)
sfreq = raw.info["sfreq"]
psds, freqs = psd_array_welch(
    raw.get_data(), sfreq, 1, 45, n_fft=int(sfreq), n_per_seg=int(sfreq)
)

mean_psd = psds.mean(axis=0)

# %%
# Alpha band (8–14 Hz)
alpha_mask = np.logical_and(freqs >= 8, freqs <= 14)
peak_alpha_freq = freqs[alpha_mask][np.argmax(mean_psd[alpha_mask])]

# Beta band (15–25 Hz)
beta_mask = np.logical_and(freqs >= 15, freqs <= 25)
peak_beta_freq = freqs[beta_mask][np.argmax(mean_psd[beta_mask])]

print(
    f"{subject}: Peak Alpha = {peak_alpha_freq:.2f} Hz,"
    f" Peak Beta = {peak_beta_freq:.2f} Hz"
)


# Divide by total power of each channel
psds /= psds.sum(-1, keepdims=True)

# %%
# global PSD
plt.semilogy(freqs, psds.mean(0))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (Raw)")
plt.title("Power Spectral Density")
sns.despine()
# plt.savefig('{}\\global.jpg'.format(out_path))
plt.savefig(os.path.join(out_path, "global.jpg"))
plt.show()
# plt.close()

# %%
# # for each channel
for ch_ix, ch_name in enumerate(ch_names):
    plt.semilogy(freqs, psds[ch_ix])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (Raw)")
    plt.title(ch_name)
    sns.despine()
    plt.hlines(y=[-1, 1], xmin=1, xmax=45, colors="grey", linestyles="--")
    plt.show()

# z-scored psds

# global

psds -= u_psd_healthy
psds /= std_psd_healthy

plt.plot(freqs, psds.mean(0))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (Z-Score)")
plt.title("Power Spectral Density")
sns.despine()
plt.hlines(y=[-1, 1], xmin=1, xmax=45, colors="grey", linestyles="--")
# plt.show()
# plt.savefig('{}\\global_z_score.jpg'.format(out_path))
plt.savefig(os.path.join(out_path, "global_z_score.jpg"))
plt.close()

### topoplot of z-scored values for each frequency band

for band_name, band_lims in zip(
    ["Delta", "Theta", "Alpha", "Beta", "Gamma"],
    [(0.5, 4), (4, 8), (8, 14), (15, 25), (35, 40)],
):
    band_mask = np.logical_and(freqs > band_lims[0], freqs < band_lims[1])
    band_psd = psds[:, band_mask].mean(-1)
    # band_psd[np.logical_or(band_psd>-1,band_psd<1)] = 0
    abs_lim = np.max([np.max(band_psd), np.abs(np.min(band_psd))])
    vmax = abs_lim
    vmin = -abs_lim
    im = plot_topomap(
        band_psd,
        cmap="RdBu_r",
        vlim=(vmin, vmax),
        sensors=True,
        names=ch_names,
        pos=info,
        show=False,
        size=5,
        contours=0,
    )[0]
    cbar = plt.colorbar(im, fraction=0.02, pad=0.04)
    cbar.set_label("Z-Score", rotation=270, labelpad=8)
    plt.title("{} Power".format(band_name))
    # plt.show()
    # plt.savefig('{}\\{}.jpg'.format(out_path, band_name))
    plt.savefig(os.path.join(out_path, f"{band_name}.jpg"))
    plt.close()

# %%
