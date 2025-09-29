# %%
# !%matplotlib qt
import os
from pathlib import Path

import matplotlib.pyplot as plt

from utils import (
    add_bad_span_annotations,
    apply_notch_filter,
    drop_channels,
    fit_ica,
    get_artifact_indices,
    inspect_raw,
    load_config,
    load_data_as_raw,
    load_ica,
    save_ica,
    save_preprocessed,
)

RUN = "run0"
INSPECT = True

# %%
# Load parameters and data
config = load_config(f"configs/config_{RUN}.yaml")
raw = load_data_as_raw(Path(config["dataset"]["raw_file"]), config=config)

# Drop unwanted channels
raw = drop_channels(raw, config["channels"].get("drop", []))

# Add bad span annotations
add_bad_span_annotations(raw, config.get("bad_spans", []))

inspect_raw(raw) if INSPECT else None

# %%
# ICA PREPROCESSING
config = load_config(f"configs/config_{RUN}.yaml")

# Define bad channels
raw.info["bads"] = config["channels"]["bads"]
raw_ica = raw.copy()

# Initial filtering and rereferencing for Extended Infomax
raw_ica.filter(1, 100)
# raw_ica = apply_notch_filter(raw_ica, config["filtering"].get("notch"))

# Crop the beginning and the end of the recording
raw_ica.crop(config["cropping"]["tmin"], config["cropping"]["tmax"])

# Set CAR
raw_ica = raw_ica.set_eeg_reference(ref_channels="average")

inspect_raw(raw_ica) if INSPECT else None

# %%
# ICA PROCESSING
config = load_config(f"configs/config_{RUN}.yaml")

if os.path.exists(
    f"{config['output']['save_dir']}/{config['dataset']['name']}_ica.{config['output']['save_format']}"
):
    ica = load_ica(config)
else:
    ica = fit_ica(raw_ica, config["ica"])
    save_ica(ica, config)

if config["dataset"]["name"] == "run1":
    ica.exclude.append(1)

if INSPECT:
    plt.close("all")
    ica.plot_components(nrows=4, ncols="auto")
    ica.plot_sources(raw_ica)

bad_ics = get_artifact_indices(
    raw_ica, ica, config["ica"]["rejection_thr"], plot=INSPECT
)
print(f"Bad ICs: {bad_ics}")

# %%
config = load_config(f"configs/config_{RUN}.yaml")

# Read the Raw again if you have overwritten it
raw_final = raw.copy()

# Custom processing
if RUN == "run3":
    raw_final.annotations.duration[1] = 0.0
    raw_final.annotations.duration[2] = 4.0
    print(
        f"Modified first annotation duration to {raw_final.annotations.duration[0]} seconds"
    )

# Final preprocessing
raw_final.filter(config["filtering"]["highpass"], config["filtering"]["lowpass"])
# raw_final = apply_notch_filter(raw_final, config["filtering"].get("notch"))
raw_final.crop(config["cropping"]["tmin"], config["cropping"]["tmax"])
ica.exclude = config["ica"]["exclude_components"]
raw_final = ica.apply(raw_final)
raw_final = raw_final.set_eeg_reference(ref_channels="average")

# Custom processing
if RUN == "run1":
    if len(raw_final.annotations) > 0:
        raw_final.annotations.delete(0)

inspect_raw(raw_final) if INSPECT else None

# %%
# Store bad channels before interpolation (for reference)
original_bads = raw_final.info["bads"].copy()
print(f"Bad channels before interpolation: {original_bads}")

# Interpolate bad channels
raw_final.interpolate_bads(reset_bads=True)
print(f"Bad channels after interpolation: {raw_final.info['bads']} (reset by MNE)")
print(f"Originally bad channels were: {original_bads}")
inspect_raw(raw_final) if INSPECT else None

# %%
# Final inspection and save
save_preprocessed(raw_final, config)

# %%
