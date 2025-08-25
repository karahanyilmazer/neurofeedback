# %%
# !%matplotlib qt
import os
from pathlib import Path

import matplotlib.pyplot as plt

from utils import (
    add_bad_span_annotations,
    fit_ica,
    get_artifact_indices,
    inspect_raw,
    load_ica,
    load_params,
    load_xdf_as_raw,
    save_ica,
    save_preprocessed,
)

RUN = "run2"

# %%
# Load parameters and data
params = load_params(f"configs/params_{RUN}.yaml")
raw = load_xdf_as_raw(Path(params["dataset"]["raw_file"]))

# Add bad span annotations
add_bad_span_annotations(raw, params.get("bad_spans", []))

inspect_raw(raw)

# %%
# ICA PREPROCESSING
params = load_params(f"configs/params_{RUN}.yaml")

# Define bad channels
raw.info["bads"] = params["channels"]["bads"]
raw_ica = raw.copy()

# Initial filtering and rereferencing for Extended Infomax
raw_ica.filter(1, 100)

# Crop the beginning and the end of the recording
raw_ica.crop(params["cropping"]["tmin"], params["cropping"]["tmax"])

# Set CAR
raw_ica = raw_ica.set_eeg_reference(ref_channels="average")

inspect_raw(raw_ica)

# %%
# ICA PROCESSING
params = load_params(f"configs/params_{RUN}.yaml")

if os.path.exists(
    f"{params['output']['save_dir']}/{params['dataset']['name']}_ica.{params['output']['save_format']}"
):
    ica = load_ica(params)
else:
    ica = fit_ica(raw_ica, params["ica"])
    save_ica(ica, params)

if params["dataset"]["name"] == "run1":
    ica.exclude.append(1)

plt.close("all")
ica.plot_components(nrows=4, ncols="auto")
ica.plot_sources(raw_ica)
bad_ics = get_artifact_indices(raw_ica, ica, params["ica"]["rejection_thr"])
print(f"Bad ICs: {bad_ics}")

# %%
params = load_params(f"configs/params_{RUN}.yaml")

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
raw_final.filter(params["filtering"]["highpass"], params["filtering"]["lowpass"])
if params["filtering"].get("notch"):
    raw_final.notch_filter(params["filtering"].get("notch"))
raw_final.crop(params["cropping"]["tmin"], params["cropping"]["tmax"])
ica.exclude = params["ica"]["exclude_components"]
raw_final = ica.apply(raw_final)
raw_final = raw_final.set_eeg_reference(ref_channels="average")

# Custom processing
if RUN == "run1":
    if len(raw_final.annotations) > 0:
        raw_final.annotations.delete(0)

inspect_raw(raw_final)

# %%
# Store bad channels before interpolation (for reference)
original_bads = raw_final.info["bads"].copy()
print(f"Bad channels before interpolation: {original_bads}")

# Interpolate bad channels
raw_final.interpolate_bads(reset_bads=True)
print(f"Bad channels after interpolation: {raw_final.info['bads']} (reset by MNE)")
print(f"Originally bad channels were: {original_bads}")
inspect_raw(raw_final)

# %%
# Final inspection and save
save_preprocessed(raw_final, params)

# %%
