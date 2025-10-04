#!/usr/bin/env python3
"""
Interactive EEG Preprocessing Pipeline
Allows dynamic parameter adjustment and configuration updates during preprocessing.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import yaml

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


class InteractivePreprocessor:
    """Interactive EEG preprocessing pipeline with dynamic config updates."""

    def __init__(self, run_name: str, config_path: Optional[str] = None):
        self.run_name = run_name
        self.config_path = config_path or f"configs/config_{run_name}.yaml"
        self.config = load_config(self.config_path)
        self.raw_original = None
        self.raw_current = None
        self.ica = None
        self.processing_history = []

    def load_data(self) -> None:
        """Load the original data."""
        print(f"Loading data for {self.run_name}...")
        self.raw_original = load_data_as_raw(
            Path(self.config["dataset"]["raw_file"]), config=self.config
        )
        self.raw_current = self.raw_original.copy()
        self.processing_history.append("data_loaded")

    def update_config(self, section: str, key: str, value, save: bool = True) -> None:
        """Update configuration and optionally save to file."""
        if section not in self.config:
            self.config[section] = {}

        # Handle nested keys
        if isinstance(key, str) and "." in key:
            keys = key.split(".")
            current = self.config[section]
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        else:
            self.config[section][key] = value

        if save:
            self.save_config()

    def save_config(self) -> None:
        """Save current configuration to file."""
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        print(f"Configuration saved to {self.config_path}")

    def add_bad_channels(self, channels: List[str], channel_type: str = "bads") -> None:
        """Add bad channels to configuration."""
        current_bads = self.config["channels"].get(channel_type, [])
        new_bads = list(set(current_bads + channels))
        self.update_config("channels", channel_type, new_bads)
        print(f"Added {channels} to {channel_type}. Current {channel_type}: {new_bads}")

    def remove_bad_channels(
        self, channels: List[str], channel_type: str = "bads"
    ) -> None:
        """Remove channels from bad channel list."""
        current_bads = self.config["channels"].get(channel_type, [])
        new_bads = [ch for ch in current_bads if ch not in channels]
        self.update_config("channels", channel_type, new_bads)
        print(
            f"Removed {channels} from {channel_type}. Current {channel_type}: {new_bads}"
        )

    def add_bad_span(
        self, onset: float, duration: float, description: str = "bad_span"
    ) -> None:
        """Add a bad time span to configuration."""
        bad_spans = self.config.get("bad_spans", [])
        new_span = {"onset": onset, "duration": duration, "description": description}
        bad_spans.append(new_span)
        self.update_config("bad_spans", "", bad_spans)
        print(f"Added bad span: {new_span}")

    def remove_bad_span(self, index: int) -> None:
        """Remove a bad span by index."""
        bad_spans = self.config.get("bad_spans", [])
        if 0 <= index < len(bad_spans):
            removed = bad_spans.pop(index)
            self.update_config("bad_spans", "", bad_spans)
            print(f"Removed bad span {index}: {removed}")
        else:
            print(
                f"Invalid bad span index: {index}. Available indices: 0-{len(bad_spans)-1}"
            )

    def list_bad_spans(self) -> None:
        """List all current bad spans."""
        bad_spans = self.config.get("bad_spans", [])
        if bad_spans:
            print("Current bad spans:")
            for i, span in enumerate(bad_spans):
                print(
                    f"  {i}: {span['onset']}s - {span['onset'] + span['duration']}s ({span.get('description', 'bad_span')})"
                )
        else:
            print("No bad spans defined")

    def step_1_initial_inspection(self) -> None:
        """Step 1: Load and inspect raw data."""
        if self.raw_original is None:
            self.load_data()

        # Apply initial channel drops
        self.raw_current = drop_channels(
            self.raw_current, self.config["channels"].get("drop", [])
        )

        # Add bad span annotations
        add_bad_span_annotations(self.raw_current, self.config.get("bad_spans", []))

        print("\n=== STEP 1: Initial Data Inspection ===")
        print("Inspect the data and mark any additional bad channels or time spans.")
        print("Use the following methods to update the configuration:")
        print("  - preprocessor.add_bad_channels(['CH1', 'CH2'])")
        print("  - preprocessor.add_bad_span(onset=100, duration=5)")
        print("  - preprocessor.list_bad_spans()")

        inspect_raw(self.raw_current, modify_original=True)

    def step_2_ica_preparation(self) -> None:
        """Step 2: Prepare data for ICA."""
        print("\n=== STEP 2: ICA Preparation ===")

        # Start fresh from original data
        self.raw_current = self.raw_original.copy()

        # Apply all preprocessing steps
        self.raw_current = drop_channels(
            self.raw_current, self.config["channels"].get("drop", [])
        )

        # Set bad channels
        self.raw_current.info["bads"] = self.config["channels"]["bads"]

        # Add bad spans
        add_bad_span_annotations(self.raw_current, self.config.get("bad_spans", []))

        # Create ICA copy
        raw_ica = self.raw_current.copy()

        # ICA preprocessing
        raw_ica.filter(1, 100)
        raw_ica = apply_notch_filter(raw_ica, self.config["filtering"].get("notch"))
        raw_ica.crop(self.config["cropping"]["tmin"], self.config["cropping"]["tmax"])
        raw_ica = raw_ica.set_eeg_reference(ref_channels="average")

        self.raw_ica = raw_ica
        print("Data prepared for ICA. Inspect if needed:")
        inspect_raw(self.raw_ica)

    def step_3_ica_fitting(self, refit: bool = False) -> None:
        """Step 3: Fit or load ICA."""
        print("\n=== STEP 3: ICA Fitting ===")

        ica_path = f"{self.config['output']['save_dir']}/{self.config['dataset']['name']}_ica.{self.config['output']['save_format']}"

        if os.path.exists(ica_path) and not refit:
            print("Loading existing ICA...")
            self.ica = load_ica(self.config)
        else:
            print("Fitting new ICA...")
            self.ica = fit_ica(self.raw_ica, self.config["ica"])
            save_ica(self.ica, self.config)

        # Show components
        plt.close("all")
        self.ica.plot_components(nrows=4, ncols="auto")
        self.ica.plot_sources(self.raw_ica)

    def step_4_ica_component_selection(self, use_iclabel: bool = True) -> None:
        """Step 4: Select bad ICA components."""
        print("\n=== STEP 4: ICA Component Selection ===")

        if use_iclabel:
            # Automatic selection with ICLabel
            bad_ics = get_artifact_indices(
                self.raw_ica, self.ica, self.config["ica"]["rejection_thr"], plot=True
            )
            print(f"ICLabel suggested components: {bad_ics}")
            print("Review the suggested components. Update manually if needed:")
            print("  - preprocessor.update_ica_exclusions([0, 1, 2, ...])")
        else:
            # Manual selection
            current_exclusions = self.config["ica"].get("exclude_components", [])
            print(f"Current ICA exclusions: {current_exclusions}")
            print("Update exclusions manually:")
            print("  - preprocessor.update_ica_exclusions([0, 1, 2, ...])")

    def update_ica_exclusions(self, component_indices: List[int]) -> None:
        """Update ICA component exclusions."""
        self.update_config("ica", "exclude_components", component_indices)
        self.ica.exclude = component_indices
        print(f"Updated ICA exclusions: {component_indices}")

    def step_5_final_preprocessing(self) -> None:
        """Step 5: Apply final preprocessing."""
        print("\n=== STEP 5: Final Preprocessing ===")

        # Start fresh
        raw_final = self.raw_original.copy()

        # Apply all steps
        raw_final = drop_channels(raw_final, self.config["channels"].get("drop", []))
        add_bad_span_annotations(raw_final, self.config.get("bad_spans", []))

        # Custom processing for specific runs
        if self.run_name == "run3":
            raw_final.annotations.duration[1] = 0.0
            raw_final.annotations.duration[2] = 4.0
            print("Applied custom processing for run3")

        # Final filtering and processing
        raw_final.filter(
            self.config["filtering"]["highpass"], self.config["filtering"]["lowpass"]
        )
        raw_final = apply_notch_filter(raw_final, self.config["filtering"].get("notch"))
        raw_final.crop(self.config["cropping"]["tmin"], self.config["cropping"]["tmax"])

        # Apply ICA
        self.ica.exclude = self.config["ica"]["exclude_components"]
        raw_final = self.ica.apply(raw_final)
        raw_final = raw_final.set_eeg_reference(ref_channels="average")

        # Custom processing for run1
        if self.run_name == "run1":
            if len(raw_final.annotations) > 0:
                raw_final.annotations.delete(0)
                print("Applied custom processing for run1")

        # Store bad channels before interpolation
        original_bads = raw_final.info["bads"].copy()
        print(f"Bad channels before interpolation: {original_bads}")

        # Interpolate bad channels
        raw_final.interpolate_bads(reset_bads=True)
        print(
            f"Bad channels after interpolation: {raw_final.info['bads']} (reset by MNE)"
        )
        print(f"Originally bad channels were: {original_bads}")

        self.raw_final = raw_final
        inspect_raw(raw_final)

    def save_final(self) -> None:
        """Save the final preprocessed data."""
        if hasattr(self, "raw_final"):
            save_preprocessed(self.raw_final, self.config)
            print("Final preprocessed data saved!")
        else:
            print("No final data to save. Run step_5_final_preprocessing() first.")

    def run_all_steps(self, inspect_each_step: bool = True) -> None:
        """Run all preprocessing steps in sequence."""
        print("Running complete preprocessing pipeline...")

        self.step_1_initial_inspection()
        if inspect_each_step:
            input("Press Enter to continue to ICA preparation...")

        self.step_2_ica_preparation()
        if inspect_each_step:
            input("Press Enter to continue to ICA fitting...")

        self.step_3_ica_fitting()
        if inspect_each_step:
            input("Press Enter to continue to component selection...")

        self.step_4_ica_component_selection()
        if inspect_each_step:
            input("Press Enter to continue to final preprocessing...")

        self.step_5_final_preprocessing()
        if inspect_each_step:
            input("Press Enter to save final data...")

        self.save_final()
        print("Preprocessing pipeline completed!")


# Convenience function for quick usage
def preprocess_run(
    run_name: str, config_path: Optional[str] = None
) -> InteractivePreprocessor:
    """Create and return an interactive preprocessor for a run."""
    return InteractivePreprocessor(run_name, config_path)


if __name__ == "__main__":
    # Example usage
    run_name = "run0"  # Change this as needed

    # Create preprocessor
    preprocessor = preprocess_run(run_name)

    # Run step by step (recommended for first time)
    preprocessor.step_1_initial_inspection()

    # Or run all steps at once
    # preprocessor.run_all_steps()
