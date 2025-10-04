#!/usr/bin/env python3
"""
Enhanced utilities for interactive EEG preprocessing.
Includes callbacks, inspection helpers, and workflow improvements.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import mne

from utils import *  # Import all existing functions


def inspect_raw_interactive(raw, callback: Optional[Callable] = None, **kwargs):
    """
    Enhanced raw data inspection with optional callback.

    Parameters:
    -----------
    raw : mne.io.Raw
        Raw object to inspect
    callback : callable, optional
        Function to call after inspection. Should accept (raw, user_input) parameters.
    **kwargs : dict
        Additional arguments passed to callback
    """
    print("\n=== Raw Data Inspection ===")
    print(f"Channels: {len(raw.ch_names)}")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1]:.1f} seconds")
    print(f"Bad channels: {raw.info['bads']}")
    print(f"Annotations: {len(raw.annotations)} spans")

    # Plot data
    inspect_raw(raw, modify_original=True)

    if callback:
        user_input = input(
            "\nEnter any notes or commands (or press Enter to continue): "
        )
        callback(raw, user_input, **kwargs)


def inspect_ica_interactive(raw, ica, callback: Optional[Callable] = None, **kwargs):
    """
    Enhanced ICA inspection with optional callback.

    Parameters:
    -----------
    raw : mne.io.Raw
        Raw object used for ICA
    ica : mne.preprocessing.ICA
        ICA object to inspect
    callback : callable, optional
        Function to call after inspection
    **kwargs : dict
        Additional arguments passed to callback
    """
    print("\n=== ICA Inspection ===")
    print(f"Number of components: {ica.n_components_}")
    print(f"Current exclusions: {ica.exclude}")

    # Plot components and sources
    plt.close("all")
    ica.plot_components(nrows=4, ncols="auto")
    ica.plot_sources(raw)

    if callback:
        user_input = input(
            "\nEnter component numbers to exclude (comma-separated) or press Enter: "
        )
        if user_input.strip():
            try:
                new_exclusions = [int(x.strip()) for x in user_input.split(",")]
                ica.exclude = new_exclusions
                print(f"Updated exclusions: {ica.exclude}")
            except ValueError:
                print("Invalid input format. Please enter comma-separated numbers.")

        if callback:
            callback(ica, user_input, **kwargs)


def create_preprocessing_report(
    raw_original, raw_final, ica, config, output_path: str = None
):
    """
    Create a comprehensive preprocessing report.

    Parameters:
    -----------
    raw_original : mne.io.Raw
        Original raw data
    raw_final : mne.io.Raw
        Final preprocessed data
    ica : mne.preprocessing.ICA
        ICA object used
    config : dict
        Configuration dictionary
    output_path : str, optional
        Path to save the report
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if output_path is None:
        output_path = f"preprocessing_report_{config['dataset']['name']}.pdf"

    with PdfPages(output_path) as pdf:
        # Page 1: Summary
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.axis("off")

        summary_text = f"""
Preprocessing Report: {config['dataset']['name']}

ORIGINAL DATA:
• File: {Path(config['dataset']['raw_file']).name}
• Channels: {len(raw_original.ch_names)}
• Duration: {raw_original.times[-1]:.1f} seconds
• Sampling rate: {raw_original.info['sfreq']} Hz

PREPROCESSING PARAMETERS:
• Cropping: {config['cropping']['tmin']} - {config['cropping']['tmax']} s
• Filtering: {config['filtering']['highpass']} - {config['filtering']['lowpass']} Hz
• Notch: {config['filtering'].get('notch', 'None')} Hz
• Dropped channels: {config['channels']['drop']}
• Bad channels: {config['channels']['bads']}
• Bad spans: {len(config.get('bad_spans', []))} segments

ICA:
• Method: {config['ica']['method']}
• Components: {ica.n_components_}
• Excluded: {config['ica']['exclude_components']}

FINAL DATA:
• Channels: {len(raw_final.ch_names)}
• Duration: {raw_final.times[-1]:.1f} seconds
• Bad channels interpolated: {len(config['channels']['bads'])}
        """

        ax.text(
            0.1,
            0.9,
            summary_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )
        plt.title("Preprocessing Summary", fontsize=16, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: Power spectral density comparison
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        raw_original.compute_psd().plot(axes=ax1, show=False)
        ax1.set_title("Original Data - Power Spectral Density")

        raw_final.compute_psd().plot(axes=ax2, show=False)
        ax2.set_title("Final Data - Power Spectral Density")

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3: ICA components
        if ica.n_components_ > 0:
            fig = ica.plot_components(show=False, title="ICA Components")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Page 4: Channel locations (if montage available)
        if raw_final.info["dig"] is not None:
            try:
                fig = raw_final.plot_sensors(show_names=True, show=False)
                plt.title("Channel Locations")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
            except Exception:
                pass  # Skip if plotting fails

    print(f"Preprocessing report saved to: {output_path}")


def batch_preprocess_runs(
    config_files: List[str], inspect_each: bool = False, save_reports: bool = True
):
    """
    Batch process multiple runs with optional inspection.

    Parameters:
    -----------
    config_files : list of str
        List of configuration file paths
    inspect_each : bool
        Whether to inspect each run individually
    save_reports : bool
        Whether to save preprocessing reports
    """
    from interactive_preprocessing import InteractivePreprocessor

    results = {}

    for config_file in config_files:
        print(f"\n{'='*60}")
        print(f"Processing: {config_file}")
        print(f"{'='*60}")

        try:
            # Extract run name from config file
            run_name = Path(config_file).stem.replace("config_", "")

            # Create preprocessor
            preprocessor = InteractivePreprocessor(run_name, config_file)

            if inspect_each:
                # Interactive mode
                preprocessor.run_all_steps(inspect_each_step=True)
            else:
                # Automatic mode
                preprocessor.load_data()
                preprocessor.step_2_ica_preparation()
                preprocessor.step_3_ica_fitting()
                preprocessor.step_4_ica_component_selection(use_iclabel=True)
                preprocessor.step_5_final_preprocessing()
                preprocessor.save_final()

            # Generate report
            if save_reports:
                report_path = f"results/reports/preprocessing_report_{run_name}.pdf"
                Path("results/reports").mkdir(parents=True, exist_ok=True)

                create_preprocessing_report(
                    preprocessor.raw_original,
                    preprocessor.raw_final,
                    preprocessor.ica,
                    preprocessor.config,
                    report_path,
                )

            results[run_name] = {
                "status": "success",
                "config_file": config_file,
                "preprocessor": preprocessor,
            }

        except Exception as e:
            print(f"Error processing {config_file}: {str(e)}")
            results[run_name] = {
                "status": "error",
                "config_file": config_file,
                "error": str(e),
            }

    # Summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")

    successful = sum(1 for r in results.values() if r["status"] == "success")
    failed = len(results) - successful

    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed runs:")
        for run_name, result in results.items():
            if result["status"] == "error":
                print(f"  - {run_name}: {result['error']}")

    return results


def compare_preprocessing_results(
    run_names: List[str], metric: str = "alpha_power", channels: List[str] = None
):
    """
    Compare preprocessing results across runs.

    Parameters:
    -----------
    run_names : list of str
        List of run names to compare
    metric : str
        Metric to compare ('alpha_power', 'psd', 'channel_variance')
    channels : list of str, optional
        Specific channels to compare (default: all EEG channels)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    results = {}

    for run_name in run_names:
        # Load preprocessed data
        data_path = f"results/preprocessed/{run_name}_raw.fif.gz"
        if Path(data_path).exists():
            raw = mne.io.read_raw_fif(data_path, preload=True)

            if channels is None:
                eeg_channels = mne.pick_types(raw.info, eeg=True)
                channels = [raw.ch_names[i] for i in eeg_channels]

            if metric == "alpha_power":
                # Calculate alpha power (8-12 Hz)
                psd = raw.compute_psd(fmin=8, fmax=12, picks=channels)
                alpha_power = np.mean(psd.get_data(), axis=-1)
                results[run_name] = alpha_power

            elif metric == "psd":
                # Full PSD
                psd = raw.compute_psd(picks=channels)
                results[run_name] = {
                    "freqs": psd.freqs,
                    "data": np.mean(psd.get_data(), axis=0),
                }

            elif metric == "channel_variance":
                # Channel variance
                data = raw.get_data(picks=channels)
                variance = np.var(data, axis=-1)
                results[run_name] = variance

    # Plot comparison
    if metric in ["alpha_power", "channel_variance"]:
        fig, ax = plt.subplots(figsize=(12, 6))

        x_pos = np.arange(len(channels))
        width = 0.8 / len(run_names)

        for i, (run_name, values) in enumerate(results.items()):
            ax.bar(x_pos + i * width, values, width, label=run_name, alpha=0.8)

        ax.set_xlabel("Channels")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison Across Runs')
        ax.set_xticks(x_pos + width * (len(run_names) - 1) / 2)
        ax.set_xticklabels(channels, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

    elif metric == "psd":
        fig, ax = plt.subplots(figsize=(12, 6))

        for run_name, data in results.items():
            ax.plot(
                data["freqs"], 10 * np.log10(data["data"]), label=run_name, linewidth=2
            )

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power Spectral Density (dB)")
        ax.set_title("Power Spectral Density Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 50)

    plt.tight_layout()
    plt.show()

    return results


# Quick access functions
def quick_inspect(run_name: str):
    """Quickly inspect a preprocessed run."""
    data_path = f"results/preprocessed/{run_name}_raw.fif.gz"
    if Path(data_path).exists():
        raw = mne.io.read_raw_fif(data_path, preload=True)
        inspect_raw(raw)
        return raw
    else:
        print(f"Preprocessed data not found: {data_path}")
        return None


def quick_compare(run_names: List[str]):
    """Quick comparison of multiple runs."""
    return compare_preprocessing_results(run_names, metric="alpha_power")


if __name__ == "__main__":
    # Example usage
    print("Enhanced preprocessing utilities loaded!")
    print("\nAvailable functions:")
    print("- inspect_raw_interactive(raw, callback)")
    print("- inspect_ica_interactive(raw, ica, callback)")
    print("- create_preprocessing_report(raw_orig, raw_final, ica, config)")
    print("- batch_preprocess_runs(config_files)")
    print("- compare_preprocessing_results(run_names)")
    print("- quick_inspect(run_name)")
    print("- quick_compare(run_names)")
