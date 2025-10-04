#!/usr/bin/env python3
"""
GUI-based EEG Preprocessing Parameter Adjuster
Provides a simple interface for modifying preprocessing parameters.
"""

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict

import yaml


class PreprocessingGUI:
    """GUI for adjusting EEG preprocessing parameters."""

    def __init__(self, config_path: str = None):
        self.root = tk.Tk()
        self.root.title("EEG Preprocessing Parameter Adjuster")
        self.root.geometry("800x900")

        self.config_path = config_path
        self.config = {}
        self.modified = False

        self.setup_ui()

        if config_path:
            self.load_config(config_path)

    def setup_ui(self):
        """Setup the user interface."""
        # Create main frame with scrollbar
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Menu bar
        self.create_menu()

        # Create notebook for different parameter categories
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        self.create_dataset_tab()
        self.create_filtering_tab()
        self.create_channels_tab()
        self.create_bad_spans_tab()
        self.create_ica_tab()

        # Save button
        save_frame = ttk.Frame(main_frame)
        save_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(
            save_frame, text="Save Configuration", command=self.save_config
        ).pack(side=tk.RIGHT)
        ttk.Button(
            save_frame, text="Load Configuration", command=self.load_config_dialog
        ).pack(side=tk.RIGHT, padx=(0, 10))

    def create_menu(self):
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Config", command=self.new_config)
        file_menu.add_command(label="Load Config", command=self.load_config_dialog)
        file_menu.add_command(label="Save Config", command=self.save_config)
        file_menu.add_command(label="Save Config As...", command=self.save_config_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

    def create_dataset_tab(self):
        """Create dataset configuration tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Dataset")

        # Dataset name
        ttk.Label(frame, text="Dataset Name:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )
        self.dataset_name = tk.StringVar()
        ttk.Entry(frame, textvariable=self.dataset_name, width=50).grid(
            row=0, column=1, padx=5, pady=5
        )

        # Raw file path
        ttk.Label(frame, text="Raw File Path:").grid(
            row=1, column=0, sticky="w", padx=5, pady=5
        )
        file_frame = ttk.Frame(frame)
        file_frame.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        self.raw_file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.raw_file_path, width=40).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(file_frame, text="Browse", command=self.browse_raw_file).pack(
            side=tk.RIGHT, padx=(5, 0)
        )

        # Cropping
        crop_frame = ttk.LabelFrame(frame, text="Cropping")
        crop_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=10)

        ttk.Label(crop_frame, text="Start Time (s):").grid(
            row=0, column=0, sticky="w", padx=5, pady=2
        )
        self.tmin = tk.DoubleVar()
        ttk.Entry(crop_frame, textvariable=self.tmin, width=10).grid(
            row=0, column=1, padx=5, pady=2
        )

        ttk.Label(crop_frame, text="End Time (s):").grid(
            row=0, column=2, sticky="w", padx=5, pady=2
        )
        self.tmax = tk.DoubleVar()
        ttk.Entry(crop_frame, textvariable=self.tmax, width=10).grid(
            row=0, column=3, padx=5, pady=2
        )

    def create_filtering_tab(self):
        """Create filtering configuration tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Filtering")

        # Bandpass filter
        bandpass_frame = ttk.LabelFrame(frame, text="Bandpass Filter")
        bandpass_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(bandpass_frame, text="Highpass (Hz):").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )
        self.highpass = tk.DoubleVar()
        ttk.Entry(bandpass_frame, textvariable=self.highpass, width=10).grid(
            row=0, column=1, padx=5, pady=5
        )

        ttk.Label(bandpass_frame, text="Lowpass (Hz):").grid(
            row=0, column=2, sticky="w", padx=5, pady=5
        )
        self.lowpass = tk.DoubleVar()
        ttk.Entry(bandpass_frame, textvariable=self.lowpass, width=10).grid(
            row=0, column=3, padx=5, pady=5
        )

        # Notch filter
        notch_frame = ttk.LabelFrame(frame, text="Notch Filter")
        notch_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(notch_frame, text="Frequencies (Hz, comma-separated):").pack(
            anchor="w", padx=5, pady=2
        )
        self.notch_freqs = tk.StringVar()
        ttk.Entry(notch_frame, textvariable=self.notch_freqs, width=50).pack(
            fill=tk.X, padx=5, pady=2
        )

    def create_channels_tab(self):
        """Create channels configuration tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Channels")

        # Channels to drop
        drop_frame = ttk.LabelFrame(frame, text="Channels to Drop")
        drop_frame.pack(fill=tk.X, padx=5, pady=5)

        drop_inner = ttk.Frame(drop_frame)
        drop_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.drop_listbox = tk.Listbox(drop_inner, height=4)
        self.drop_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        drop_buttons = ttk.Frame(drop_inner)
        drop_buttons.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

        self.drop_entry = tk.StringVar()
        ttk.Entry(drop_buttons, textvariable=self.drop_entry, width=15).pack(pady=2)
        ttk.Button(drop_buttons, text="Add", command=self.add_drop_channel).pack(pady=2)
        ttk.Button(drop_buttons, text="Remove", command=self.remove_drop_channel).pack(
            pady=2
        )

        # Bad channels
        bad_frame = ttk.LabelFrame(frame, text="Bad Channels")
        bad_frame.pack(fill=tk.X, padx=5, pady=5)

        bad_inner = ttk.Frame(bad_frame)
        bad_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.bad_listbox = tk.Listbox(bad_inner, height=6)
        self.bad_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        bad_buttons = ttk.Frame(bad_inner)
        bad_buttons.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

        self.bad_entry = tk.StringVar()
        ttk.Entry(bad_buttons, textvariable=self.bad_entry, width=15).pack(pady=2)
        ttk.Button(bad_buttons, text="Add", command=self.add_bad_channel).pack(pady=2)
        ttk.Button(bad_buttons, text="Remove", command=self.remove_bad_channel).pack(
            pady=2
        )

    def create_bad_spans_tab(self):
        """Create bad spans configuration tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Bad Spans")

        # Bad spans list
        list_frame = ttk.LabelFrame(frame, text="Bad Time Spans")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Treeview for bad spans
        columns = ("Onset", "Duration", "Description")
        self.spans_tree = ttk.Treeview(
            list_frame, columns=columns, show="headings", height=8
        )

        for col in columns:
            self.spans_tree.heading(col, text=col)
            self.spans_tree.column(col, width=100)

        self.spans_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Scrollbar for treeview
        span_scroll = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=self.spans_tree.yview
        )
        span_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.spans_tree.configure(yscrollcommand=span_scroll.set)

        # Add/remove bad spans
        span_controls = ttk.Frame(frame)
        span_controls.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(span_controls, text="Onset (s):").grid(row=0, column=0, padx=5)
        self.span_onset = tk.DoubleVar()
        ttk.Entry(span_controls, textvariable=self.span_onset, width=10).grid(
            row=0, column=1, padx=5
        )

        ttk.Label(span_controls, text="Duration (s):").grid(row=0, column=2, padx=5)
        self.span_duration = tk.DoubleVar()
        ttk.Entry(span_controls, textvariable=self.span_duration, width=10).grid(
            row=0, column=3, padx=5
        )

        ttk.Label(span_controls, text="Description:").grid(row=0, column=4, padx=5)
        self.span_description = tk.StringVar(value="bad_span")
        ttk.Entry(span_controls, textvariable=self.span_description, width=15).grid(
            row=0, column=5, padx=5
        )

        ttk.Button(span_controls, text="Add Span", command=self.add_bad_span).grid(
            row=0, column=6, padx=5
        )
        ttk.Button(
            span_controls, text="Remove Selected", command=self.remove_bad_span
        ).grid(row=0, column=7, padx=5)

    def create_ica_tab(self):
        """Create ICA configuration tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="ICA")

        # ICA parameters
        params_frame = ttk.LabelFrame(frame, text="ICA Parameters")
        params_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(params_frame, text="Random State:").grid(
            row=0, column=0, sticky="w", padx=5, pady=2
        )
        self.ica_random_state = tk.IntVar()
        ttk.Entry(params_frame, textvariable=self.ica_random_state, width=10).grid(
            row=0, column=1, padx=5, pady=2
        )

        ttk.Label(params_frame, text="Method:").grid(
            row=0, column=2, sticky="w", padx=5, pady=2
        )
        self.ica_method = tk.StringVar()
        method_combo = ttk.Combobox(
            params_frame, textvariable=self.ica_method, width=15
        )
        method_combo["values"] = ("infomax", "fastica", "picard")
        method_combo.grid(row=0, column=3, padx=5, pady=2)

        ttk.Label(params_frame, text="Rejection Threshold:").grid(
            row=1, column=0, sticky="w", padx=5, pady=2
        )
        self.ica_rejection_thr = tk.DoubleVar()
        ttk.Entry(params_frame, textvariable=self.ica_rejection_thr, width=10).grid(
            row=1, column=1, padx=5, pady=2
        )

        # Exclude components
        exclude_frame = ttk.LabelFrame(frame, text="Excluded Components")
        exclude_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        exclude_inner = ttk.Frame(exclude_frame)
        exclude_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.exclude_listbox = tk.Listbox(exclude_inner, height=6)
        self.exclude_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        exclude_buttons = ttk.Frame(exclude_inner)
        exclude_buttons.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

        self.exclude_entry = tk.IntVar()
        ttk.Entry(exclude_buttons, textvariable=self.exclude_entry, width=10).pack(
            pady=2
        )
        ttk.Button(
            exclude_buttons, text="Add", command=self.add_exclude_component
        ).pack(pady=2)
        ttk.Button(
            exclude_buttons, text="Remove", command=self.remove_exclude_component
        ).pack(pady=2)
        ttk.Button(
            exclude_buttons, text="Clear All", command=self.clear_exclude_components
        ).pack(pady=2)

    def browse_raw_file(self):
        """Browse for raw file."""
        filename = filedialog.askopenfilename(
            title="Select Raw EEG File",
            filetypes=[
                ("BrainVision", "*.vhdr"),
                ("XDF", "*.xdf"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            self.raw_file_path.set(filename)

    def add_drop_channel(self):
        """Add channel to drop list."""
        channel = self.drop_entry.get().strip()
        if channel:
            self.drop_listbox.insert(tk.END, channel)
            self.drop_entry.set("")

    def remove_drop_channel(self):
        """Remove selected channel from drop list."""
        selection = self.drop_listbox.curselection()
        if selection:
            self.drop_listbox.delete(selection[0])

    def add_bad_channel(self):
        """Add bad channel."""
        channel = self.bad_entry.get().strip()
        if channel:
            self.bad_listbox.insert(tk.END, channel)
            self.bad_entry.set("")

    def remove_bad_channel(self):
        """Remove selected bad channel."""
        selection = self.bad_listbox.curselection()
        if selection:
            self.bad_listbox.delete(selection[0])

    def add_bad_span(self):
        """Add bad time span."""
        try:
            onset = self.span_onset.get()
            duration = self.span_duration.get()
            description = self.span_description.get() or "bad_span"

            self.spans_tree.insert("", tk.END, values=(onset, duration, description))

            # Clear entries
            self.span_onset.set(0.0)
            self.span_duration.set(0.0)
            self.span_description.set("bad_span")
        except tk.TclError:
            messagebox.showerror(
                "Error", "Please enter valid onset and duration values."
            )

    def remove_bad_span(self):
        """Remove selected bad span."""
        selection = self.spans_tree.selection()
        if selection:
            self.spans_tree.delete(selection[0])

    def add_exclude_component(self):
        """Add ICA component to exclude."""
        try:
            component = self.exclude_entry.get()
            self.exclude_listbox.insert(tk.END, str(component))
            self.exclude_entry.set(0)
        except tk.TclError:
            messagebox.showerror("Error", "Please enter a valid component number.")

    def remove_exclude_component(self):
        """Remove selected excluded component."""
        selection = self.exclude_listbox.curselection()
        if selection:
            self.exclude_listbox.delete(selection[0])

    def clear_exclude_components(self):
        """Clear all excluded components."""
        self.exclude_listbox.delete(0, tk.END)

    def new_config(self):
        """Create new configuration."""
        self.config = {}
        self.config_path = None
        self.update_ui_from_config()

    def load_config_dialog(self):
        """Load configuration from file dialog."""
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
        )
        if filename:
            self.load_config(filename)

    def load_config(self, filename: str = None):
        """Load configuration from file."""
        try:
            if filename:
                self.config_path = filename

            if self.config_path:
                with open(self.config_path, "r") as f:
                    self.config = yaml.safe_load(f)
                self.update_ui_from_config()
                messagebox.showinfo(
                    "Success", f"Configuration loaded from {self.config_path}"
                )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")

    def update_ui_from_config(self):
        """Update UI elements from configuration."""
        # Dataset
        self.dataset_name.set(self.config.get("dataset", {}).get("name", ""))
        self.raw_file_path.set(self.config.get("dataset", {}).get("raw_file", ""))

        # Cropping
        cropping = self.config.get("cropping", {})
        self.tmin.set(cropping.get("tmin", 0.0))
        self.tmax.set(cropping.get("tmax", 0.0))

        # Filtering
        filtering = self.config.get("filtering", {})
        self.highpass.set(filtering.get("highpass", 1.0))
        self.lowpass.set(filtering.get("lowpass", 40.0))

        notch = filtering.get("notch", [])
        if notch:
            self.notch_freqs.set(", ".join(map(str, notch)))
        else:
            self.notch_freqs.set("")

        # Channels
        channels = self.config.get("channels", {})

        # Drop channels
        self.drop_listbox.delete(0, tk.END)
        for ch in channels.get("drop", []):
            self.drop_listbox.insert(tk.END, ch)

        # Bad channels
        self.bad_listbox.delete(0, tk.END)
        for ch in channels.get("bads", []):
            self.bad_listbox.insert(tk.END, ch)

        # Bad spans
        for item in self.spans_tree.get_children():
            self.spans_tree.delete(item)

        for span in self.config.get("bad_spans", []):
            self.spans_tree.insert(
                "",
                tk.END,
                values=(
                    span.get("onset", 0),
                    span.get("duration", 0),
                    span.get("description", "bad_span"),
                ),
            )

        # ICA
        ica = self.config.get("ica", {})
        self.ica_random_state.set(ica.get("random_state", 1))
        self.ica_method.set(ica.get("method", "infomax"))
        self.ica_rejection_thr.set(ica.get("rejection_thr", 0.9))

        # Exclude components
        self.exclude_listbox.delete(0, tk.END)
        for comp in ica.get("exclude_components", []):
            self.exclude_listbox.insert(tk.END, str(comp))

    def save_config(self):
        """Save configuration to file."""
        if not self.config_path:
            self.save_config_as()
            return

        try:
            self.update_config_from_ui()
            with open(self.config_path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            messagebox.showinfo("Success", f"Configuration saved to {self.config_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")

    def save_config_as(self):
        """Save configuration to new file."""
        filename = filedialog.asksaveasfilename(
            title="Save Configuration As",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
        )
        if filename:
            self.config_path = filename
            self.save_config()

    def update_config_from_ui(self):
        """Update configuration from UI elements."""
        # Dataset
        if "dataset" not in self.config:
            self.config["dataset"] = {}
        self.config["dataset"]["name"] = self.dataset_name.get()
        self.config["dataset"]["raw_file"] = self.raw_file_path.get()

        # Cropping
        self.config["cropping"] = {"tmin": self.tmin.get(), "tmax": self.tmax.get()}

        # Filtering
        notch_str = self.notch_freqs.get().strip()
        notch_list = []
        if notch_str:
            try:
                notch_list = [
                    float(x.strip()) for x in notch_str.split(",") if x.strip()
                ]
            except ValueError:
                messagebox.showerror("Error", "Invalid notch frequencies format")
                return

        self.config["filtering"] = {
            "highpass": self.highpass.get(),
            "lowpass": self.lowpass.get(),
            "notch": notch_list,
        }

        # Channels
        drop_channels = [
            self.drop_listbox.get(i) for i in range(self.drop_listbox.size())
        ]
        bad_channels = [self.bad_listbox.get(i) for i in range(self.bad_listbox.size())]

        self.config["channels"] = {"drop": drop_channels, "bads": bad_channels}

        # Bad spans
        bad_spans = []
        for item in self.spans_tree.get_children():
            values = self.spans_tree.item(item)["values"]
            bad_spans.append(
                {
                    "onset": float(values[0]),
                    "duration": float(values[1]),
                    "description": str(values[2]),
                }
            )
        self.config["bad_spans"] = bad_spans

        # ICA
        exclude_components = []
        for i in range(self.exclude_listbox.size()):
            try:
                exclude_components.append(int(self.exclude_listbox.get(i)))
            except ValueError:
                pass

        self.config["ica"] = {
            "random_state": self.ica_random_state.get(),
            "method": self.ica_method.get(),
            "rejection_thr": self.ica_rejection_thr.get(),
            "exclude_components": exclude_components,
            "n_components": None,
            "max_iter": "auto",
            "reject_by_annotation": True,
            "fit_params": {"extended": True},
        }

        # Add other required fields
        if "montage" not in self.config:
            self.config["montage"] = {"montage_file": None, "head_size": None}

        if "rereferencing" not in self.config:
            self.config["rereferencing"] = {"method": "average"}

        if "output" not in self.config:
            self.config["output"] = {
                "save_dir": "results/preprocessed/",
                "save_format": "fif.gz",
            }

    def run(self):
        """Run the GUI."""
        self.root.mainloop()


if __name__ == "__main__":
    # Example usage
    gui = PreprocessingGUI()
    gui.run()
