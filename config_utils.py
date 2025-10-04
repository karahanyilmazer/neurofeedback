#!/usr/bin/env python3
"""
Configuration utilities for EEG preprocessing pipeline.
Includes validation, template generation, and batch processing helpers.
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


class ConfigValidator:
    """Validate EEG preprocessing configuration files."""

    REQUIRED_SECTIONS = {
        "dataset": ["name", "raw_file"],
        "cropping": ["tmin", "tmax"],
        "filtering": ["highpass", "lowpass"],
        "channels": ["drop", "bads"],
        "ica": ["random_state", "method", "rejection_thr", "exclude_components"],
        "output": ["save_dir", "save_format"],
    }

    OPTIONAL_SECTIONS = {
        "montage": ["montage_file", "head_size"],
        "rereferencing": ["method"],
        "bad_spans": [],
        "filtering": ["notch"],
        "channels": ["sketchy"],
    }

    VALID_VALUES = {
        "ica.method": ["infomax", "fastica", "picard"],
        "output.save_format": ["fif", "fif.gz"],
        "rereferencing.method": ["average", "cz", "linked_mastoids"],
    }

    def __init__(self):
        self.errors = []
        self.warnings = []

    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Validate configuration dictionary.

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        # Check required sections and fields
        for section, fields in self.REQUIRED_SECTIONS.items():
            if section not in config:
                self.errors.append(f"Missing required section: {section}")
                continue

            for field in fields:
                if field not in config[section]:
                    self.errors.append(f"Missing required field: {section}.{field}")

        # Check data types and values
        self._validate_types(config)
        self._validate_values(config)
        self._validate_file_paths(config)

        return len(self.errors) == 0, self.errors, self.warnings

    def _validate_types(self, config: Dict[str, Any]) -> None:
        """Validate data types."""
        type_checks = [
            ("dataset.name", str),
            ("dataset.raw_file", str),
            ("cropping.tmin", (int, float)),
            ("cropping.tmax", (int, float)),
            ("filtering.highpass", (int, float)),
            ("filtering.lowpass", (int, float)),
            ("channels.drop", list),
            ("channels.bads", list),
            ("ica.random_state", int),
            ("ica.method", str),
            ("ica.rejection_thr", (int, float)),
            ("ica.exclude_components", list),
        ]

        for path, expected_type in type_checks:
            value = self._get_nested_value(config, path)
            if value is not None and not isinstance(value, expected_type):
                self.errors.append(
                    f"{path} should be {expected_type.__name__}, got {type(value).__name__}"
                )

    def _validate_values(self, config: Dict[str, Any]) -> None:
        """Validate specific values."""
        # Check valid choices
        for path, valid_choices in self.VALID_VALUES.items():
            value = self._get_nested_value(config, path)
            if value is not None and value not in valid_choices:
                self.errors.append(
                    f"{path} must be one of {valid_choices}, got '{value}'"
                )

        # Check logical constraints
        cropping = config.get("cropping", {})
        if "tmin" in cropping and "tmax" in cropping:
            if cropping["tmax"] <= cropping["tmin"]:
                self.errors.append("cropping.tmax must be greater than cropping.tmin")

        filtering = config.get("filtering", {})
        if "highpass" in filtering and "lowpass" in filtering:
            if filtering["lowpass"] <= filtering["highpass"]:
                self.errors.append(
                    "filtering.lowpass must be greater than filtering.highpass"
                )

        # Check ICA rejection threshold
        ica = config.get("ica", {})
        if "rejection_thr" in ica:
            if not 0 <= ica["rejection_thr"] <= 1:
                self.errors.append("ica.rejection_thr must be between 0 and 1")

    def _validate_file_paths(self, config: Dict[str, Any]) -> None:
        """Validate file paths exist."""
        raw_file = self._get_nested_value(config, "dataset.raw_file")
        if raw_file and not Path(raw_file).exists():
            self.warnings.append(f"Raw file does not exist: {raw_file}")

        montage_file = self._get_nested_value(config, "montage.montage_file")
        if montage_file and not Path(montage_file).exists():
            self.warnings.append(f"Montage file does not exist: {montage_file}")

    def _get_nested_value(self, d: Dict[str, Any], path: str) -> Any:
        """Get nested dictionary value using dot notation."""
        keys = path.split(".")
        value = d
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value


class ConfigTemplate:
    """Generate configuration templates for different scenarios."""

    @staticmethod
    def create_base_template() -> Dict[str, Any]:
        """Create base configuration template."""
        return {
            "dataset": {"name": "run0", "raw_file": "/path/to/your/data.vhdr"},
            "montage": {"montage_file": None, "head_size": None},
            "cropping": {"tmin": 0.0, "tmax": 600.0},
            "filtering": {"highpass": 1.0, "lowpass": 40.0, "notch": [50.0, 100.0]},
            "channels": {
                "drop": ["ecg", "eda", "envelope", "envelope_am"],
                "bads": [],
                "sketchy": [],
            },
            "rereferencing": {"method": "average"},
            "bad_spans": [],
            "ica": {
                "n_components": None,
                "random_state": 1,
                "method": "infomax",
                "max_iter": "auto",
                "reject_by_annotation": True,
                "fit_params": {"extended": True},
                "rejection_thr": 0.9,
                "exclude_components": [],
            },
            "output": {"save_dir": "results/preprocessed/", "save_format": "fif.gz"},
        }

    @staticmethod
    def create_template_for_file(
        raw_file_path: str, run_name: str = None
    ) -> Dict[str, Any]:
        """Create template for specific raw file."""
        template = ConfigTemplate.create_base_template()

        # Set file path
        template["dataset"]["raw_file"] = str(Path(raw_file_path).resolve())

        # Set run name based on file name if not provided
        if run_name is None:
            run_name = Path(raw_file_path).stem
        template["dataset"]["name"] = run_name

        return template

    @staticmethod
    def create_minimal_template() -> Dict[str, Any]:
        """Create minimal configuration template with only required fields."""
        return {
            "dataset": {"name": "run0", "raw_file": "/path/to/your/data.vhdr"},
            "cropping": {"tmin": 0.0, "tmax": 600.0},
            "filtering": {"highpass": 1.0, "lowpass": 40.0},
            "channels": {"drop": [], "bads": []},
            "ica": {
                "random_state": 1,
                "method": "infomax",
                "rejection_thr": 0.9,
                "exclude_components": [],
            },
            "output": {"save_dir": "results/preprocessed/", "save_format": "fif.gz"},
        }


class BatchConfigManager:
    """Manage configurations for batch processing multiple runs."""

    def __init__(self, base_config_path: str = None):
        self.base_config = None
        if base_config_path:
            with open(base_config_path, "r") as f:
                self.base_config = yaml.safe_load(f)

    def create_configs_for_files(
        self, raw_files: List[str], output_dir: str = "configs/"
    ) -> List[str]:
        """
        Create configuration files for multiple raw files.

        Returns:
            List of created configuration file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        created_configs = []

        for i, raw_file in enumerate(raw_files):
            # Generate run name
            run_name = f"run{i}" if len(raw_files) > 1 else "run0"

            # Create config
            if self.base_config:
                config = deepcopy(self.base_config)
                config["dataset"]["name"] = run_name
                config["dataset"]["raw_file"] = str(Path(raw_file).resolve())
            else:
                config = ConfigTemplate.create_template_for_file(raw_file, run_name)

            # Save config
            config_path = output_dir / f"config_{run_name}.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)

            created_configs.append(str(config_path))
            print(f"Created config: {config_path}")

        return created_configs

    def update_all_configs(self, pattern: str, updates: Dict[str, Any]) -> None:
        """
        Update multiple configuration files matching a pattern.

        Args:
            pattern: Glob pattern for config files (e.g., "configs/config_*.yaml")
            updates: Dictionary of updates to apply
        """
        config_files = list(Path().glob(pattern))

        for config_file in config_files:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            # Apply updates
            self._apply_updates(config, updates)

            # Save updated config
            with open(config_file, "w") as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)

            print(f"Updated: {config_file}")

    def _apply_updates(self, config: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Apply updates to configuration using dot notation."""
        for path, value in updates.items():
            keys = path.split(".")
            current = config

            # Navigate to parent
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Set value
            current[keys[-1]] = value


def validate_config_file(config_path: str) -> None:
    """Validate a configuration file and print results."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        validator = ConfigValidator()
        is_valid, errors, warnings = validator.validate(config)

        print(f"\nValidation results for {config_path}:")
        print(f"Valid: {is_valid}")

        if errors:
            print("\nErrors:")
            for error in errors:
                print(f"  - {error}")

        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  - {warning}")

        if is_valid and not warnings:
            print("✓ Configuration is valid!")

    except Exception as e:
        print(f"Error loading config file: {e}")


def create_template_config(output_path: str, template_type: str = "base") -> None:
    """Create a template configuration file."""
    if template_type == "base":
        template = ConfigTemplate.create_base_template()
    elif template_type == "minimal":
        template = ConfigTemplate.create_minimal_template()
    else:
        raise ValueError(f"Unknown template type: {template_type}")

    with open(output_path, "w") as f:
        yaml.dump(template, f, default_flow_style=False, indent=2)

    print(f"Created {template_type} template: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EEG Configuration Utilities")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate configuration file"
    )
    validate_parser.add_argument("config", help="Configuration file to validate")

    # Template command
    template_parser = subparsers.add_parser(
        "template", help="Create configuration template"
    )
    template_parser.add_argument("output", help="Output file path")
    template_parser.add_argument(
        "--type", choices=["base", "minimal"], default="base", help="Template type"
    )

    # Batch command
    batch_parser = subparsers.add_parser(
        "batch", help="Create configs for multiple files"
    )
    batch_parser.add_argument(
        "files", nargs="+", help="Raw files to create configs for"
    )
    batch_parser.add_argument(
        "--output-dir", default="configs/", help="Output directory"
    )
    batch_parser.add_argument(
        "--base-config", help="Base configuration to use as template"
    )

    args = parser.parse_args()

    if args.command == "validate":
        validate_config_file(args.config)
    elif args.command == "template":
        create_template_config(args.output, args.type)
    elif args.command == "batch":
        manager = BatchConfigManager(args.base_config)
        manager.create_configs_for_files(args.files, args.output_dir)
    else:
        parser.print_help()
