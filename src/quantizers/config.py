"""
Configuration loading for quantization pipeline.

Supports:
- Model configs: name, revision
- Calibration sets: dataset specs with shared shuffle/seed
- Recipes: quantization settings (path passed directly to llm-compressor)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml  # type: ignore

from .calibration_sets import CalibrationSetConfig


@dataclass
class ModelConfig:
    """Model configuration."""

    name: str
    revision: str = "main"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(name=data.get("name", ""), revision=data.get("revision", "main"))

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Model name is required")


@dataclass
class QuantizationConfig:
    """Quantization configuration (recipe path only)."""

    recipe: str
    calibration_set: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantizationConfig":
        return cls(recipe=data.get("recipe", ""), calibration_set=data.get("calibration_set"))

    def validate(self) -> None:
        if not self.recipe:
            raise ValueError("Recipe path is required")


@dataclass
class QuantizationRunConfig:
    """Complete quantization run configuration."""

    model: ModelConfig
    quantization: QuantizationConfig
    calibration_set_config: Optional[CalibrationSetConfig] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], config_path: Optional[str] = None) -> "QuantizationRunConfig":
        model = ModelConfig.from_dict(data.get("model", {}))
        quant_data = data.get("quantization", {})
        quantization = QuantizationConfig.from_dict(quant_data)

        # Load calibration set if referenced
        calib_set_config = None
        calib_set_path = quantization.calibration_set or data.get("calibration_set")
        if calib_set_path:
            if not calib_set_path.endswith(".yaml"):
                calib_set_path = calib_set_path + ".yaml"
            # Make path relative to the main config file if it's not absolute
            if not Path(calib_set_path).is_absolute() and config_path:
                main_config_path = Path(config_path)
                parent_dir = main_config_path.parent

                # Handle paths starting with "configs/"
                configs_prefix = "configs/"
                if calib_set_path.startswith(configs_prefix):
                    # Skip the "configs/" prefix to avoid duplication
                    relative_path = calib_set_path[len(configs_prefix) :]  # Remove prefix using length
                    calib_set_path = str(parent_dir / relative_path)
                else:
                    calib_set_path = str(parent_dir / calib_set_path)

            calib_set_config = CalibrationSetConfig.from_file(calib_set_path)

        return cls(
            model=model,
            quantization=quantization,
            calibration_set_config=calib_set_config,
        )

    def validate(self) -> None:
        self.model.validate()
        self.quantization.validate()
        if self.calibration_set_config:
            self.calibration_set_config.validate()


def load_yaml(path: str) -> Dict[str, Any]:
    """Load and parse YAML file."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise ValueError(f"Config file not found: {path}")
    with open(path_obj, "r") as f:
        return yaml.safe_load(f) or {}


def load_quantization_config(config_path: str) -> QuantizationRunConfig:
    """Load complete quantization configuration."""
    config = load_yaml(config_path)
    run_config = QuantizationRunConfig.from_dict(config, config_path)
    run_config.validate()
    return run_config
