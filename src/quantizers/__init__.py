"""
Quantization pipeline library built on llm-compressor.

This package provides a clean, parameterized interface for post-training
quantization of large language models.

Main Components:
- config: Configuration loading and validation
- calibration_sets: Dataset loading and caching
- logging_: Structured logging for runs
- formatters: Dataset format converters

Quick Start:
    from quantizers.config import load_quantization_config, CalibrationSetConfig
    from quantizers.calibration_sets import CalibrationSet

    config = load_quantization_config("configs/quantize_model.yaml")
    calib_set = CalibrationSet.from_config(config.calibration_set_config)
    dataset = calib_set.get_tokenized(tokenizer)
"""

from .config import (
    CalibrationSetConfig,
    ModelConfig,
    QuantizationConfig,
    QuantizationRunConfig,
    load_quantization_config,
    load_yaml,
)
from .formatters import DatasetFmt

__all__ = [
    # Config
    "load_quantization_config",
    "load_yaml",
    "QuantizationRunConfig",
    "ModelConfig",
    "QuantizationConfig",
    "CalibrationSetConfig",
    # CalibrationSets
    "CalibrationSet",
    # Formatters
    "DatasetFmt",
]
