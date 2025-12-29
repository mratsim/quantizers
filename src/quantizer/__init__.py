# src/quantizer/__init__.py
"""
Quantization pipeline library built on llm-compressor.

This package provides a clean, parameterized interface for post-training
quantization of large language models.

Main Components:
- config: Configuration loading and validation
- cache: Dataset caching layer
- logging_: Structured logging for runs
- formatters: Dataset format converters

Quick Start:
    from quantizer.config import load_quantization_config
    from quantizer.cache import load_multiple_calibration_datasets
    from quantizer.logging_ import setup_logging
    
    config = load_quantization_config("configs/quantize_model-gptq.yaml")
    
    with setup_logging(model_name=config.model.name, method=config.quantization.method) as logger:
        dataset = load_multiple_calibration_datasets(
            config.calibration_sets, tokenizer, config.inference.max_seq_length
        )
        # ... run quantization
"""

from .config import (
    load_quantization_config,
    load_yaml,
    load_recipe,
    QuantizationRunConfig,
    ModelConfig,
    QuantizationConfig,
    CalibrationConfig,
    CalibrationSetConfig,
    InferenceConfig,
    ConfigError,
    ConfigNotFoundError,
    ConfigValidationError,
)

from .cache import (
    DatasetCache,
    load_calibration_dataset,
    load_multiple_calibration_datasets,
)

from .logging_ import (
    QuantizationLogger,
    setup_logging,
    log_dataset_info,
    log_recipe_summary,
)

from .formatters import DatasetFmt

__all__ = [
    # Config
    "load_quantization_config",
    "load_yaml",
    "load_recipe",
    "QuantizationRunConfig",
    "ModelConfig", 
    "QuantizationConfig",
    "CalibrationConfig",
    "CalibrationSetConfig",
    "InferenceConfig",
    "ConfigError",
    "ConfigNotFoundError",
    "ConfigValidationError",
    # Cache
    "DatasetCache",
    "load_calibration_dataset",
    "load_multiple_calibration_datasets",
    # Logging
    "QuantizationLogger",
    "setup_logging",
    "log_dataset_info",
    "log_recipe_summary",
    # Formatters
    "DatasetFmt",
]

__version__ = "0.1.0"