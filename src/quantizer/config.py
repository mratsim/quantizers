# src/quantizer/config.py
"""
Configuration loading and validation for quantization pipeline.

Supports loading:
- Model configs: model name, revision, sequence length
- Calibration sets: dataset specs, sample counts, shuffling
- Recipes: quantization modifiers and settings

Configuration files are YAML-based and referenced from main config files.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field


class ConfigError(Exception):
    """Base exception for configuration errors."""
    pass


class ConfigNotFoundError(ConfigError):
    """Raised when a config file cannot be found."""
    pass


class ConfigValidationError(ConfigError):
    """Raised when config validation fails."""
    pass


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    name: str
    revision: str = "main"
    max_seq_length: int = 4096
    trust_remote_code: bool = False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create from dictionary (YAML parsed data)."""
        return cls(
            name=data.get("name", ""),
            revision=data.get("revision", "main"),
            max_seq_length=data.get("max_seq_length", 4096),
            trust_remote_code=data.get("trust_remote_code", False)
        )
    
    def validate(self) -> None:
        """Validate configuration."""
        if not self.name:
            raise ConfigValidationError("Model name is required")
        if self.max_seq_length <= 0:
            raise ConfigValidationError("max_seq_length must be positive")


@dataclass
class CalibrationSetConfig:
    """Calibration dataset configuration."""
    dataset: Union[str, List[str]]
    split: str = "train"
    num_samples: int = 512
    column: Optional[str] = None
    reformatting_fn: Optional[str] = None
    shuffle: bool = True
    seed: int = 42
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationSetConfig":
        """Create from dictionary (YAML parsed data)."""
        # Handle dataset as string or list (for multi-config datasets)
        dataset = data.get("dataset")
        if isinstance(dataset, str):
            pass  # Keep as string
        elif isinstance(dataset, list):
            dataset = tuple(dataset)  # Convert to tuple for consistency
        
        return cls(
            dataset=dataset,
            split=data.get("split", "train"),
            num_samples=data.get("num_samples", 512),
            column=data.get("column"),
            reformatting_fn=data.get("reformatting_fn"),
            shuffle=data.get("shuffle", True),
            seed=data.get("seed", 42)
        )
    
    def validate(self) -> None:
        """Validate configuration."""
        if not self.dataset:
            raise ConfigValidationError("Dataset is required for calibration set")
        if self.num_samples <= 0:
            raise ConfigValidationError("num_samples must be positive")
        if self.split and not isinstance(self.split, str):
            raise ConfigValidationError("split must be a string")


@dataclass 
class QuantizationConfig:
    """Quantization method and recipe configuration."""
    method: str
    recipe: str
    scheme: Optional[str] = None
    recipe_args: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantizationConfig":
        """Create from dictionary (YAML parsed data)."""
        return cls(
            method=data.get("method", ""),
            recipe=data.get("recipe", ""),
            scheme=data.get("scheme"),
            recipe_args=data.get("recipe_args")
        )
    
    def validate(self) -> None:
        """Validate configuration."""
        valid_methods = {"gptq", "awq", "fp8", "nvfp4", "nvfp4a16", "fp8_dynamic", "fp8_block"}
        if self.method.lower() not in valid_methods:
            raise ConfigValidationError(
                f"Invalid quantization method: {self.method}. "
                f"Valid methods: {valid_methods}"
            )
        if not self.recipe:
            raise ConfigValidationError("Recipe is required")


@dataclass
class InferenceConfig:
    """Inference/generation parameters."""
    max_seq_length: int = 4096
    batch_size: int = 1
    do_sample: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceConfig":
        """Create from dictionary (YAML parsed data)."""
        return cls(
            max_seq_length=data.get("max_seq_length", 4096),
            batch_size=data.get("batch_size", 1),
            do_sample=data.get("do_sample", False),
            temperature=data.get("temperature"),
            top_p=data.get("top_p"),
            min_p=data.get("min_p")
        )
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.max_seq_length <= 0:
            raise ConfigValidationError("max_seq_length must be positive")
        if self.batch_size <= 0:
            raise ConfigValidationError("batch_size must be positive")


@dataclass
class CalibrationConfig:
    """Calibration settings for oneshot quantization."""
    num_samples: int = 512
    shuffle: bool = True
    seed: int = 42
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationConfig":
        """Create from dictionary (YAML parsed data)."""
        if data is None:
            data = {}
        return cls(
            num_samples=data.get("num_samples", 512),
            shuffle=data.get("shuffle", True),
            seed=data.get("seed", 42)
        )
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.num_samples <= 0:
            raise ConfigValidationError("num_samples must be positive")


@dataclass
class QuantizationRunConfig:
    """
    Complete configuration for a quantization run.
    
    Combines model, quantization, inference, and calibration settings.
    
    Supports two modes for calibration configuration:
    - `calibration_set`: Single path to a calibration set file (recommended)
    - `calibration_sets`: List of inline calibration set configurations
    """
    model: ModelConfig
    quantization: QuantizationConfig
    inference: InferenceConfig
    calibration: Optional[CalibrationConfig] = None
    calibration_set: Optional[str] = None  # Path to calibration set file
    calibration_sets: List[CalibrationSetConfig] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantizationRunConfig":
        """Create from dictionary (YAML parsed data)."""
        # Handle calibration_set (path to file) vs calibration_sets (inline)
        calibration_set = data.get("calibration_set")
        calibration_sets_data = data.get("calibration_sets", [])
        
        # Validate that only one is provided
        if calibration_set and calibration_sets_data:
            raise ConfigValidationError(
                "Cannot specify both 'calibration_set' and 'calibration_sets'. "
                "Use 'calibration_set' for single file reference, "
                "'calibration_sets' for inline configuration."
            )
        
        return cls(
            model=ModelConfig.from_dict(data.get("model", {})),
            quantization=QuantizationConfig.from_dict(data.get("quantization", {})),
            inference=InferenceConfig.from_dict(data.get("inference", {})),
            calibration=CalibrationConfig.from_dict(data.get("calibration")),
            calibration_set=calibration_set,
            calibration_sets=[
                CalibrationSetConfig.from_dict(cs) 
                for cs in calibration_sets_data
            ]
        )
    
    def validate(self) -> None:
        """Validate entire configuration."""
        self.model.validate()
        self.quantization.validate()
        self.inference.validate()
        
        if self.calibration:
            self.calibration.validate()
        
        # Validate that calibration is configured
        if not self.calibration_set and not self.calibration_sets:
            raise ConfigValidationError(
                "Must specify either 'calibration_set' or 'calibration_sets'"
            )
        
        for cs in self.calibration_sets:
            cs.validate()


def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load and parse a YAML configuration file.
    
    Args:
        path: Path to YAML file (relative or absolute)
        
    Returns:
        Parsed configuration dictionary
        
    Raises:
        ConfigNotFoundError: If file doesn't exist
        ConfigError: If parsing fails
    """
    path = Path(path)
    
    if not path.exists():
        # Try relative to current working directory
        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            path = cwd_path
        else:
            raise ConfigNotFoundError(f"Config file not found: {path}")
    
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse config file {path}: {e}")


def resolve_path(base_dir: str, relative_path: str) -> str:
    """
    Resolve a relative path against a base directory.
    
    Args:
        base_dir: Base directory path
        relative_path: Path relative to base_dir
        
    Returns:
        Resolved absolute path
    """
    base = Path(base_dir)
    
    # If already absolute, return as-is
    if Path(relative_path).is_absolute():
        return relative_path
    
    # Resolve against base directory
    resolved = (base / relative_path).resolve()
    
    # Also check if it exists relative to CWD
    if not resolved.exists():
        cwd_resolved = (Path.cwd() / relative_path).resolve()
        if cwd_resolved.exists():
            return str(cwd_resolved)
    
    return str(resolved)


def load_recipe(recipe_path: str, config_root: str = "./configs/recipes") -> Dict[str, Any]:
    """
    Load a recipe configuration file.
    
    Args:
        recipe_path: Path to recipe file (relative or absolute)
        config_root: Root directory for recipes
        
    Returns:
        Recipe configuration dictionary
        
    Raises:
        ConfigNotFoundError: If recipe file doesn't exist
    """
    # Try direct path first
    if Path(recipe_path).exists():
        return load_yaml(recipe_path)
    
    # Try resolving against config root
    resolved = resolve_path(config_root, recipe_path)
    if Path(resolved).exists():
        return load_yaml(resolved)
    
    # Try with .yaml extension
    if not recipe_path.endswith(".yaml"):
        yaml_path = resolved + ".yaml"
        if Path(yaml_path).exists():
            return load_yaml(yaml_path)
    
    raise ConfigNotFoundError(f"Recipe not found: {recipe_path}")


def load_calibration_set(
    config: Union[str, Dict],
    config_root: str = "./configs/calibration_sets"
) -> Dict[str, Any]:
    """
    Load a calibration set configuration.
    
    Args:
        config: Either a path string or a dict with calibration set spec
        config_root: Root directory for calibration sets
        
    Returns:
        Calibration set configuration dictionary
    """
    if isinstance(config, dict):
        return config
    
    # It's a path string
    if Path(config).exists():
        return load_yaml(config)
    
    resolved = resolve_path(config_root, config)
    if Path(resolved).exists():
        return load_yaml(resolved)
    
    # Try with .yaml extension
    if not config.endswith(".yaml"):
        yaml_path = resolved + ".yaml"
        if Path(yaml_path).exists():
            return load_yaml(yaml_path)
    
    raise ConfigNotFoundError(f"Calibration set not found: {config}")


def load_quantization_config(
    config_path: str,
    config_root: str = "./configs"
) -> QuantizationRunConfig:
    """
    Load a complete quantization configuration file.
    
    This loads the main model config which may reference:
    - Recipe files in ./configs/recipes/
    - Calibration sets in ./configs/calibration_sets/
    
    Args:
        config_path: Path to quantization config YAML
        config_root: Root directory for configs
        
    Returns:
        Fully resolved QuantizationRunConfig
        
    Raises:
        ConfigNotFoundError: If config file doesn't exist
        ConfigError: If validation fails
    """
    # Load main config
    if Path(config_path).exists():
        main_config = load_yaml(config_path)
        config_dir = str(Path(config_path).parent)
    else:
        resolved = resolve_path(config_root, config_path)
        main_config = load_yaml(resolved)
        config_dir = str(Path(resolved).parent)
    
    # Resolve calibration sets
    calibration_sets = []
    for cs in main_config.get("calibration_sets", []):
        if isinstance(cs, dict):
            # Inline config
            calibration_sets.append(cs)
        else:
            # Reference to calibration set file
            cs_config = load_calibration_set(cs, config_root + "/calibration_sets")
            calibration_sets.append(cs_config)
    
    main_config["calibration_sets"] = calibration_sets
    
    # Create config object
    config = QuantizationRunConfig.from_dict(main_config)
    
    # Validate
    config.validate()
    
    return config


def get_output_dirname(config: QuantizationRunConfig) -> str:
    """
    Generate output directory name from configuration.
    
    Args:
        config: Quantization run configuration
        
    Returns:
        Directory name for outputs
    """
    model_name = config.model.name.split("/")[-1]
    method = config.quantization.method.upper()
    return f"{model_name}-{method}"


def save_config(
    config: QuantizationRunConfig, 
    output_path: str,
    include_recipe: bool = True,
    recipe_path: Optional[str] = None
) -> None:
    """
    Save a configuration to YAML file.
    
    Args:
        config: Configuration to save
        output_path: Path for output file
        include_recipe: If True, embed recipe content
        recipe_path: Path to recipe file (required if include_recipe=True)
    """
    data = {
        "model": {
            "name": config.model.name,
            "revision": config.model.revision,
            "max_seq_length": config.model.max_seq_length
        },
        "quantization": {
            "method": config.quantization.method,
            "recipe": config.quantization.recipe,
            "scheme": config.quantization.scheme
        },
        "inference": {
            "max_seq_length": config.inference.max_seq_length,
            "batch_size": config.inference.batch_size
        }
    }
    
    if config.calibration:
        data["calibration"] = {
            "num_samples": config.calibration.num_samples,
            "shuffle": config.calibration.shuffle,
            "seed": config.calibration.seed
        }
    
    if config.calibration_sets:
        data["calibration_sets"] = [
            {
                "dataset": cs.dataset if isinstance(cs.dataset, str) else list(cs.dataset),
                "split": cs.split,
                "num_samples": cs.num_samples,
                "column": cs.column,
                "seed": cs.seed
            }
            for cs in config.calibration_sets
        ]
    
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)