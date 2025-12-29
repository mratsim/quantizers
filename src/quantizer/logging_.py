# src/quantizer/logging_.py
"""
Logging setup and utilities for quantization runs.

Creates timestamped log files in ./logs/ directory with:
- Main execution log: <datetime>-<model>-<method>.log
- Configuration YAML: <datetime>-<model>-<method>-config.yaml
- Recipe YAML: <datetime>-<model>-<method>-recipe.yaml
- Calibration set YAML: <datetime>-<model>-<method>-calibration.yaml

The naming convention ensures easy ordering and prevents overwrites.
"""

import os
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TextIO
from contextlib import contextmanager
from io import StringIO


class QuantizationLogger:
    """
    Logger for quantization runs.
    
    Manages main log file and parallel YAML files for config, recipe,
    and calibration set. All files use consistent timestamp-based naming.
    
    Attributes:
        log_dir: Directory for log files
        log_name: Base name for log files (without extension)
        log_file: Path to main log file
        config_file: Path to config YAML file
        recipe_file: Path to recipe YAML file
        calibration_file: Path to calibration YAML file
        start_time: When the run started (for timing)
    """
    
    def __init__(
        self,
        log_dir: str = "./logs",
        model_name: str = "unknown",
        method: str = "unknown",
        timestamp: Optional[datetime] = None
    ):
        """
        Initialize logger with timestamped file paths.
        
        Args:
            log_dir: Directory for log files
            model_name: Model identifier for log naming
            method: Quantization method for log naming
            timestamp: Optional datetime (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Generate timestamp prefix: YYYY-MM-DD_HH-MM-SS
        ts_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        
        # Sanitize model name for filesystem
        safe_model = model_name.replace("/", "_").replace(":", "-")
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_name = f"{ts_str}-{safe_model}-{method}"
        self.log_file = self.log_dir / f"{self.base_name}.log"
        self.config_file = self.log_dir / f"{self.base_name}-config.yaml"
        self.recipe_file = self.log_dir / f"{self.base_name}-recipe.yaml"
        self.calibration_file = self.log_dir / f"{self.base_name}-calibration.yaml"
        
        self.start_time = timestamp
        self._log_file_handle: Optional[TextIO] = None
    
    def _ensure_log_dir(self) -> None:
        """Ensure log directory exists."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def open(self) -> None:
        """Open log file for writing."""
        self._ensure_log_dir()
        self._log_file_handle = open(self.log_file, 'w', encoding='utf-8')
    
    def close(self) -> None:
        """Close log file."""
        if self._log_file_handle:
            self._log_file_handle.close()
            self._log_file_handle = None
    
    def log(self, message: str, level: str = "INFO") -> None:
        """
        Write a message to the main log file and stdout.
        
        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR, DEBUG)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {message}"
        
        # Write to file
        if self._log_file_handle:
            self._log_file_handle.write(formatted + "\n")
            self._log_file_handle.flush()
        
        # Write to stdout
        print(formatted)
    
    def log_section(self, title: str) -> None:
        """Log a section header."""
        separator = "=" * 60
        self.log(separator)
        self.log(f"  {title}")
        self.log(separator)
    
    def log_dict(self, title: str, data: Dict[str, Any], level: str = "INFO") -> None:
        """
        Log a dictionary with title.
        
        Args:
            title: Section title
            data: Dictionary to log
            level: Log level
        """
        self.log(f"{title}:", level)
        try:
            yaml_str = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
            for line in yaml_str.strip().split("\n"):
                self.log(f"  {line}", level)
        except Exception:
            # Fallback to repr if YAML fails
            self.log(f"  {repr(data)}", level)
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary to save
        """
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        self.log(f"Saved config to: {self.config_file}")
    
    def save_recipe(self, recipe: Dict[str, Any]) -> None:
        """
        Save recipe to YAML file.
        
        Args:
            recipe: Recipe dictionary to save
        """
        with open(self.recipe_file, 'w') as f:
            yaml.dump(recipe, f, default_flow_style=False, sort_keys=False)
        self.log(f"Saved recipe to: {self.recipe_file}")
    
    def save_calibration_set(self, calibration: Dict[str, Any]) -> None:
        """
        Save calibration set to YAML file.
        
        Args:
            calibration: Calibration configuration to save
        """
        with open(self.calibration_file, 'w') as f:
            yaml.dump(calibration, f, default_flow_style=False, sort_keys=False)
        self.log(f"Saved calibration set to: {self.calibration_file}")
    
    def log_launch_command(self, command: str) -> None:
        """Log the launch command."""
        self.log_section("LAUNCH COMMAND")
        self.log(command)
    
    def log_model_info(self, model_name: str, revision: str = "main") -> None:
        """Log model information."""
        self.log_section("MODEL")
        self.log(f"Name: {model_name}")
        self.log(f"Revision: {revision}")
    
    def log_quantization_info(self, method: str, scheme: Optional[str] = None) -> None:
        """Log quantization method and scheme."""
        self.log_section("QUANTIZATION")
        self.log(f"Method: {method}")
        if scheme:
            self.log(f"Scheme: {scheme}")
    
    def log_calibration_info(
        self,
        num_samples: int,
        shuffle: bool = True,
        seed: int = 42
    ) -> None:
        """Log calibration settings."""
        self.log_section("CALIBRATION")
        self.log(f"Num samples: {num_samples}")
        self.log(f"Shuffle: {shuffle}")
        self.log(f"Seed: {seed}")
    
    def log_step(self, step: str, status: str = "STARTED") -> None:
        """
        Log a processing step.
        
        Args:
            step: Step name
            status: Status (STARTED, COMPLETED, FAILED, SKIPPED)
        """
        self.log(f"Step: {step} [{status}]", level="INFO")
    
    def log_exception(self, exception: Exception, context: str = "") -> None:
        """
        Log an exception with context.
        
        Args:
            exception: The exception that occurred
            context: Additional context about where it happened
        """
        self.log(f"ERROR in {context}: {type(exception).__name__}", level="ERROR")
        self.log(str(exception), level="ERROR")
        if hasattr(exception, '__traceback__'):
            import traceback
            tb = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
            self.log(f"Traceback:\n{tb}", level="ERROR")
    
    def log_timing(self, operation: str, duration_seconds: float) -> None:
        """
        Log timing information.
        
        Args:
            operation: Name of operation
            duration_seconds: Duration in seconds
        """
        if duration_seconds < 60:
            self.log(f"{operation}: {duration_seconds:.2f}s")
        elif duration_seconds < 3600:
            self.log(f"{operation}: {duration_seconds/60:.2f}min")
        else:
            self.log(f"{operation}: {duration_seconds/3600:.2f}h")
    
    def get_log_path(self) -> Path:
        """Get path to main log file."""
        return self.log_file
    
    def __enter__(self) -> "QuantizationLogger":
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if exc_type is not None:
            self.log_exception(exc_val, context="quantization_run")
        self.close()


@contextmanager
def setup_logging(
    log_dir: str = "./logs",
    model_name: str = "unknown",
    method: str = "unknown",
    launch_command: Optional[str] = None
):
    """
    Context manager for setting up logging.
    
    Usage:
        with setup_logging(model_name="meta-llama/Llama-3.3-70B", method="gptq") as logger:
            logger.log("Starting quantization...")
            ...
    
    Args:
        log_dir: Directory for log files
        model_name: Model identifier
        method: Quantization method
        launch_command: Optional command line that launched this run
        
    Yields:
        QuantizationLogger instance
    """
    logger = QuantizationLogger(log_dir, model_name, method)
    logger.open()
    
    try:
        logger.log_section("QUANTIZATION RUN")
        if launch_command:
            logger.log_launch_command(launch_command)
        
        yield logger
        
    finally:
        logger.log_section("RUN COMPLETED")
        logger.close()


def log_dataset_info(logger: QuantizationLogger, datasets_info: list) -> None:
    """
    Log information about calibration datasets.
    
    Args:
        logger: Logger instance
        datasets_info: List of dicts with dataset information
    """
    logger.log_section("DATASETS")
    for i, ds_info in enumerate(datasets_info):
        logger.log(f"Dataset {i+1}:")
        logger.log(f"  Name: {ds_info.get('name', 'unknown')}")
        logger.log(f"  Split: {ds_info.get('split', 'unknown')}")
        logger.log(f"  Samples: {ds_info.get('num_samples', 'unknown')}")
        logger.log(f"  Column: {ds_info.get('column', 'none')}")


def log_recipe_summary(logger: QuantizationLogger, recipe: Dict[str, Any]) -> None:
    """
    Log a summary of the quantization recipe.
    
    Args:
        logger: Logger instance
        recipe: Recipe configuration dict
    """
    logger.log_section("RECIPE SUMMARY")
    
    # Extract key info from recipe
    scheme = recipe.get("quantization_scheme", {})
    if isinstance(scheme, dict):
        logger.log(f"Type: {scheme.get('type', 'unknown')}")
        logger.log(f"Targets: {scheme.get('targets', [])}")
        if "config" in scheme:
            logger.log(f"Config: {scheme['config']}")
    
    # Log modifiers
    modifiers = recipe.get("modifiers", [])
    if modifiers:
        logger.log(f"Modifiers: {len(modifiers)}")


def redirect_logging_to_string() -> StringIO:
    """
    Temporarily redirect stdout/stderr to a StringIO for capturing output.
    
    Returns:
        StringIO object that captured output is written to
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    captured = StringIO()
    sys.stdout = captured
    sys.stderr = captured
    
    return captured, old_stdout, old_stderr


def restore_logging(old_stdout: TextIO, old_stderr: TextIO, captured: StringIO) -> str:
    """
    Restore stdout/stderr and return captured content.
    
    Args:
        old_stdout: Original stdout
        old_stderr: Original stderr
        captured: StringIO that captured output
        
    Returns:
        Captured output as string
    """
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    return captured.getvalue()