"""
Configuration and calibration set management module.

This module contains:
1. Configuration classes for calibration sets:
   - DatasetEntryConfig: Configuration for a single dataset entry
   - CalibrationSetConfig: Configuration for a complete calibration set

2. CalibrationSet class: The main class for loading, caching, and managing calibration datasets
"""

import dataclasses
import hashlib
import json
import logging
import re
from dataclasses import field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml  # type: ignore
from datasets import Dataset, concatenate_datasets, load_dataset
from jinja2 import Environment, StrictUndefined

from .formatters import DatasetFmt

# Default sample limit for streaming datasets when "all" is requested
STREAMING_DEFAULT_SAMPLE_LIMIT = 256


@dataclasses.dataclass
class DatasetEntryConfig:
    """Single dataset entry in calibration set.

    Mandatory fields:
    - dataset: HuggingFace dataset identifier (required)
    - split: Dataset split (required, default "train")
    - columns: List of column names (required)
    - formatter: Format converter name (required)
    - num_samples: Number of samples to use (required, positive integer or "all")

    Optional fields:
    - subset: Dataset subset (optional, None for no subset)
    - streaming: Whether to use streaming when loading (optional, default False)
    - formatter_params: Additional parameters to pass to the formatter (optional)
    """

    def __init__(
        self,
        dataset: str,
        formatter: str,
        split: str = "train",
        subset: Optional[str] = None,
        columns: Optional[List[str]] = None,
        num_samples: Optional[Union[int, str]] = None,
        streaming: bool = False,
        formatter_params: Optional[Dict[str, Any]] = None,
    ):
        self.dataset = dataset
        self.split = split
        self.subset = subset
        self.columns = columns or []
        self.formatter = formatter
        self.num_samples = num_samples
        self.streaming = streaming
        self.formatter_params = formatter_params or {}
        self.validate()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetEntryConfig":
        dataset = data.get("dataset", "")
        if not dataset:
            raise ValueError("Dataset is required in calibration entry")

        split = data.get("split")
        if not split:
            raise ValueError("Split is required in calibration entry")
        subset = data.get("subset")
        columns = data.get("columns", [])
        if not isinstance(columns, list):
            raise ValueError(f"columns must be a list, got {type(columns)}")

        formatter = data.get("formatter")
        if not formatter:
            raise ValueError("formatter is required in calibration entry")

        num_samples = data.get("num_samples", None)
        if num_samples is None:
            raise ValueError("num_samples is required in calibration entry")
        elif isinstance(num_samples, str) and num_samples == "all":
            pass  # Special value to indicate use all samples
        elif not (isinstance(num_samples, int) and num_samples > 0):
            raise ValueError("num_samples must be a positive integer or 'all'")

        streaming = data.get("streaming", False)
        formatter_params = data.get("formatter_params", {})

        return cls(
            dataset=dataset,
            split=split,
            subset=subset,
            columns=columns,
            formatter=formatter,
            num_samples=num_samples,
            streaming=streaming,
            formatter_params=formatter_params,
        )

    def validate(self) -> None:
        if not self.dataset:
            raise ValueError("Dataset is required in calibration entry")
        if not self.split:
            raise ValueError("Split is required in calibration entry")
        if self.num_samples is None:
            raise ValueError("num_samples is required in calibration entry")
        if not (isinstance(self.num_samples, str) and self.num_samples == "all") and not (
            isinstance(self.num_samples, int) and self.num_samples > 0
        ):
            raise ValueError("num_samples must be a positive integer or 'all'")
        if not self.columns:
            raise ValueError("columns list cannot be empty")
        if not self.formatter:
            raise ValueError("formatter is required in calibration entry")

        # Validate formatter_params that look like Jinja templates
        if self.formatter_params:
            for key, value in self.formatter_params.items():
                if isinstance(value, str) and "{{" in value and "}}" in value:
                    self._validate_template_modulus(value)

    def _validate_template_modulus(self, template_str: str) -> None:
        """Validate template modulus operations at config load time.

        Args:
            template_str: The template string to validate

        Raises:
            ValueError: If the template has a list index error
        """
        if "[hash(row|string)" in template_str and "]" in template_str:
            modulus_match = re.search(r"\[hash\(row\|string\)\s*%\s*(\d+)\]", template_str)
            if modulus_match:
                modulus_val = int(modulus_match.group(1))
                list_match = re.search(r"(\[.*?\])\s*\[hash\(row\|string\)\s*%\s*\d+\]", template_str)
                if list_match:
                    list_str = list_match.group(1)
                    elements = [item.strip() for item in list_str.strip("[]").split(",") if item.strip()]
                    if len(elements) < modulus_val:
                        raise ValueError(
                            f"List index error: Template has list with {len(elements)} elements "
                            f"but uses modulus {modulus_val}, which can cause index errors at runtime."
                        )

    def resolve_num_samples(self, dataset_name: str, dataset: Any) -> int:
        """Resolve num_samples based on actual dataset size.

        Args:
            dataset_name: Name of the dataset for logging.
            dataset: HuggingFace Dataset or IterableDataset to get actual size from.

        Returns:
            int: Resolved number of samples.
        """
        # For streaming datasets, we can't get length, so use requested number directly
        try:
            actual_size = len(dataset)
        except TypeError:
            # Streaming IterableDataset has no len()
            actual_size = None

        # If "all" was requested but we can't determine size (streaming), use a default
        if isinstance(self.num_samples, str) and self.num_samples == "all":
            if actual_size is not None:
                requested_num = actual_size
            else:
                # For streaming datasets with "all", set to a reasonable default
                requested_num = STREAMING_DEFAULT_SAMPLE_LIMIT
                log = logging.getLogger(__name__)
                log.info(
                    f"Using default sample count of {requested_num} for streaming "
                    f"dataset {dataset_name} ('all' requested but length unknown)"
                )
        else:
            # Use the requested number if it's a positive integer
            requested_num = int(self.num_samples if self.num_samples is not None else 0)

        # Cap at actual dataset size if requested exceeds what's available (only for non-streaming)
        if actual_size is not None and requested_num > actual_size:
            log = logging.getLogger(__name__)
            log.warning(
                f"Requested {requested_num} samples from {dataset_name}, "
                f"but only {actual_size} available. Using all available samples."
            )
            return actual_size

        return requested_num


@dataclasses.dataclass
class CalibrationSetConfig:
    """Calibration set with shared shuffle/seed and list of datasets."""

    max_seq_length: int = 4096
    shuffle: bool = True
    seed: int = 42
    datasets: List[DatasetEntryConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationSetConfig":
        if isinstance(data, str):
            return cls.from_file(data)

        # The file MUST have "calibration_set" key at the root level
        if "calibration_set" not in data:
            raise ValueError("Configuration must have 'calibration_set' key at the root level")
        calib_data = data["calibration_set"]

        datasets_data = calib_data.get("datasets", [])
        return cls(
            max_seq_length=calib_data.get("max_seq_length", 4096),
            shuffle=calib_data.get("shuffle", True),
            seed=calib_data.get("seed", 42),
            datasets=[DatasetEntryConfig.from_dict(ds) for ds in datasets_data],
        )

    @classmethod
    def from_file(cls, path: str) -> "CalibrationSetConfig":
        """Load calibration set from YAML file."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise ValueError(f"Calibration set not found: {path}")

        with open(path_obj, "r") as f:
            data = yaml.safe_load(f) or {}

        # The file MUST have "calibration_set" key containing the config
        if "calibration_set" not in data:
            raise ValueError(f"Calibration set file must have 'calibration_set' key: {path}")

        # Call from_dict with the full data dict, which must have a "calibration_set" key at the root
        return cls.from_dict(data)

    def validate(self) -> None:
        if not self.datasets:
            raise ValueError("Calibration set must have at least one dataset")
        for ds in self.datasets:
            ds.validate()


class CalibrationSet:
    """Container for calibration dataset with its configuration.

    This class represents a calibration set with its configuration and provides
    methods to load, cache, and process calibration datasets.

    CRITICAL SEPARATION OF CONCERNS:

    1. DATA LOADING & CACHING:
       - `from_cache()`: Load from cached (untokenized) data
       - `from_config()`: Build from raw data and consolidate (untokenized)
       - `save_to_cache()`: Save consolidated (untokenized) data to cache
       - `compute_cache_key()`: Generate deterministic cache key from config

    2. DATA CONSOLIDATION/Formatting:
       - `_consolidate_datasets()`: Loads, formats, and filters raw datasets
       - Stores only untokenized data to maintain proper caching

    3. TOKENIZATION:
       - `get_tokenized()`: Applies tokenization to already consolidated data
       - Called only when tokenization is needed, not during caching

    WHY THIS MATTERS:
    - Uncached datasets stored with tokenization cannot be reused with different tokenizers
    - Cache keys must be deterministic and independent of tokenization parameters
    - Same cached dataset should be usable across different quantization configurations
    """

    def __init__(
        self,
        config: CalibrationSetConfig,
        cache_dir: str = "./cache",
        _from_factory: bool = False,
    ):
        """Initialize CalibrationSet.

        Args:
            config: CalibrationSetConfig to initialize with.
            cache_dir: Directory to cache calibration sets.
            _from_factory: Internal flag to indicate creation from factory method.

        Raises:
            RuntimeError: If CalibrationSet is instantiated directly instead of using factory methods.

        Note:
            Users should use CalibrationSet.from_config() or CalibrationSet.from_cache()
            instead of directly instantiating this class.
        """
        if not _from_factory:
            raise RuntimeError(
                "CalibrationSet should not be instantiated directly. "
                "Use CalibrationSet.from_config() or CalibrationSet.from_cache() instead."
            )

        self.config = config
        self.cache_dir = Path(cache_dir)
        self._untokenized_calibration_set = None

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def is_cached(config: "CalibrationSetConfig", cache_dir: str = "./cache") -> bool:
        """Check if calibration set is already cached.

        Args:
            config: CalibrationSetConfig to check for cached version.
            cache_dir: Directory to check for cached calibration sets.

        Returns:
            bool: True if cached data exists and matches config, False otherwise.

        Raises:
            ValueError: If the configuration is invalid.
        """
        # Validate configuration first
        config.validate()

        cache_key = CalibrationSet.compute_cache_key(config)
        cache_path = Path(cache_dir) / cache_key
        return cache_path.exists()

    @classmethod
    def from_cache(cls, config: CalibrationSetConfig, cache_dir: str = "./cache") -> "CalibrationSet":
        """Factory method to load calibration set from cache if available.

        Args:
            config: CalibrationSetConfig to load for.
            cache_dir: Directory to cache calibration sets.

        Returns:
            CalibrationSet instance if cached data exists and matches config.
            This does not load the dataset into memory - use get_tokenized() for that.

        Raises:
            ValueError: If the configuration is invalid.
        """
        # Validate configuration first
        config.validate()

        # Create instance using constructor with factory flag
        instance = cls(config, cache_dir, _from_factory=True)

        # Try to load from cache
        cache_key = CalibrationSet.compute_cache_key(config)
        cache_path = instance.cache_dir / cache_key

        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cache not found for configuration. Use CalibrationSet.is_cached() to check "
                f"before loading, or CalibrationSet.from_config() to build from raw data. "
                f"Expected cache file: {cache_path}"
            )

        try:
            logging.info(f"Loading from cache: {cache_path}")
            # Load directly as HuggingFace Dataset from Parquet
            dataset = Dataset.from_parquet(str(cache_path))

            # Validate the loaded dataset has samples
            if len(dataset) == 0:
                raise ValueError(f"Cache found but empty: {cache_path}")
            else:
                instance._untokenized_calibration_set = dataset
        except Exception as e:
            raise RuntimeError(f"Failed to load cache file {cache_path}: {e}") from e

        return instance

    @classmethod
    def from_config(cls, config: CalibrationSetConfig, cache_dir: str = "./cache") -> "CalibrationSet":
        """Factory method to create calibration set from raw data.

        This method loads and consolidates datasets according to configuration,
        storing only untokenized data to enable proper caching. Tokenization
        should be applied using get_tokenized() when needed.

        Args:
            config: CalibrationSetConfig to build for.
            cache_dir: Directory to cache calibration sets.

        Returns:
            CalibrationSet instance with loaded and consolidated dataset (untokenized).

        Raises:
            ValueError: If the configuration is invalid.
        """
        # Validate configuration first
        config.validate()

        # Create instance using constructor with factory flag
        instance = cls(config, cache_dir, _from_factory=True)

        # Build the dataset from raw data (without tokenization)
        instance._consolidate_datasets()

        return instance

    @staticmethod
    def compute_cache_key(config: CalibrationSetConfig) -> str:
        """Generate deterministic cache key from calibration config.

        Format: <first-7-hex-char>-<total_samples>.parquet

        Args:
            config: CalibrationSetConfig to generate cache key for.

        Returns:
            str: Cache key that can be used for filesystem operations.
        """
        # Create canonical representation (columns as tuple for deterministic ordering)
        dataset_configs = []
        for idx, ds in enumerate(config.datasets):
            dataset_configs.append(
                (
                    ds.dataset,
                    ds.split,
                    ds.subset,
                    ds.num_samples,
                    tuple(ds.columns),  # Tuple for hash stability
                    ds.formatter,
                )
            )

        # Sort by dataset name, split, subset to ensure deterministic ordering
        dataset_configs.sort(key=lambda x: (x[0], x[1], str(x[2])))

        canonical = {
            "datasets": dataset_configs,
            "max_seq_length": config.max_seq_length,
            "shuffle": config.shuffle,
            "seed": config.seed,
        }

        # Generate hash
        json_str = json.dumps(canonical, sort_keys=True)
        hash_prefix = hashlib.sha256(json_str.encode()).hexdigest()[:7]

        # Calculate total samples for the filename
        total_samples = 0
        total_samples_str = ""
        for ds in config.datasets:
            if isinstance(ds.num_samples, str) and ds.num_samples == "all":
                total_samples_str = "length_TBD"
                break
            elif isinstance(ds.num_samples, int) and ds.num_samples > 0:
                total_samples += ds.num_samples
            else:
                raise ValueError(f"Invalid sample count in dataset {ds.dataset}: {ds.num_samples}")
        if total_samples_str == "":
            total_samples_str = str(total_samples)

        return f"{hash_prefix}-{total_samples_str}.parquet"

    # _build_tokenized_dataset method removed - replaced by _consolidate_datasets() and get_tokenized()
    # methods to maintain proper separation of concerns between data loading and tokenization.

    def _consolidate_datasets(self) -> Dataset:
        """Consolidate datasets from a config file

        Loads each dataset according to the configuration, applies formatting,
        filters samples, and concatenates all datasets into a single Dataset.
        The resulting dataset contains formatted messages ready for tokenization.

        Returns:
            Dataset: The combined dataset with formatted data.
        """
        all_datasets = []

        for ds_config in self.config.datasets:
            # Load dataset from HuggingFace,
            # nosec because we have to download from remote
            # user is expected to do due diligence

            # TODO: log the dataset name and number (progress bar?)
            if isinstance(ds_config.dataset, str):
                if ds_config.subset is not None:
                    # nosec B615, B703
                    dataset = load_dataset(  # nosec
                        ds_config.dataset, ds_config.subset, split=ds_config.split, streaming=ds_config.streaming
                    )
                else:
                    # nosec B615, B703
                    dataset = load_dataset(  # nosec
                        ds_config.dataset, split=ds_config.split, streaming=ds_config.streaming
                    )
            else:
                # Handle cases where dataset is a tuple of multiple arguments
                # nosec B615, B703
                if ds_config.subset is not None:
                    dataset = load_dataset(
                        ds_config.dataset[0], ds_config.subset, split=ds_config.split, streaming=ds_config.streaming
                    )  # nosec
                else:
                    dataset = load_dataset(ds_config.dataset[0], split=ds_config.split, streaming=ds_config.streaming)  # nosec

            # Resolve number of samples based on actual dataset size
            num_samples = ds_config.resolve_num_samples(ds_config.dataset, dataset)

            # Filter to number of samples requested
            if ds_config.num_samples != "all":
                dataset = dataset.filter(lambda e, i: i < num_samples, with_indices=True)

            # Apply formatter function
            formatter_func = DatasetFmt.get_formatter(ds_config.formatter)

            # Setup Jinja environment for template rendering with Python built-ins
            # Note: autoescape=True is kept as the default for security reasons. After evaluating
            # the risks and benefits, we're keeping this setting as discussed in:
            # https://github.com/mratsim/quantizers/pull/6#pullrequestreview-3619760720
            # The current templates only contain language names (e.g., Python, C++, F#) which
            # don't contain HTML special characters that would be escaped.
            jinja_env = Environment(undefined=StrictUndefined, autoescape=True)
            # Add Python built-ins to Jinja context
            jinja_env.globals.update(
                {
                    "hash": hash,
                    "len": len,
                    "abs": abs,
                    "max": max,
                    "min": min,
                    "sum": sum,
                    "sorted": sorted,
                    "enumerate": enumerate,
                    "zip": zip,
                }
            )

            # Process formatter parameters that might contain Jinja templates
            processed_params = {}
            if ds_config.formatter_params:
                for key, value in ds_config.formatter_params.items():
                    if isinstance(value, str) and "{{" in value and "}}" in value:
                        # This is a Jinja template, render it later
                        processed_params[key] = value
                    else:
                        processed_params[key] = value

            # Helper function to render Jinja templates
            def render_template(template_str, row):
                try:
                    template = jinja_env.from_string(template_str)
                    return template.render(row=row)
                except Exception as e:
                    logging.error(f"Failed to render Jinja template '{template_str}': {e}")
                    raise

            # Apply formatting directly using named function to avoid lambda variable capture issues
            # Pass formatter parameters if they exist
            def apply_formatter(row):
                # Render any Jinja templates in the parameters
                rendered_params = {}
                for key, value in processed_params.items():
                    if isinstance(value, str) and "{{" in value and "}}" in value:
                        rendered_value = render_template(value, row)
                        rendered_params[key] = rendered_value
                    else:
                        rendered_params[key] = value

                if rendered_params:
                    result = formatter_func(ds_config.columns, row, **rendered_params)
                else:
                    result = formatter_func(ds_config.columns, row)
                return {"formatted": result}

            # Note: a function indirection is absolutely necessary due to Python lambda scoping rules
            # https://github.com/mratsim/quantizers/pull/2#discussion_r2653416980
            # This would lead to incorrect type 'Dict' instead of 'List' in certain cases
            #       dataset = dataset.map(
            #           lambda row: {"formatted": formatter_func(ds_config.columns, row)},
            #           remove_columns=dataset.column_names,
            #       )
            dataset = dataset.map(
                apply_formatter,
                remove_columns=dataset.column_names,
            )

            # Convert streaming datasets to regular datasets to allow concatenation
            if ds_config.streaming:
                # For streaming datasets, take only the samples we need
                if ds_config.num_samples != "all":
                    # Take exactly the number of samples we need
                    dataset = dataset.take(ds_config.num_samples)
                else:
                    # For "all" with streaming, take a reasonable default
                    dataset = dataset.take(STREAMING_DEFAULT_SAMPLE_LIMIT)

                # Convert iterable dataset to regular dataset by creating a list and then a Dataset
                dataset = list(dataset)  # Collect all items
                dataset = Dataset.from_dict({"formatted": [d["formatted"] for d in dataset]})
            else:
                # For non-streaming, ensure we have exactly the samples we need
                if isinstance(ds_config.num_samples, int) and len(dataset) > ds_config.num_samples:  # type: ignore[operator, arg-type]
                    # The mypy type ignore is intentional here because:
                    # 1. `isinstance(ds_config.num_samples, int)` ensures num_samples is an int
                    # 2. MyPy can't properly track the narrowed type after isinstance checks
                    dataset = dataset.select(range(ds_config.num_samples))  # type: ignore[arg-type]

            # Store for later concatenation
            all_datasets.append(dataset)

        if not all_datasets:
            raise ValueError("No datasets were processed - this should not happen")

        # Concatenate all datasets
        result = concatenate_datasets(all_datasets)

        # Apply shuffling if requested
        if self.config.shuffle:
            result = result.shuffle(seed=self.config.seed)

        self._untokenized_calibration_set = result
        return result

    def _tokenize_row(self, row, tokenizer) -> Dict[str, Any]:
        """Tokenize a single row using the specified formatter.

        Args:
            row: HuggingFace dataset row with formatted messages
            tokenizer: Tokenizer to use for tokenization

        Returns:
            dict: Tokenized data with input_ids
        """
        # Get the formatted messages from the row
        messages = row["formatted"]

        # Tokenize using chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokenized = tokenizer(
            text,
            padding=False,
            max_length=self.config.max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )

        return tokenized

    def get_tokenized(self, tokenizer) -> Dataset:
        """Get the tokenized calibration dataset.

        This method applies tokenization to the already consolidated dataset,
        maintaining clear separation between data loading/consolidation and tokenization.
        If the dataset isn't loaded yet (from cache), throws an error.

        Args:
            tokenizer: Tokenizer to use for tokenization.

        Returns:
            HuggingFace Dataset with tokenized calibration data.
        """
        if self._untokenized_calibration_set is None:
            raise RuntimeError(
                "Calibration dataset is not loaded. "
                "Use CalibrationSet.from_cache() with cached data or "
                "CalibrationSet.from_config() to build from raw data."
            )

        # Apply tokenization to the entire dataset
        tokenized_dataset = self._untokenized_calibration_set.map(
            lambda row: self._tokenize_row(row, tokenizer=tokenizer),
            batched=False,
            remove_columns=self._untokenized_calibration_set.column_names,
        )

        return tokenized_dataset

    @property
    def total_num_samples(self) -> int:
        """Get the total number of samples in the calibration set.

        Returns:
            int: Number of samples, or 0 if dataset is not loaded.
        """
        if self._untokenized_calibration_set is None:
            return 0
        return len(self._untokenized_calibration_set)

    def save_to_cache(self) -> None:
        """Save calibration set to cache.

        Saves the current _untokenized_calibration_set to the cache.
        """
        if self._untokenized_calibration_set is None:
            raise RuntimeError("No calibration dataset to save. Ensure dataset is available before calling save_to_cache().")

        if len(self._untokenized_calibration_set) == 0:
            logging.warning("Cannot save empty dataset to cache")
            return

        cache_key = CalibrationSet.compute_cache_key(self.config)
        cache_path = self.cache_dir / cache_key

        try:
            logging.info(f"Saving to cache: {cache_path}")
            # Use Dataset.to_parquet to save directly as Parquet
            self._untokenized_calibration_set.to_parquet(str(cache_path))

            # Update total_num_samples is now a property, no need to set it
        except Exception as e:
            logging.error(f"Failed to save cache file {cache_path}: {e}")
