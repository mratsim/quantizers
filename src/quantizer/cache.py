"""
Dataset caching layer for quantization calibration.

Caches calibration sets at the raw dataset level (before tokenization) to enable
reuse across models with different tokenizers. This is the key caching layer since
loading/downloading datasets is significantly more expensive than tokenization.

Cache Structure:
    ./cache/
        {calibration_set_hash}.pt

The cache is keyed by the calibration set configuration, enabling reuse across
multiple models. Cache validation includes verifying dataset size matches expected.

Before tokenization:
- Check if cache exists for this calibration set
- If yes, verify size matches expected (from yaml config)
- If no, load from HuggingFace and cache
"""

import os
import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional, Callable, Any, Union, List, Dict

from datasets import Dataset, load_dataset, concatenate_datasets


def canonicalize_config(config: Dict[str, Any]) -> str:
    """
    Convert config dict to a stable, minified string for cache key.
    
    Removes comments and normalizes ordering for consistent cache keys.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        JSON string representation
    """
    # Sort keys for consistent ordering
    sorted_config = json.dumps(config, sort_keys=True, separators=(',', ':'))
    return sorted_config


def get_calibration_set_hash(calibration_config: Dict[str, Any]) -> str:
    """
    Generate a hash key for a calibration set configuration.
    
    Args:
        calibration_config: Calibration set configuration dict
        
    Returns:
        Short hash string for cache naming
    """
    canonical = canonicalize_config(calibration_config)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class CalibrationSetCache:
    """
    Cache manager for calibration sets.
    
    Caches the combined, filtered, and shuffled calibration dataset at the
    raw level (before tokenization). This maximizes cache reuse since:
    1. Loading from HuggingFace is expensive (network + I/O)
    2. Filtering and shuffling is deterministic
    3. Tokenization depends on model-specific tokenizer
    
    Cache validation includes verifying the dataset size matches the expected
    size from the calibration set configuration.
    """
    
    def __init__(
        self,
        cache_dir: str = "./cache",
        overwrite: bool = False
    ):
        """
        Initialize calibration set cache.
        
        Args:
            cache_dir: Root directory for cache files
            overwrite: If True, ignore existing cache and regenerate
        """
        self.cache_dir = Path(cache_dir)
        self.overwrite = overwrite
        self._cache_dir_created = False
    
    def _ensure_cache_dir(self) -> Path:
        """Ensure cache directory exists."""
        if not self._cache_dir_created:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_dir_created = True
        return self.cache_dir
    
    def _get_cache_path(self, cache_hash: str) -> Path:
        """
        Generate cache file path from hash.
        
        Args:
            cache_hash: Hash string for the calibration set
            
        Returns:
            Path to cache file
        """
        self._ensure_cache_dir()
        return self.cache_dir / f"calib_{cache_hash}.pt"
    
    def _validate_cache(
        self,
        cache_path: Path,
        expected_size: int,
        calibration_config: Dict[str, Any]
    ) -> tuple[bool, Optional[Dataset]]:
        """
        Validate existing cache and optionally load it.
        
        Validation checks:
        1. Cache file exists
        2. Cache hash matches current config
        3. Dataset size matches expected size
        
        Args:
            cache_path: Path to potential cache file
            expected_size: Expected number of samples from config
            calibration_config: Calibration set configuration
            
        Returns:
            Tuple of (is_valid, dataset or None)
        """
        if not cache_path.exists():
            return False, None
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verify hash matches
            config_hash = get_calibration_set_hash(calibration_config)
            stored_hash = cache_data.get('_config_hash', '')
            
            if stored_hash != config_hash:
                return False, None
            
            dataset = cache_data.get('dataset')
            if dataset is None:
                return False, None
            
            # Verify size matches expected
            actual_size = len(dataset)
            if actual_size != expected_size:
                print(
                    f"Cache size mismatch: expected {expected_size}, "
                    f"got {actual_size}. Regenerating."
                )
                return False, None
            
            return True, dataset
            
        except (pickle.PickleError, KeyError, OSError, TypeError) as e:
            print(f"Cache validation failed: {e}")
            return False, None
    
    def load_or_create(
        self,
        calibration_config: Dict[str, Any],
        format_fn: Optional[Callable] = None,
        text_column: str = "messages"
    ) -> Dataset:
        """
        Load calibration set from cache or create if needed.
        
        This is the main entry point for calibration set loading. It handles:
        1. Generating cache key from calibration config
        2. Checking for existing valid cache
        3. Loading from HuggingFace if no cache
        4. Filtering, shuffling, and formatting
        5. Caching the result
        
        Args:
            calibration_config: Calibration set configuration dict with keys:
                - dataset: HuggingFace dataset name or (name, config) tuple
                - split: Dataset split (e.g., "train")
                - num_samples: Expected number of samples
                - shuffle: Whether to shuffle (bool)
                - seed: Random seed for shuffling
                - column: Column name for data (optional)
            format_fn: Optional function to format rows before caching
            text_column: Column name for formatted text in output
            
        Returns:
            Calibration dataset ready for tokenization
        """
        cache_hash = get_calibration_set_hash(calibration_config)
        cache_path = self._get_cache_path(cache_hash)
        expected_size = calibration_config.get("num_samples", 64)
        
        # Try to load from cache
        if not self.overwrite:
            is_valid, dataset = self._validate_cache(
                cache_path, expected_size, calibration_config
            )
            if is_valid:
                print(f"Using cached calibration set: {cache_path.name}")
                return dataset
        
        # Need to create cache
        print(f"Creating calibration set: hash={cache_hash}")
        
        dataset_spec = calibration_config["dataset"]
        split = calibration_config.get("split", "train")
        num_samples = calibration_config.get("num_samples", 64)
        shuffle = calibration_config.get("shuffle", True)
        seed = calibration_config.get("seed", 42)
        column = calibration_config.get("column")
        
        # Load from HuggingFace
        if isinstance(dataset_spec, str):
            ds = load_dataset(dataset_spec, split=split)
        else:
            # Multi-config dataset: tuple of (name, config)
            ds = load_dataset(dataset_spec[0], dataset_spec[1], split=split)
        
        print(f"  Loaded {len(ds)} samples from {dataset_spec}")
        
        # Shuffle before sampling
        if shuffle:
            ds = ds.shuffle(seed=seed)
        
        # Sample to required size
        if num_samples and num_samples < len(ds):
            ds = ds.select(range(num_samples))
        
        # Apply formatting if provided
        if format_fn is not None:
            def apply_format(example):
                messages = format_fn(example)
                return {text_column: messages}
            ds = ds.map(apply_format)
        elif column is not None:
            # Rename column to standard name
            if column != text_column:
                ds = ds.rename_column(column, text_column)
        
        # Verify final size
        actual_size = len(ds) if ds is not None else 0
        if actual_size != expected_size:
            raise ValueError(
                f"Size mismatch: requested {expected_size} samples, "
                f"got {actual_size}"
            )
        
        # Cache the result
        cache_data = {
            '_config_hash': cache_hash,
            '_config': calibration_config,
            'dataset': ds
        }
        
        print(f"  Caching to: {cache_path.name}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        return ds
    
    def load_multiple(
        self,
        calibration_configs: List[Dict[str, Any]],
        format_fn: Optional[Callable] = None,
        text_column: str = "messages",
        final_shuffle: bool = True,
        final_seed: Optional[int] = None
    ) -> Dataset:
        """
        Load and combine multiple calibration sets.
        
        Args:
            calibration_configs: List of calibration configurations
            format_fn: Optional formatting function applied to each row
            text_column: Column name for formatted text
            final_shuffle: Whether to shuffle the combined dataset
            final_seed: Seed for final shuffle (uses first config's seed if None)
            
        Returns:
            Combined calibration dataset
        """
        datasets = []
        
        for config in calibration_configs:
            ds = self.load_or_create(config, format_fn, text_column)
            datasets.append(ds)
            print(f"  Loaded {len(ds)} samples")
        
        # Concatenate all datasets
        combined = concatenate_datasets(datasets)
        
        # Final shuffle if requested
        if final_shuffle:
            shuffle_seed = final_seed or calibration_configs[0].get("seed", 42)
            combined = combined.shuffle(seed=shuffle_seed)
        
        return combined
    
    def clear_cache(self, cache_hash: Optional[str] = None) -> int:
        """
        Clear cached calibration sets.
        
        Args:
            cache_hash: If provided, only clear this specific cache
            
        Returns:
            Number of cache files removed
        """
        if cache_hash:
            cache_path = self._get_cache_path(cache_hash)
            if cache_path.exists():
                cache_path.unlink()
                print(f"Cleared cache: {cache_path.name}")
                return 1
            return 0
        else:
            # Clear all
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                print("Cleared all calibration set cache")
            return -1
    
    def get_cache_info(self, calibration_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get information about a cached calibration set.
        
        Args:
            calibration_config: Calibration set configuration
            
        Returns:
            Dict with cache status and info
        """
        cache_hash = get_calibration_set_hash(calibration_config)
        cache_path = self._get_cache_path(cache_hash)
        expected_size = calibration_config.get("num_samples", 0)
        
        exists = cache_path.exists()
        is_valid = False
        actual_size = 0
        
        if exists:
            is_valid, dataset = self._validate_cache(
                cache_path, expected_size, calibration_config
            )
            if is_valid:
                actual_size = len(dataset)
        
        return {
            "cache_hash": cache_hash,
            "cache_path": str(cache_path),
            "exists": exists,
            "valid": is_valid,
            "expected_size": expected_size,
            "actual_size": actual_size
        }


# =====================================================================
# Convenience functions for common patterns
# =====================================================================

def load_calibration_set(
    calibration_config: Dict[str, Any],
    format_fn: Optional[Callable] = None,
    cache_dir: str = "./cache",
    overwrite_cache: bool = False
) -> Dataset:
    """
    Convenience function to load a calibration set.
    
    Args:
        calibration_config: Calibration set configuration dict
        format_fn: Optional function to format rows
        cache_dir: Cache directory path
        overwrite_cache: If True, ignore existing cache
        
    Returns:
        Calibration dataset ready for tokenization
    """
    cache = CalibrationSetCache(cache_dir=cache_dir, overwrite=overwrite_cache)
    return cache.load_or_create(calibration_config, format_fn)


def load_calibration_sets(
    calibration_configs: List[Dict[str, Any]],
    format_fn: Optional[Callable] = None,
    cache_dir: str = "./cache",
    overwrite_cache: bool = False
) -> Dataset:
    """
    Load multiple calibration sets and combine them.
    
    Args:
        calibration_configs: List of calibration configurations
        format_fn: Optional formatting function
        cache_dir: Cache directory path
        overwrite_cache: If True, ignore existing cache
        
    Returns:
        Combined calibration dataset
    """
    cache = CalibrationSetCache(cache_dir=cache_dir, overwrite=overwrite_cache)
    return cache.load_multiple(calibration_configs, format_fn)