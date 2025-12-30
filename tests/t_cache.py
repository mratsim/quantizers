#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for cache functionality.

This test file validates that the caching system works correctly for
calibration datasets, focusing on cache key generation and proper loading/saving
without getting into data formatting complexities.

To run these tests:

NOTE: Tests should NEVER be created in the tests/test_cache/ directory.
This file was moved from tests/test_cache/test_cache.py to tests/t_cache.py.
All future tests should follow the naming convention of t_*.py.
    uv run python tests/t_cache.py
"""

import sys
import tempfile
from pathlib import Path

import datasets

from quantizers.calibration_sets import (
    CalibrationSet,
    CalibrationSetConfig,
    DatasetEntryConfig,
)

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_cache_key_generation():
    """Test that cache keys are generated correctly."""
    print("\n=== Testing Cache Key Generation ===")

    # Create a test configuration
    config = CalibrationSetConfig(
        max_seq_length=4096,
        shuffle=True,
        seed=42,
        datasets=[
            DatasetEntryConfig(
                dataset="test/dataset1",
                split="train",
                columns=["messages"],
                formatter="sharegpt",
                num_samples=100,
            ),
            DatasetEntryConfig(
                dataset="test/dataset2",
                split="validation",
                subset="v1",
                columns=["input", "output"],
                formatter="prompt_answer",
                num_samples=50,
            ),
        ],
    )

    # Create CalibrationSet instance using factory method and generate cache key using static method
    cache_set = CalibrationSet.from_config(config=config, cache_dir="./cache")
    key = CalibrationSet.compute_cache_key(config)
    print(f"Generated cache key: {key}")

    # Verify format: first-7-char-hex-num_samples.parquet
    assert len(key) == 19, f"Key length is {len(key)}, expected 19"
    assert key.endswith(".parquet"), f"Key doesn't end with .parquet: {key}"

    parts = key[:-8]  # Remove .parquet
    hex_part = parts.split("-")[0]
    num_part = parts.split("-")[1]

    assert len(hex_part) == 7, f"Hex part length is {len(hex_part)}, expected 7"
    assert num_part.isdigit(), f"Number part is not digits: {num_part}"
    assert int(num_part) == 150, f"Expected 150 samples, got {num_part}"

    print("✅ Cache key generation test passed")


def test_cache_save_and_load():
    """Test saving and loading datasets from cache."""
    print("\n=== Testing Cache Save and Load ===")

    # Create test data with a simple format
    test_data = [{"text": f"Sample {i}"} for i in range(10)]

    # Create a test configuration
    config = CalibrationSetConfig(
        max_seq_length=4096,
        shuffle=False,
        seed=42,
        datasets=[
            DatasetEntryConfig(
                dataset="test/dataset",
                split="train",
                columns=["text"],
                formatter="raw_text",
                num_samples=10,
            )
        ],
    )

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create CalibrationSet instance using factory method and populate with test data
        cache_set = CalibrationSet.from_config(config=config, cache_dir=temp_dir)

        # Create Dataset object from test data with proper formatting
        test_dataset = datasets.Dataset.from_dict({"text": test_data}).map(
            lambda row: {"formatted": row["text"]}
        )

        # Set the dataset and save to cache
        cache_set._untokenized_calibration_set = test_dataset

        # Save data to cache
        cache_set.save_to_cache()
        print("✅ Dataset saved to cache")

        # Verify cache file exists
        cache_key = CalibrationSet.compute_cache_key(config)
        cache_file = Path(temp_dir) / cache_key
        assert cache_file.exists(), f"Cache file {cache_file} was not created"

        # Create a new instance and load from cache
        new_cache_set = CalibrationSet.from_cache(config, temp_dir)

        # Verify data was loaded correctly
        assert new_cache_set._untokenized_calibration_set is not None
        assert len(new_cache_set._untokenized_calibration_set) == len(test_data)

        print("✅ Dataset loaded from cache")

        # Create a different config but with valid datasets (not empty)
        different_config = CalibrationSetConfig(
            max_seq_length=8192,
            shuffle=False,
            seed=999,
            datasets=[
                DatasetEntryConfig(
                    dataset="test/dataset",
                    split="train",
                    columns=["text"],
                    formatter="raw_text",
                    num_samples=5,
                )
            ],
        )

        # Loading with different config should NOT raise ValueError
        # It should gracefully return an empty cache set since the cache keys won't match
        _different_cache_set = CalibrationSet.from_cache(different_config, temp_dir)
        # This should return a CalibrationSet with no cached data
        assert _different_cache_set._untokenized_calibration_set is None
        print("✅ Correctly returned None for cache with different config")


def test_cache_from_config_and_cache():
    """Test that creating from config caches the data."""
    print("\n=== Testing From Config Caching ===")

    # Create a simple test configuration
    config = CalibrationSetConfig(
        max_seq_length=4096,
        shuffle=False,
        seed=42,
        datasets=[
            DatasetEntryConfig(
                dataset="tests/test_datasets/raw_text",
                split="train",
                columns=["text"],
                formatter="raw_text",
                num_samples=3,
            )
        ],
    )

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create CalibrationSet from config (should build and cache data)
        cache_set = CalibrationSet.from_config(config, cache_dir=temp_dir)

        # Verify properties
        assert cache_set.config == config
        assert cache_set._untokenized_calibration_set is not None
        assert cache_set.total_num_samples == 3

        print(
            f"✅ Created calibration set with {cache_set.total_num_samples} total samples"
        )

        # Manually save to cache to ensure cache file is created
        cache_set.save_to_cache()
        print("✅ Dataset was cached successfully")

        # Verify cache file was created
        cache_key = CalibrationSet.compute_cache_key(config)
        cache_file = Path(temp_dir) / cache_key
        assert cache_file.exists(), f"Cache file {cache_file} was not created"

        # Create a new instance and load from cache
        cache_set_from_cache = CalibrationSet.from_cache(config, cache_dir=temp_dir)

        # Verify that the data was loaded from cache
        assert cache_set_from_cache._untokenized_calibration_set is not None
        assert len(cache_set_from_cache._untokenized_calibration_set) == 3

        print("✅ Successfully loaded from cache")


def test_cache_key_consistency():
    """Test that cache keys are consistent for the same configuration."""
    print("\n=== Testing Cache Key Consistency ===")

    # Create identical configurations
    config1 = CalibrationSetConfig(
        max_seq_length=4096,
        shuffle=True,
        seed=42,
        datasets=[
            DatasetEntryConfig(
                dataset="test/dataset",
                split="train",
                columns=["messages"],
                formatter="sharegpt",
                num_samples=10,
            )
        ],
    )

    config2 = CalibrationSetConfig(
        max_seq_length=4096,
        shuffle=True,
        seed=42,
        datasets=[
            DatasetEntryConfig(
                dataset="test/dataset",
                split="train",
                columns=["messages"],
                formatter="sharegpt",
                num_samples=10,
            )
        ],
    )

    # Generate cache keys for both configurations using static method
    key1 = CalibrationSet.compute_cache_key(config1)
    key2 = CalibrationSet.compute_cache_key(config2)

    # Keys should be identical
    assert key1 == key2, f"Keys differ: {key1} vs {key2}"
    print("✅ Cache keys are consistent for identical configurations")


def test_cache_different_configurations():
    """Test that different configurations generate different cache keys."""
    print("\n=== Testing Cache Keys for Different Configurations ===")

    # Create different configurations
    config1 = CalibrationSetConfig(
        max_seq_length=4096,
        shuffle=True,
        seed=42,
        datasets=[
            DatasetEntryConfig(
                dataset="test/dataset1",
                split="train",
                columns=["messages"],
                formatter="sharegpt",
                num_samples=10,
            )
        ],
    )

    config2 = CalibrationSetConfig(
        max_seq_length=8192,  # Different max_seq_length
        shuffle=True,
        seed=42,
        datasets=[
            DatasetEntryConfig(
                dataset="test/dataset1",
                split="train",
                columns=["messages"],
                formatter="sharegpt",
                num_samples=10,
            )
        ],
    )

    config3 = CalibrationSetConfig(
        max_seq_length=4096,
        shuffle=False,  # Different shuffle
        seed=42,
        datasets=[
            DatasetEntryConfig(
                dataset="test/dataset1",
                split="train",
                columns=["messages"],
                formatter="sharegpt",
                num_samples=10,
            )
        ],
    )

    # Generate cache keys using static method
    key1 = CalibrationSet.compute_cache_key(config1)
    key2 = CalibrationSet.compute_cache_key(config2)
    key3 = CalibrationSet.compute_cache_key(config3)

    # Keys should all be different
    assert key1 != key2, "Keys should differ with different max_seq_length"
    assert key1 != key3, "Keys should differ with different shuffle setting"
    assert key2 != key3, "Keys should differ between different configurations"

    print("✅ Cache keys are correctly different for different configurations")


def test_cache_key():
    """Test generation of cache keys."""
    print("\n=== Testing Cache Key Generation ===")

    # Create calibration set config
    config1 = CalibrationSetConfig(
        max_seq_length=1024,
        shuffle=True,
        seed=42,
        datasets=[
            DatasetEntryConfig(
                dataset="test/dataset1",
                split="train",
                columns=["messages"],
                formatter="chat_completion",
                num_samples=100,
            )
        ],
    )

    # Create second config with different parameters
    config2 = CalibrationSetConfig(
        max_seq_length=2048,
        shuffle=True,
        seed=42,
        datasets=[
            DatasetEntryConfig(
                dataset="test/dataset1",
                split="train",
                columns=["messages"],
                formatter="chat_completion",
                num_samples=100,
            )
        ],
    )

    # Create third config with different formatter
    config3 = CalibrationSetConfig(
        max_seq_length=1024,
        shuffle=True,
        seed=42,
        datasets=[
            DatasetEntryConfig(
                dataset="test/dataset1",
                split="train",
                columns=["prompt", "response"],
                formatter="prompt_answer",
                num_samples=100,
            )
        ],
    )

    # Generate cache keys
    key1 = CalibrationSet.compute_cache_key(config1)
    key2 = CalibrationSet.compute_cache_key(config2)
    key3 = CalibrationSet.compute_cache_key(config3)

    # Verify keys are different for different configurations
    assert (
        key1 != key2
    ), f"Cache keys should be different for different max_seq_length, got {key1} == {key2}"
    assert (
        key1 != key3
    ), f"Cache keys should be different for different formatters, got {key1} == {key3}"

    print("✅ Cache keys are correctly different for different configurations")


def test_cache_defensive_validation():
    """Test that from_cache, from_config, and is_cached validate configurations."""
    print("\n=== Testing Defensive Validation ===")

    # Create invalid config with empty datasets
    invalid_config = CalibrationSetConfig(
        max_seq_length=4096,
        shuffle=False,
        seed=42,
        datasets=[],  # Empty datasets - should fail validation
    )

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test that is_cached validates
        try:
            _ = CalibrationSet.is_cached(invalid_config, temp_dir)
            assert False, "is_cached should raise ValueError for invalid config"
        except ValueError as e:
            assert "at least one dataset" in str(e)
            print("✅ is_cached correctly validates configuration")

        # Test that from_cache validates
        try:
            _ = CalibrationSet.from_cache(invalid_config, temp_dir)
            assert False, "from_cache should raise ValueError for invalid config"
        except ValueError as e:
            assert "at least one dataset" in str(e)
            print("✅ from_cache correctly validates configuration")

        # Test that from_config validates
        try:
            _ = CalibrationSet.from_config(invalid_config, temp_dir)
            assert False, "from_config should raise ValueError for invalid config"
        except ValueError as e:
            assert "at least one dataset" in str(e)
            print("✅ from_config correctly validates configuration")


if __name__ == "__main__":
    print("🔍 Running cache functionality tests...")

    try:
        test_cache_key()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        print("✅ Cache functionality tests completed")
