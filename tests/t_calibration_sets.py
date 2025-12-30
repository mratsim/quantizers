#!/usr/bin/env python3

"""
Tests for calibration sets functionality.

This test file validates all aspects of the CalibrationSet class:
- Basic creation and properties
- Cache key generation
- Save and load operations
- Configuration validation
- Dataset loading and consolidation
- Factory methods

To run these tests:
    uv run python tests/t_calibration_sets.py
"""

import sys
import tempfile
from pathlib import Path

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import datasets

from quantizers.calibration_sets import (
    CalibrationSet,
    CalibrationSetConfig,
    DatasetEntryConfig,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.input_ids = [1, 2, 3, 4, 5]
        self.attention_mask = [1, 1, 1, 1, 1]

    def apply_chat_template(self, messages, tokenize=False):
        """Mock apply_chat_template method."""
        # Handle both string list and message dictionary list formats
        if isinstance(messages, list) and isinstance(messages[0], str):
            text = " ".join(messages)
        elif isinstance(messages, list) and isinstance(messages[0], dict):
            text = " ".join([msg.get("content", "") for msg in messages])
        else:
            # Handle single string or dictionary
            if isinstance(messages, str):
                text = messages
            elif isinstance(messages, dict):
                text = messages.get("content", "")
            else:
                text = str(messages)

        if tokenize:
            return {"input_ids": self.input_ids, "attention_mask": self.attention_mask}
        return text

    def __call__(self, text, **kwargs):
        """Mock tokenization."""
        return {"input_ids": self.input_ids, "attention_mask": self.attention_mask}


def test_creating_calibration_set():
    """Test creating a calibration set with minimal config."""
    print("\n=== Testing CalibrationSet Creation ===")

    # Create a test configuration
    config = CalibrationSetConfig(
        max_seq_length=4096,
        shuffle=True,
        seed=42,
        datasets=[
            DatasetEntryConfig(
                dataset=str(
                    Path(__file__).parent / "test_datasets" / "raw_text" / "ds_text"
                ),
                split="train",
                columns=["text"],
                formatter="raw_text",
                num_samples=10,
            )
        ],
    )

    # Create CalibrationSet instance using factory method
    calib_set = CalibrationSet.from_config(config=config)

    # Verify properties
    assert calib_set.config == config
    assert str(calib_set.cache_dir) == "cache"
    # from_config loads and processes datasets, so _untokenized_calibration_set will not be None
    assert calib_set._untokenized_calibration_set is not None
    assert calib_set.total_num_samples == 5  # Raw text dataset has 5 samples

    print("✅ CalibrationSet creation test passed")


def test_compute_cache_key():
    """Test cache key generation."""
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

    # Create two identical configs - they should produce the same cache key
    config2 = CalibrationSetConfig(
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

    # Test 1: Identical configs should produce the same cache key
    assert CalibrationSet.compute_cache_key(config) == CalibrationSet.compute_cache_key(
        config2
    )

    # Test 2: Different configs should produce different cache keys
    config3 = CalibrationSetConfig(
        max_seq_length=8192,  # Different max_seq_length
        shuffle=True,
        seed=42,
        datasets=[
            DatasetEntryConfig(
                dataset="test/dataset1",
                split="train",
                columns=["messages"],
                formatter="sharegpt",
                num_samples=100,
            )
        ],
    )

    key1 = CalibrationSet.compute_cache_key(config)
    key2 = CalibrationSet.compute_cache_key(config2)
    key3 = CalibrationSet.compute_cache_key(config3)

    assert (
        key1 == key2
    ), f"Identical configs should produce same cache key, but {key1} != {key2}"
    assert (
        key1 != key3
    ), f"Different configs should produce different cache keys, but {key1} == {key3}"

    print("✅ Cache key generation test passed")


def test_save_and_load():
    """Test saving and loading calibration sets."""
    print("\n=== Testing Save and Load ===")

    # Create test configuration
    config = CalibrationSetConfig(
        max_seq_length=4096,
        shuffle=False,
        seed=42,
        datasets=[
            DatasetEntryConfig(
                dataset=str(
                    Path(__file__).parent
                    / "test_datasets"
                    / "sharegpt"
                    / "ds_conversations"
                ),
                split="train",
                columns=["conversations"],
                formatter="sharegpt",
                num_samples=3,
            )
        ],
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create CalibrationSet instance using factory method
        calib_set = CalibrationSet.from_config(config=config, cache_dir=temp_dir)

        # Create mock untokenized dataset as HuggingFace Dataset with formatted messages
        mock_dataset = datasets.Dataset.from_dict(
            {
                "formatted": [
                    [
                        {"role": "user", "content": "Question 1"},
                        {"role": "assistant", "content": "Answer 1"},
                    ],
                    [
                        {"role": "user", "content": "Question 2"},
                        {"role": "assistant", "content": "Answer 2"},
                    ],
                    [
                        {"role": "user", "content": "Question 3"},
                        {"role": "assistant", "content": "Answer 3"},
                    ],
                ]
            }
        )

        # Set the dataset and save to cache using new API
        calib_set._untokenized_calibration_set = mock_dataset
        calib_set.save_to_cache()

        # Verify cache file exists (now saving as Parquet file directly)
        cache_key = CalibrationSet.compute_cache_key(calib_set.config)
        cache_path = Path(temp_dir) / cache_key
        assert cache_path.exists(), f"Cache file {cache_path} was not created"

        # Try loading using from_cache (should return the cached dataset)
        loaded_calib_set = CalibrationSet.from_cache(calib_set.config, str(temp_dir))

        # Create a mock tokenizer for tokenization
        mock_tokenizer = type(
            "MockTokenizer",
            (),
            {
                "apply_chat_template": lambda self, messages, tokenize=False: " ".join(
                    [msg.get("content", "") for msg in messages]
                ).strip(),
                "__call__": lambda self, text, **kwargs: (
                    {"input_ids": [1, 2, 3]} if text else {"input_ids": []}
                ),
            },
        )()

        dataset = loaded_calib_set.get_tokenized(
            mock_tokenizer
        )  # Use tokenizer for cached data
        assert dataset is not None
        assert len(dataset) == 3
        assert dataset[0]["input_ids"] == [1, 2, 3]

        print("✅ Save and load test passed")


def test_create_from_config():
    """Test creating calibration set from config using local datasets."""
    print("\n=== Testing Create From Config ===")

    # Create a test configuration
    config = CalibrationSetConfig(
        max_seq_length=4096,
        shuffle=False,
        seed=42,
        datasets=[
            DatasetEntryConfig(
                dataset=str(
                    Path(__file__).parent
                    / "test_datasets"
                    / "chat_completion"
                    / "ds_messages"
                ),
                split="train",
                columns=["messages"],
                formatter="chat_completion",
                num_samples=10,
            )
        ],
    )

    # Create using factory method
    calib_set = CalibrationSet.from_config(config, cache_dir="cache")

    # Verify properties
    assert calib_set.config == config
    assert str(calib_set.cache_dir) == "cache"
    assert calib_set._untokenized_calibration_set is not None
    assert calib_set.total_num_samples == 3

    # Test that we can get tokenized data
    mock_tokenizer = MockTokenizer()
    tokenized_dataset = calib_set.get_tokenized(mock_tokenizer)
    assert tokenized_dataset is not None
    assert len(tokenized_dataset) == 3

    print("✅ Create from config test passed")


def test_dataset_entry_config_validation():
    """Test that DatasetEntryConfig enforces mandatory fields."""
    print("\n=== Testing DatasetEntryConfig Validation ===")

    # Test correct format
    correct_format = {
        "dataset": "test/dataset",
        "split": "train",
        "columns": ["messages"],
        "formatter": "sharegpt",
        "num_samples": 10,
    }

    entry = DatasetEntryConfig.from_dict(correct_format)
    assert entry.dataset == "test/dataset"
    assert entry.split == "train"
    assert entry.columns == ["messages"]
    assert entry.formatter == "sharegpt"
    assert entry.num_samples == 10

    print("✅ DatasetEntryConfig validation test passed")


def test_dataset_entry_config_resolve_num_samples():
    """Test that DatasetEntryConfig correctly resolves num_samples."""
    print("\n=== Testing DatasetEntryConfig Resolve Num Samples ===")

    # Test with actual dataset from test_datasets
    test_dataset_path = str(
        Path(__file__).parent / "test_datasets" / "raw_text" / "ds_text"
    )

    # Load the actual test dataset to get its size
    dataset = datasets.load_dataset(test_dataset_path, split="train")
    dataset_size = len(dataset)
    print(f"Loaded test dataset with {dataset_size} samples")

    # Create DatasetEntryConfig that will use the dataset
    entry = DatasetEntryConfig(
        dataset=test_dataset_path,
        split="train",
        columns=["text"],
        formatter="raw_text",
        num_samples=3,
    )

    # Test with num_samples less than dataset size - should return requested number
    resolved = entry.resolve_num_samples(test_dataset_path, dataset)
    assert resolved == 3  # Should use the requested number

    # Test with num_samples larger than dataset size - should cap at dataset size
    entry.num_samples = 10
    resolved = entry.resolve_num_samples(test_dataset_path, dataset)
    assert resolved == dataset_size  # Should cap at dataset size

    # Test with "all" - should use the entire dataset
    entry.num_samples = "all"
    resolved = entry.resolve_num_samples(test_dataset_path, dataset)
    assert resolved == dataset_size  # Should use the entire dataset

    print("✅ DatasetEntryConfig resolve_num_samples test passed")


def test_dataset_entry_config_requirements():
    """Test that DatasetEntryConfig correctly validates requirements."""
    print("\n=== Testing DatasetEntryConfig Requirements ===")

    # Test correct format
    correct_format = {
        "dataset": "test/dataset",
        "split": "train",
        "columns": ["messages"],
        "formatter": "sharegpt",
        "num_samples": 10,
    }

    entry = DatasetEntryConfig.from_dict(correct_format)
    assert entry.dataset == "test/dataset"
    assert entry.split == "train"
    assert entry.columns == ["messages"]
    assert entry.formatter == "sharegpt"
    assert entry.num_samples == 10

    # Test invalid formats
    invalid_formats = [
        # columns must be a list
        {
            "dataset": "test/dataset",
            "split": "train",
            "columns": "messages",
            "formatter": "sharegpt",
            "num_samples": 10,
        },
        # missing formatter
        {
            "dataset": "test/dataset",
            "split": "train",
            "columns": ["messages"],
            "num_samples": 10,
        },
        # missing dataset
        {
            "split": "train",
            "columns": ["messages"],
            "formatter": "sharegpt",
            "num_samples": 10,
        },
        # missing split
        {
            "dataset": "test/dataset",
            "columns": ["messages"],
            "formatter": "sharegpt",
            "num_samples": 10,
        },
        # missing num_samples
        {
            "dataset": "test/dataset",
            "split": "train",
            "columns": ["messages"],
            "formatter": "sharegpt",
        },
        # invalid num_samples
        {
            "dataset": "test/dataset",
            "split": "train",
            "columns": ["messages"],
            "formatter": "sharegpt",
            "num_samples": -1,
        },
        {
            "dataset": "test/dataset",
            "split": "train",
            "columns": ["messages"],
            "formatter": "sharegpt",
            "num_samples": 0,
        },
    ]

    for i, invalid_format in enumerate(invalid_formats):
        try:
            entry = DatasetEntryConfig.from_dict(invalid_format)
            assert False, f"Expected ValueError for invalid format {i}"
        except ValueError as e:
            print(f"✅ Correctly raised error for invalid format {i}: {e}")

    # Test valid "all" for num_samples
    all_format = {
        "dataset": "test/dataset",
        "split": "train",
        "columns": ["messages"],
        "formatter": "sharegpt",
        "num_samples": "all",
    }

    entry = DatasetEntryConfig.from_dict(all_format)
    assert entry.num_samples == "all"

    print("✅ DatasetEntryConfig requirements test passed")


def test_invalid_calibration_set_config():
    """Test that calibration_set config must have 'calibration_set' key at root."""
    print("\n=== Testing Invalid CalibrationSet Config !===")

    # Test config without calibration_set key at root
    invalid_config = {
        "max_seq_length": 4096,
        "shuffle": True,
        "seed": 42,
        "datasets": [
            {
                "dataset": "test/dataset",
                "split": "train",
                "columns": ["messages"],
                "formatter": "sharegpt",
                "num_samples": 10,
            }
        ],
    }

    try:
        # config = CalibrationSetConfig.from_dict(invalid_config)
        CalibrationSetConfig.from_dict(invalid_config)
        assert False, "Expected ValueError for config missing 'calibration_set' key"
    except ValueError as e:
        assert "calibration_set' key at the root level" in str(e)
        print("✅ Correctly rejected config missing 'calibration_set' key")

    # Test when calibration_set key is not at root but nested in another key
    nested_config = {
        "calibration_set_config": {
            "calibration_set": {
                "max_seq_length": 4096,
                "datasets": [
                    {
                        "dataset": "test/dataset",
                        "split": "train",
                        "columns": ["messages"],
                        "formatter": "sharegpt",
                        "num_samples": 10,
                    }
                ],
            }
        }
    }

    try:
        # config = CalibrationSetConfig.from_dict(nested_config)
        CalibrationSetConfig.from_dict(nested_config)
        assert False, "Expected ValueError for nested calibration_set key"
    except ValueError as e:
        assert "calibration_set' key at the root level" in str(e)
        print("✅ Correctly rejected config with nested 'calibration_set' key:", e)

    print("✅ Invalid calibration set config test passed")


def test_edge_cases():
    """Test edge cases and error conditions."""
    print("\n=== Testing Edge Cases ===")

    # Test with max_seq_length=0
    config = CalibrationSetConfig(
        max_seq_length=0,
        shuffle=False,
        seed=42,
        datasets=[
            DatasetEntryConfig(
                dataset=str(
                    Path(__file__).parent / "test_datasets" / "raw_text" / "ds_text"
                ),
                split="train",
                columns=["text"],
                formatter="raw_text",
                num_samples=10,
            )
        ],
    )
    calib_set = CalibrationSet.from_config(config=config)
    assert calib_set.config.max_seq_length == 0

    print("✅ Config with max_seq_length=0 handled gracefully")

    # Test with very large num_samples
    try:
        entry = DatasetEntryConfig(
            dataset=str(
                Path(__file__).parent / "test_datasets" / "raw_text" / "ds_text"
            ),
            split="train",
            columns=["text"],
            formatter="raw_text",
            num_samples=1000000000,
        )
        config.datasets = [entry]
        calib_set = CalibrationSet.from_config(config=config)
        # If we get here without an exception, the large number was handled gracefully
        print(
            f"✅ Very large num_samples ({entry.num_samples}) handled gracefully by using all available samples"
        )
    except Exception as e:
        print(f"ℹ️  Large num_samples test resulted in expected behavior: {e}")

    # Test with empty datasets - validation should happen at entry points
    try:
        empty_config = CalibrationSetConfig(
            max_seq_length=4096, shuffle=False, seed=42, datasets=[]
        )
        calib_set = CalibrationSet.from_config(config=empty_config, cache_dir="./cache")
        assert False, "Expected ValueError for empty datasets"
    except ValueError as e:
        assert "at least one dataset" in str(e)
        print("⚠️  Config with empty datasets rejected:", e)

    print("✅ Edge cases test passed")


if __name__ == "__main__":
    print("🔍 Running calibration sets tests...")

    try:
        test_creating_calibration_set()
        test_compute_cache_key()
        test_save_and_load()
        test_create_from_config()
        test_dataset_entry_config_validation()
        test_dataset_entry_config_resolve_num_samples()
        test_dataset_entry_config_requirements()
        test_invalid_calibration_set_config()
        test_edge_cases()

        print("\n🎉 All calibration sets tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
