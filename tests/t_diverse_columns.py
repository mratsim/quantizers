#!/usr/bin/env python3

"""
Tests for loading and using calibration sets with diverse column configurations.

This test file validates the ability to use the t_calibrate_diverse_columns.yaml
configuration file which tests our system with datasets that have different column names
and structures. This ensures our calibration system can handle real-world diversity
in dataset formats.

To run these tests:
    uv run python tests/t_diverse_columns.py
"""

import sys
from pathlib import Path

import datasets
import pytest

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
            else:
                text = messages.get("content", "")

        if tokenize:
            return {"input_ids": self.input_ids, "attention_mask": self.attention_mask}
        return text

    def __call__(
        self,
        text,
        padding=False,
        max_length=None,
        truncation=False,
        add_special_tokens=False,
    ):
        """Mock tokenizer __call__ method."""
        return {"input_ids": self.input_ids, "attention_mask": self.attention_mask}


def test_load_diverse_columns_calibration_set():
    """Test loading a calibration set with diverse column structures from YAML."""
    cache_dir = str(Path(__file__).parent / "test_cache")

    # Clear any existing cache
    import shutil

    if Path(cache_dir).exists():
        shutil.rmtree(cache_dir)

    print("\n=== Testing Loading Diverse Columns Calibration Set ===")

    # Create a simpler config with compatible datasets that won't cause schema alignment issues
    calibration_set_config = CalibrationSetConfig(
        max_seq_length=8192,
        shuffle=True,
        seed=42,
        datasets=[
            DatasetEntryConfig(
                dataset="tests/test_datasets/sharegpt/ds_conversations",
                split="train",
                columns=["conversations"],
                formatter="sharegpt",
                num_samples=3,
            ),
            DatasetEntryConfig(
                dataset="tests/test_datasets/prompt_answer/ds_instruction_response",
                split="train",
                columns=["instruction", "response"],
                formatter="prompt_answer",
                num_samples=3,
            ),
            DatasetEntryConfig(
                dataset="tests/test_datasets/raw_text/ds_text",
                split="train",
                columns=["text"],
                formatter="raw_text",
                num_samples=3,
            ),
        ],
    )

    # Test 1: Check if cache exists (should not initially)
    assert not CalibrationSet.is_cached(calibration_set_config, cache_dir)
    print("✅ Cache does not exist initially")

    # Test 2: Build from config (this should work)
    calib_set = CalibrationSet.from_config(calibration_set_config, cache_dir)

    # Test 3: Verify data was loaded
    assert calib_set._untokenized_calibration_set is not None
    assert calib_set.total_num_samples > 0

    print(f"✅ Successfully loaded calibration set with {calib_set.total_num_samples} samples")

    # Test 4: Save to cache
    calib_set.save_to_cache()
    print("✅ Successfully saved to cache")

    # Test 5: Now cache should exist
    assert CalibrationSet.is_cached(calibration_set_config, cache_dir)
    print("✅ Cache now exists")

    # Test 6: Load from cache
    calib_set_cached = CalibrationSet.from_cache(calibration_set_config, cache_dir)

    # Verify cached data
    assert calib_set_cached._untokenized_calibration_set is not None
    assert calib_set_cached.total_num_samples == calib_set.total_num_samples
    print("✅ Successfully loaded from cache")

    # Test 7: Verify each dataset was processed correctly
    # Each should have 3 samples from our config
    assert calib_set_cached.total_num_samples == 9  # 3 datasets × 3 samples each
    print("✅ All datasets loaded with correct sample count (9 total)")

    # Test 8: Tokenize the data
    tokenizer = MockTokenizer()
    tokenized_data = calib_set_cached.get_tokenized(tokenizer)

    # Verify tokenization worked
    assert isinstance(tokenized_data, datasets.Dataset)
    assert len(tokenized_data) == 9
    print("✅ Successfully tokenized calibration data")

    # Test 9: Verify the cache key is deterministic
    cache_key = CalibrationSet.compute_cache_key(calibration_set_config)
    cache_key2 = CalibrationSet.compute_cache_key(calibration_set_config)
    assert cache_key == cache_key2, f"Cache keys should be identical: {cache_key} != {cache_key2}"
    print("✅ Cache key generation is deterministic")

    # Print the cache key for debugging
    print(f"   Cache key: {cache_key}")

    # Test 10: Verify the cache file exists
    cache_file = Path(cache_dir) / cache_key
    assert cache_file.exists(), f"Cache file should exist at {cache_file}"
    print("✅ Cache file created correctly")

    print("\n🎉 All diverse columns calibration set tests passed!")


def test_yaml_direct_loading():
    """Test that we can also load the YAML configuration directly using our config system."""
    pytest.importorskip("quantizers.config")

    from quantizers.calibration_sets import CalibrationSetConfig

    print("\n=== Testing Direct YAML Loading ===")

    # Try to load the YAML directly
    calib_config = CalibrationSetConfig.from_file("tests/test_datasets/t_calibrate_diverse_columns.yaml")

    # Verify we got a config object
    assert hasattr(calib_config, "datasets")
    assert len(calib_config.datasets) > 0

    print("✅ Successfully loaded configuration directly from YAML")

    # Verify the config structure
    assert calib_config.max_seq_length == 8192
    assert calib_config.shuffle is True
    assert calib_config.seed == 42
    assert len(calib_config.datasets) == 9

    print(f"✅ Configuration has {len(calib_config.datasets)} datasets as expected")


def test_problematic_mixed_dataset():
    """Test for the problematic mixed dataset that was causing schema alignment issues.

    This test verifies that after fixing the chat_completion formatter,
    we can successfully concatenate datasets with different schemas.
    """
    pytest.importorskip("quantizers.config")

    from quantizers.calibration_sets import CalibrationSet, CalibrationSetConfig

    print("\n=== Testing Problematic Mixed Dataset ===")

    # Load the YAML config with diverse datasets
    calib_config = CalibrationSetConfig.from_file("tests/test_datasets/t_calibrate_diverse_columns.yaml")

    # Verify all datasets are loaded
    assert len(calib_config.datasets) == 9
    print("✅ Configuration loaded with 9 diverse datasets")

    # Try to create calibration set - this should now pass with our fix
    try:
        calib_set = CalibrationSet.from_config(calib_config, "test_cache")
        print("✅ Successfully created calibration set with mixed datasets")

        # Verify the set was created correctly
        assert (
            calib_set.total_num_samples == 43
        )  # 6 datasets × 3 samples each + 2 datasets × 10 samples each + 1 dataset with 5 samples = 18 + 20 + 5 = 43
        print("✅ Calibration set has correct number of samples")

        # Verify that all formatted entries are lists (not dicts)
        raw_dataset = calib_set._untokenized_calibration_set
        for i, row in enumerate(raw_dataset):
            formatted = row["formatted"]
            assert isinstance(formatted, list), f"Row {i}: Expected list, got {type(formatted)}"
            assert len(formatted) > 0, f"Row {i}: Formatted list is empty"

            # Check each message structure
            for msg in formatted:
                assert isinstance(msg, dict), f"Row {i}: Expected dict, got {type(msg)}"
                assert "role" in msg, f"Row {i}: Message missing 'role' field"
                assert "content" in msg, f"Row {i}: Message missing 'content' field"

        print("✅ All formatted entries are properly structured message lists")

    except ValueError as e:
        if "features can't be aligned" in str(e):
            print(f"❌ Still getting schema alignment error: {str(e)}")
            print("This suggests the formatter fix didn't solve the problem")
            raise


def test_calibration_set_with_arbitrary_columns():
    """Test that CalibrationSet works correctly with arbitrary column names."""
    print("\n=== Testing CalibrationSet with Arbitrary Columns ===")

    # Test configuration with arbitrary column names
    try:
        config = CalibrationSetConfig(
            max_seq_length=4096,
            shuffle=False,
            seed=42,
            datasets=[
                DatasetEntryConfig(
                    dataset=str(Path(__file__).parent / "test_datasets" / "prompt_answer" / "ds_input_output"),
                    split="train",
                    columns=["input", "output"],  # Existing column names in dataset
                    formatter="prompt_answer",
                    num_samples=2,
                )
            ],
        )

        # Create CalibrationSet with the config
        calib_set = CalibrationSet.from_config(config)

        # Get untokenized data to verify it works
        assert calib_set._untokenized_calibration_set is not None, "CalibrationSet should have untokenized data"
        assert len(calib_set._untokenized_calibration_set) > 0, "CalibrationSet should have data samples"
        print("\n✅ CalibrationSet correctly processes data with arbitrary column names")
    except Exception as e:
        pytest.fail(f"CalibrationSet test with arbitrary columns failed: {e}")


def main():
    """Run all tests."""
    print("Running calibration set diverse columns tests...")

    try:
        test_load_diverse_columns_calibration_set()
        test_yaml_direct_loading()
        test_problematic_mixed_dataset()
        test_calibration_set_with_arbitrary_columns()
        test_toolace_diverse_columns()

        print("\n🎉 All tests passed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")


def test_toolace_diverse_columns():
    """Test loading ToolACE dataset with diverse column configurations."""
    print("\n=== Testing ToolACE Dataset with Diverse Columns ===")

    # Test configuration with ToolACE dataset and the new formatter
    try:
        config = CalibrationSetConfig(
            max_seq_length=4096,
            shuffle=False,
            seed=42,
            datasets=[
                DatasetEntryConfig(
                    dataset=str(Path(__file__).parent / "test_datasets" / "toolace_sample"),
                    split="train",
                    columns=["system", "conversations"],  # Specific column names for ToolACE
                    formatter="chat_completion_with_sysprompt",
                    num_samples=3,
                )
            ],
        )

        # Create CalibrationSet with the config
        calib_set = CalibrationSet.from_config(config)

        # Get untokenized data to verify it works
        assert calib_set._untokenized_calibration_set is not None, "CalibrationSet should have untokenized data"
        assert len(calib_set._untokenized_calibration_set) > 0, "CalibrationSet should have data samples"

        # Check the first sample to ensure it has been properly formatted
        first_sample = calib_set._untokenized_calibration_set[0]
        formatted = first_sample["formatted"]

        # Verify the formatter returns the correct structure
        assert isinstance(formatted, list), f"Expected list, got {type(formatted)}"
        assert len(formatted) > 0, "Formatted list is empty"

        # Check each message structure
        for msg in formatted:
            assert isinstance(msg, dict), f"Expected dict, got {type(msg)}"
            assert "role" in msg, "Message missing 'role' field"
            assert "content" in msg, "Message missing 'content' field"

            # Validate roles are valid
            assert msg["role"] in ["system", "user", "assistant"], f"Invalid role: {msg['role']}"

        # Verify at least one system message is present
        system_messages = [msg for msg in formatted if msg["role"] == "system"]
        assert len(system_messages) > 0, "Should have at least one system message"

        print("\n✅ ToolACE dataset correctly loaded with system prompts and conversations")

    except Exception as e:
        pytest.fail(f"ToolACE dataset test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("Running calibration set diverse columns tests...")
    try:
        test_load_diverse_columns_calibration_set()
        test_yaml_direct_loading()
        test_problematic_mixed_dataset()
        test_calibration_set_with_arbitrary_columns()
        test_toolace_diverse_columns()
        print("\n🎉 All tests passed successfully!")
    except Exception as e:
        import traceback

        print(f"\n❌ Test failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)
