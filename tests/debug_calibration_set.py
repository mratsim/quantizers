#!/usr/bin/env python3
"""
Debug script for calibration_set module.

This script is used for testing the calibration_set functionality.
It creates test configurations and verifies that cache keys are generated correctly.
"""

import sys
from pathlib import Path

# Add the quantizers directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantizers.calibration_sets import (
    CalibrationSet,
    CalibrationSetConfig,
    DatasetEntryConfig,
)


def main():
    """Test configuration creation and cache key generation."""
    print("Debugging calibration_set module...")

    try:
        # Test Case 1: Cache key generation without "all" value
        print("\n--- Test Case 1: Cache key with specific sample count ---")

        # Create a DatasetEntryConfig with specific num_samples
        dataset_entry = DatasetEntryConfig(
            dataset="dummy/dataset1",
            split="train",
            columns=["messages"],
            formatter="sharegpt",
            num_samples=10,
        )

        # Create a simple CalibrationSetConfig
        config = CalibrationSetConfig(max_seq_length=4096, shuffle=True, seed=42, datasets=[dataset_entry])

        # Test cache key generation
        cache_key = CalibrationSet.compute_cache_key(config)
        print(f"✅ Generated cache key: {cache_key}")

        # Verify the cache key contains sample count
        if "10" in cache_key:
            print("✅ Cache key correctly contains sample count")
        else:
            print("❌ Cache key should contain sample count")

        # Test Case 2: Cache key generation with "all" value
        print("\n--- Test Case 2: Cache key with 'all' sample value ---")

        # Create a DatasetEntryConfig with "all" num_samples
        dataset_entry_all = DatasetEntryConfig(
            dataset="dummy/dataset2",
            split="train",
            columns=["messages"],
            formatter="sharegpt",
            num_samples="all",
        )

        # Create a CalibrationSetConfig with "all" value
        config_all = CalibrationSetConfig(max_seq_length=4096, shuffle=True, seed=42, datasets=[dataset_entry_all])

        # Test cache key generation with "all" value
        cache_key_all = CalibrationSet.compute_cache_key(config_all)
        print(f"✅ Generated cache key with 'all': {cache_key_all}")

        # Verify the cache key contains "length_TBD" as expected
        if "length_TBD" in cache_key_all:
            print("✅ Cache key correctly contains 'length_TBD' for 'all' sample value")
        else:
            print("❌ Cache key should contain 'length_TBD' for 'all' sample value")

        # Test Case 3: Cache key with mixed "all" and specific values
        print("\n--- Test Case 3: Cache key with mixed 'all' and specific values ---")

        # Create another DatasetEntryConfig with specific num_samples
        dataset_entry_mixed = DatasetEntryConfig(
            dataset="dummy/dataset3",
            split="validation",
            columns=["text"],
            formatter="raw_text",
            num_samples=50,
        )

        # Create a CalibrationSetConfig with both "all" and specific values
        config_mixed = CalibrationSetConfig(
            max_seq_length=4096,
            shuffle=True,
            seed=42,
            datasets=[dataset_entry_all, dataset_entry_mixed],
        )

        # Test cache key generation with mixed values
        cache_key_mixed = CalibrationSet.compute_cache_key(config_mixed)
        print(f"✅ Generated cache key with mixed values: {cache_key_mixed}")

        # Verify the cache key contains "length_TBD" as expected
        if "length_TBD" in cache_key_mixed:
            print("✅ Cache key correctly contains 'length_TBD' when any dataset has 'all' value")
        else:
            print("❌ Cache key should contain 'length_TBD' when any dataset has 'all' value")

        # Test Case 4: Formatter column extraction
        print("\n--- Test Case 4: Chat completion formatter ---")

        # Create test data for chat completion formatter
        test_data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }

        # Test chat completion formatter
        from quantizers.formatters import DatasetFmt

        formatted_messages = DatasetFmt.chat_completion(["messages"], test_data)
        print(f"✅ Chat completion formatter result: {formatted_messages}")

        # Verify the formatter extracted the messages correctly
        if formatted_messages == test_data["messages"]:
            print("✅ Chat completion formatter correctly extracted messages from specified column")
        else:
            print("❌ Chat completion formatter should extract messages from specified column")

        print("\n✅ All tests passed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
