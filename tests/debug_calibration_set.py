#!/usr/bin/env python3
"""
Debug script for calibration_set module.

This script is used for testing the calibration_set functionality.
It creates test configurations and verifies that cache keys are generated correctly.
"""

import sys
from pathlib import Path

from quantizers.calibration_sets import (
    CalibrationSet,
    CalibrationSetConfig,
    DatasetEntryConfig,
)

# Add the quantizers directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    """Test configuration creation and cache key generation."""
    print("Debugging calibration_set module...")

    try:
        # Create a simple DatasetEntryConfig
        dataset_entry = DatasetEntryConfig(
            dataset="test/dataset",
            split="train",
            columns=["messages"],
            formatter="sharegpt",
            num_samples=10,
        )
        print("✅ Created DatasetEntryConfig:", dataset_entry)

        # Create a simple CalibrationSetConfig
        config = CalibrationSetConfig(
            max_seq_length=4096, shuffle=True, seed=42, datasets=[dataset_entry]
        )
        print("✅ Created CalibrationSetConfig:", config)

        # Create CalibrationSet instance
        calib_set = CalibrationSet(config)
        print("✅ Created CalibrationSet instance")

        # Test cache key generation
        cache_key = calib_set.compute_cache_key()
        print(f"✅ Generated cache key: {cache_key}")

        # Verify config validation
        config.validate()
        print("✅ Config validation passed")

        # Test resolve_num_samples method
        class MockDataset:
            def __init__(self, length):
                self._length = length

            def __len__(self):
                return self._length

        mock_dataset = MockDataset(50)
        resolved_num = dataset_entry.resolve_num_samples("test/dataset", mock_dataset)
        print(f"✅ resolve_num_samples returned: {resolved_num}")

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
