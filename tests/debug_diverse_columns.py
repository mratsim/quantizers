#!/usr/bin/env python3
"""Debug script for testing diverse column configurations and formatters."""

# ruff: noqa: E402
import json
import sys
from pathlib import Path

# Add the source directory to Python path
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

# Import the modules
from datasets import load_dataset

from quantizers.calibration_sets import (
    CalibrationSetConfig,
)
from quantizers.formatters import DatasetFmt


def debug_chat_completion_formatter():
    """Debug test for chat completion formatter to verify our fix works correctly."""
    print("\n=== Debugging Chat Completion Formatter ===")

    # Load the YAML config
    calib_config = CalibrationSetConfig.from_file("tests/test_datasets/t_calibrate_diverse_columns.yaml")

    # Find the chat completion dataset
    chat_ds = None
    for ds in calib_config.datasets:
        if ds.formatter == "chat_completion":
            chat_ds = ds
            break

    assert chat_ds is not None, "Chat completion dataset not found"
    print(f"Found chat completion dataset: {chat_ds.dataset}, columns: {chat_ds.columns}")

    # Load the dataset directly and examine its structure
    dataset = load_dataset(chat_ds.dataset, split="train")

    # Look at first few examples
    print("\n=== Raw Data Structure ===")
    for i, row in enumerate(dataset):
        if i >= 3:  # Only look at first 3 examples
            break

        print(f"\n--- Row {i} ---")
        for key, value in row.items():
            if key in chat_ds.columns:
                print(f"Column '{key}': {type(value)} - {value}")
            else:
                print(f"Other column '{key}': {type(value)}")

        # Try applying the fixed formatter
        try:
            formatter_func = DatasetFmt.get_formatter(chat_ds.formatter)
            formatted = formatter_func(chat_ds.columns, row)
            print(f"\nFormatted result: {type(formatted)} - {formatted}")

            # Verify it's a list (this was the original issue)
            assert isinstance(formatted, list), f"Expected list, got {type(formatted)}"
            print("✅ Formatted result is a list as expected")

            # Verify each item in the list is a message dict
            for msg in formatted:
                assert isinstance(msg, dict), f"Expected dict, got {type(msg)}"
                assert "role" in msg, "Message missing 'role' field"
                assert "content" in msg, "Message missing 'content' field"
            print("✅ All messages have correct structure")
        except Exception as e:
            print(f"\nFormatter error: {str(e)}")
            raise

    return chat_ds


def verify_diverse_column_formats():
    """Manually verify each dataset format individually."""
    print("\n=== Verifying Individual Diverse Column Formats ===")

    # Test each dataset individually to verify column formats
    configs = [
        # Note: Skipping ds_musings as it has extra metadata fields that cause
        # schema alignment issues when combined with other datasets
        (
            "tests/test_datasets/sharegpt/ds_conversations/dataset.json",
            ["conversations"],
        ),
        (
            "tests/test_datasets/prompt_answer/ds_instruction_response/dataset.json",
            ["instruction", "response"],
        ),
        ("tests/test_datasets/raw_text/ds_text/dataset.json", ["text"]),
        ("tests/test_datasets/chat_completion/ds_messages/dataset.json", ["messages"]),
    ]

    for dataset_path, columns in configs:
        print(f"\n--- Testing {Path(dataset_path).name} with columns {columns} ---")

        try:
            # Load dataset directly
            with open(dataset_path, "r") as f:
                data = json.load(f)

            first_row = data[0]
            print(f"Columns in dataset: {list(first_row.keys())}")

            # Test each formatter
            for formatter_name in [
                "sharegpt",
                "prompt_answer",
                "raw_text",
                "chat_completion",
            ]:
                try:
                    formatter_func = DatasetFmt.get_formatter(formatter_name)
                    formatted = formatter_func(columns, first_row)
                    print(f"  - {formatter_name}: {type(formatted)} - {str(formatted)[:50]}...")
                except ValueError as e:
                    if formatter_name != columns[0].replace("s", ""):  # rough heuristic
                        print(f"  - {formatter_name}: Expected error - {str(e)}")
                except Exception as e:
                    print(f"  - {formatter_name}: Unexpected error - {str(e)}")

        except Exception as e:
            print(f"  Error loading dataset: {str(e)}")


def main():
    """Main debug function."""
    print("🔧 Debugging Diverse Columns Formatters")
    print("=" * 50)

    # Debug individual formatters
    debug_chat_completion_formatter()

    # Verify all column formats
    verify_diverse_column_formats()

    print("\n🎉 Debugging complete!")


if __name__ == "__main__":
    main()
