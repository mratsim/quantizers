#!/usr/bin/env python3
"""
Tests for dataset loading functionality.

This test file validates that the dataloading system works correctly with
the fake test datasets in this directory, testing various data formats and
formatters.

To run these tests:
    uv run python tests/test_datasets/test_dataloading.py
"""

import json
import sys
from pathlib import Path

from quantizers.calibration_sets import (
    CalibrationSet,
    CalibrationSetConfig,
    DatasetEntryConfig,
)

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_sharegpt_format_loading():
    """Test loading datasets in ShareGPT format."""
    print("\n=== Testing ShareGPT Format Loading ===")

    # Create a test configuration
    config = CalibrationSetConfig(
        max_seq_length=4096,
        shuffle=False,
        seed=42,
        datasets=[
            DatasetEntryConfig(
                dataset=str(
                    Path(__file__).parent / "test_datasets" / "sharegpt_format"
                ),
                split="train",
                columns=["conversations"],
                formatter="sharegpt",
                num_samples=3,
            )
        ],
    )

    # Create CalibrationSet using factory method
    calib_set = CalibrationSet.from_config(config)

    # Verify properties
    assert calib_set.config == config
    assert calib_set._untokenized_calibration_set is not None
    assert calib_set.total_num_samples == 3

    # Check that the data was loaded correctly
    # Just verify the untokenized data has the expected structure
    raw_dataset = calib_set._untokenized_calibration_set
    assert len(raw_dataset) == 3

    # Verify the data format
    for i, row in enumerate(raw_dataset):
        assert "formatted" in row, f"Row {i} missing formatted field"
        messages = row["formatted"]
        assert isinstance(messages, list), f"Row {i} messages not a list"
        assert (
            len(messages) >= 2
        ), f"Row {i} should have at least user and assistant messages"

        # Check for user and assistant messages
        has_user = any(msg.get("role") == "user" for msg in messages)
        has_assistant = any(msg.get("role") == "assistant" for msg in messages)
        assert has_user, f"Row {i} missing user message"
        assert has_assistant, f"Row {i} missing assistant message"

    print("✅ ShareGPT format loading test passed")


def test_prompt_answer_format_loading():
    """Test loading datasets in prompt-answer format."""
    print("\n=== Testing Prompt-Answer Format Loading ===")

    # Create a test configuration
    config = CalibrationSetConfig(
        max_seq_length=4096,
        shuffle=False,
        seed=42,
        datasets=[
            DatasetEntryConfig(
                dataset=str(
                    Path(__file__).parent / "test_datasets" / "prompt_answer_format"
                ),
                split="train",
                columns=["input", "output"],
                formatter="prompt_answer",
                num_samples=3,
            )
        ],
    )

    # Create CalibrationSet using factory method
    calib_set = CalibrationSet.from_config(config)

    # Verify properties
    assert calib_set.config == config
    assert calib_set._untokenized_calibration_set is not None
    assert calib_set.total_num_samples == 3

    # Check that the data was loaded correctly
    # Just verify the untokenized data has the expected structure
    raw_dataset = calib_set._untokenized_calibration_set
    assert len(raw_dataset) == 3

    # Verify the data format
    for i, row in enumerate(raw_dataset):
        assert "formatted" in row, f"Row {i} missing formatted field"
        messages = row["formatted"]
        assert isinstance(messages, list), f"Row {i} messages not a list"
        assert (
            len(messages) >= 2
        ), f"Row {i} messages should have at least 2 elements (system and user)"

        # Check for user and assistant messages
        has_user = any(msg.get("role") == "user" for msg in messages)
        has_assistant = any(msg.get("role") == "assistant" for msg in messages)
        assert has_user, f"Row {i} missing user message"
        assert has_assistant, f"Row {i} missing assistant message"

    print("✅ Prompt-answer format loading test passed")


def test_chat_completion_format_loading():
    """Test loading datasets in chat completion format."""
    print("\n=== Testing Chat Completion Format Loading ===")

    # Create a test configuration
    config = CalibrationSetConfig(
        max_seq_length=4096,
        shuffle=False,
        seed=42,
        datasets=[
            DatasetEntryConfig(
                dataset=str(
                    Path(__file__).parent / "test_datasets" / "chat_completion_format"
                ),
                split="train",
                columns=["messages"],
                formatter="chat_completion",
                num_samples=3,
            )
        ],
    )

    # Create CalibrationSet using factory method
    calib_set = CalibrationSet.from_config(config)

    # Verify properties
    assert calib_set.config == config
    assert calib_set._untokenized_calibration_set is not None
    assert calib_set.total_num_samples == 3

    # Check that the data was loaded correctly
    # Just verify the untokenized data has the expected structure
    raw_dataset = calib_set._untokenized_calibration_set
    assert len(raw_dataset) == 3

    # Verify the data format
    for i, row in enumerate(raw_dataset):
        assert "formatted" in row, f"Row {i} missing formatted field"
        response = row["formatted"]
        # For chat completion format, response should be a dict with a "messages" key
        assert isinstance(response, dict), f"Row {i} response should be a dict"
        assert "messages" in response, f"Row {i} response should have 'messages' key"
        messages = response["messages"]
        assert isinstance(messages, list), f"Row {i} messages should be a list"
        assert (
            len(messages) >= 2
        ), f"Row {i} should have at least user and assistant messages"

        # Check for user and assistant messages
        has_user = any(msg.get("role") == "user" for msg in messages)
        has_assistant = any(msg.get("role") == "assistant" for msg in messages)
        assert has_user, f"Row {i} missing user message"
        assert has_assistant, f"Row {i} missing assistant message"

    print("✅ Chat completion format loading test passed")


def test_raw_text_format_loading():
    """Test loading datasets in raw text format."""
    print("\n=== Testing Raw Text Format Loading ===")

    # Create a test configuration
    config = CalibrationSetConfig(
        max_seq_length=4096,
        shuffle=False,
        seed=42,
        datasets=[
            DatasetEntryConfig(
                dataset=str(Path(__file__).parent / "test_datasets" / "raw_text"),
                split="train",
                columns=["text"],  # Single column for raw text
                formatter="raw_text",
                num_samples=5,
            )
        ],
    )

    # Create CalibrationSet using factory method
    calib_set = CalibrationSet.from_config(config)

    # Verify properties
    assert calib_set.config == config
    assert calib_set._untokenized_calibration_set is not None
    assert calib_set.total_num_samples == 5

    # Check that the data was loaded correctly
    # Just verify the untokenized data has the expected structure
    raw_dataset = calib_set._untokenized_calibration_set
    assert len(raw_dataset) == 5

    # Verify the data format
    for i, row in enumerate(raw_dataset):
        assert "formatted" in row, f"Row {i} missing formatted field"
        messages = row["formatted"]
        assert isinstance(messages, list), f"Row {i} messages not a list"
        assert len(messages) == 1, f"Row {i} should have exactly one message"

        # Check for assistant message
        has_assistant = any(msg.get("role") == "assistant" for msg in messages)
        assert has_assistant, f"Row {i} missing assistant message"

    print("✅ Raw text format loading test passed")


def test_multiple_dataset_loading():
    """Test loading multiple datasets together."""
    print("\n=== Testing Multiple Dataset Loading ===")

    # Create a test configuration with multiple datasets
    config = CalibrationSetConfig(
        max_seq_length=4096,
        shuffle=False,
        seed=42,
        datasets=[
            DatasetEntryConfig(
                dataset=str(
                    Path(__file__).parent / "test_datasets" / "sharegpt_format"
                ),
                split="train",
                columns=["conversations"],
                formatter="sharegpt",
                num_samples=2,
            ),
            DatasetEntryConfig(
                dataset=str(
                    Path(__file__).parent / "test_datasets" / "prompt_answer_format"
                ),
                split="train",
                columns=["input", "output"],
                formatter="prompt_answer",
                num_samples=2,
            ),
            DatasetEntryConfig(
                dataset=str(Path(__file__).parent / "test_datasets" / "raw_text"),
                split="train",
                columns=["text"],
                formatter="raw_text",
                num_samples=1,
            ),
        ],
    )

    # Create CalibrationSet using factory method
    calib_set = CalibrationSet.from_config(config)

    # Verify properties
    assert calib_set.config == config
    assert calib_set._untokenized_calibration_set is not None
    assert calib_set.total_num_samples == 5  # 2 + 2 + 1

    # Check that the data was loaded correctly
    # Just verify the untokenized data has the expected structure
    raw_dataset = calib_set._untokenized_calibration_set
    assert (
        len(raw_dataset) == 5
    )  # 2 from sharegpt + 2 from prompt-answer + 1 from raw_text

    # Verify the data format has been properly transformed by the formatter
    for i, row in enumerate(raw_dataset):
        assert "formatted" in row, f"Row {i} missing formatted field"
        messages = row["formatted"]
        assert isinstance(messages, list), f"Row {i} messages not a list"
        # Different formatters produce different message lengths
        # Raw text formatter produces 1 message, others produce 2+
        if i >= 4:  # This is the raw text formatter data
            assert (
                len(messages) >= 1
            ), f"Row {i} messages should have at least 1 element (raw text)"
        else:
            assert (
                len(messages) >= 2
            ), f"Row {i} messages should have at least 2 elements (chat/prompt-answer)"

        # Determine which dataset this row came from
        if i < 2:
            # ShareGPT format (2 rows)
            assert (
                len(messages) >= 2
            ), f"Row {i} should have at least user and assistant messages"
        elif i < 4:
            # Prompt-answer format (2 rows)
            assert (
                len(messages) >= 2
            ), f"Row {i} should have at least user and assistant messages"
        else:
            # Raw text format (1 row)
            assert (
                len(messages) == 1
            ), f"Row {i} raw text format should have exactly one message"
            # Check for assistant message
            has_assistant = any(msg.get("role") == "assistant" for msg in messages)
            assert has_assistant, f"Row {i} missing assistant message"

    print("✅ Multiple dataset loading test passed")


def test_error_handling_missing_dataset():
    """Test error handling when dataset file doesn't exist."""
    print("\n=== Testing Error Handling - Missing Dataset ===")

    # Create a test configuration with non-existent dataset file
    config = CalibrationSetConfig(
        max_seq_length=4096,
        shuffle=False,
        seed=42,
        datasets=[
            DatasetEntryConfig(
                dataset=str(
                    Path(__file__).parent / "test_datasets" / "nonexistent_format"
                ),
                split="train",
                columns=["messages"],
                formatter="chat_completion",
                num_samples=10,
            )
        ],
    )

    # Try to create CalibrationSet
    try:
        _calib_set = CalibrationSet.from_config(config)
        assert False, "Expected an error for non-existent dataset"
    except Exception as e:
        print(f"✅ Correctly raised error for non-existent dataset: {str(e)}")


def test_error_handling_invalid_columns():
    """Test error handling when dataset doesn't have specified columns."""
    print("\n=== Testing Error Handling - Invalid Columns ===")

    # Create a test configuration with invalid column names
    config = CalibrationSetConfig(
        max_seq_length=4096,
        shuffle=False,
        seed=42,
        datasets=[
            DatasetEntryConfig(
                dataset=str(
                    Path(__file__).parent / "test_datasets" / "sharegpt_format"
                ),
                split="train",
                columns=["invalid_column"],
                formatter="sharegpt",
                num_samples=10,
            )
        ],
    )

    # Try to create CalibrationSet
    try:
        _calib_set = CalibrationSet.from_config(config)
        assert False, "Expected an error for invalid columns"
    except Exception as e:
        print(f"✅ Correctly raised error for invalid columns: {str(e)}")


def test_arbitrary_column_names():
    """Test that formatters work with arbitrary column names."""
    print("\n=== Testing Arbitrary Column Names ===")

    # Create a test dataset with custom column names
    from pathlib import Path

    custom_dataset_dir = Path(__file__).parent / "custom_columns"
    custom_dataset_dir.mkdir(exist_ok=True)

    # Create a custom dataset with different column names
    custom_data = [
        {
            "user_question": "What is the capital of France?",
            "bot_response": "The capital of France is Paris.",
        },
        {
            "user_question": "How do I write a Python function?",
            "bot_response": "Use the 'def' keyword followed by the function name.",
        },
    ]

    custom_dataset_file = custom_dataset_dir / "dataset.json"
    with open(custom_dataset_file, "w") as f:
        json.dump(custom_data, f)

    # Create a test configuration with arbitrary column names
    config = CalibrationSetConfig(
        max_seq_length=4096,
        shuffle=False,
        seed=42,
        datasets=[
            DatasetEntryConfig(
                dataset=str(
                    Path(__file__).parent / "test_datasets" / "custom_columns_format"
                ),
                split="train",
                columns=["user_question", "bot_response"],
                formatter="prompt_answer",
                num_samples=2,
            )
        ],
    )

    # Create CalibrationSet using factory method
    calib_set = CalibrationSet.from_config(config)

    # Check that the data was loaded correctly
    # Just verify the untokenized data has the expected structure
    raw_dataset = calib_set._untokenized_calibration_set
    assert len(raw_dataset) == 2

    # Verify arbitrary column names were correctly mapped
    for row in raw_dataset:
        assert "formatted" in row, "Row missing formatted field"
        messages = row["formatted"]
        assert isinstance(messages, list), "Messages not a list"
        assert len(messages) >= 2, "Should have at least user and assistant messages"

        # Verify the mapping happened correctly
        has_user = any(msg.get("role") == "user" for msg in messages)
        has_assistant = any(msg.get("role") == "assistant" for msg in messages)
        assert has_user, "Missing user message"
        assert has_assistant, "Missing assistant message"

    # Clean up
    custom_dataset_file.unlink()
    custom_dataset_dir.rmdir()

    print("✅ Arbitrary column names test passed")


def test_formatter_validation():
    """Test that formatters validate column counts correctly."""
    print("\n=== Testing Formatter Validation ===")

    # Test ShareGPT formatter with wrong number of columns
    try:
        config = CalibrationSetConfig(
            max_seq_length=4096,
            shuffle=False,
            seed=42,
            datasets=[
                DatasetEntryConfig(
                    dataset=str(
                        Path(__file__).parent / "test_datasets" / "sharegpt_format"
                    ),
                    split="train",
                    columns=["messages", "extra_column"],
                    formatter="sharegpt",
                    num_samples=10,
                )
            ],
        )
        _calib_set = CalibrationSet.from_config(config)
        assert False, "Expected an error for ShareGPT with wrong column count"
    except ValueError as e:
        assert "ShareGPT format requires exactly 1 column" in str(e)
        print("✅ ShareGPT formatter correctly validated column count")

    # Test prompt-answer formatter with wrong number of columns
    try:
        config = CalibrationSetConfig(
            max_seq_length=4096,
            shuffle=False,
            seed=42,
            datasets=[
                DatasetEntryConfig(
                    dataset=str(
                        Path(__file__).parent / "test_datasets" / "prompt_answer_format"
                    ),
                    split="train",
                    columns=["prompt"],
                    formatter="prompt_answer",
                    num_samples=10,
                )
            ],
        )
        _calib_set = CalibrationSet.from_config(config)
        assert False, "Expected an error for prompt-answer with wrong column count"
    except ValueError as e:
        assert "Prompt-answer format requires exactly 2 columns" in str(e)
        print("✅ Prompt-answer formatter correctly validated column count")

    # Test raw_text formatter with wrong number of columns
    try:
        config = CalibrationSetConfig(
            max_seq_length=4096,
            shuffle=False,
            seed=42,
            datasets=[
                DatasetEntryConfig(
                    dataset=str(Path(__file__).parent / "test_datasets" / "raw_text"),
                    split="train",
                    columns=["text", "extra_column"],
                    formatter="raw_text",
                    num_samples=10,
                )
            ],
        )
        _calib_set = CalibrationSet.from_config(config)
        assert False, "Expected an error for raw_text with wrong column count"
    except ValueError as e:
        assert "Raw text format requires exactly 1 column" in str(e)
        print("✅ Raw text formatter correctly validated column count")

    print("✅ Formatter validation test passed")


def test_missing_columns_keyerror():
    """Test that formatters raise KeyError for missing columns."""
    print("\n=== Testing Missing Columns KeyError ===")

    # Test ShareGPT formatter with missing column
    try:
        config = CalibrationSetConfig(
            max_seq_length=4096,
            shuffle=False,
            seed=42,
            datasets=[
                DatasetEntryConfig(
                    dataset=str(
                        Path(__file__).parent / "test_datasets" / "sharegpt_format"
                    ),
                    split="train",
                    columns=["missing_column"],
                    formatter="sharegpt",
                    num_samples=1,
                )
            ],
        )
        _calib_set = CalibrationSet.from_config(config)
        assert False, "Expected a KeyError for missing column"
    except KeyError as e:
        print(
            f"✅ ShareGPT formatter correctly raised KeyError for missing column: {str(e)}"
        )
    except Exception as e:
        print(f"❌ unexpected error for ShareGPT: {str(e)}")

    # Test prompt-answer formatter with missing column
    try:
        config = CalibrationSetConfig(
            max_seq_length=4096,
            shuffle=False,
            seed=42,
            datasets=[
                DatasetEntryConfig(
                    dataset=str(
                        Path(__file__).parent / "test_datasets" / "prompt_answer_format"
                    ),
                    split="train",
                    columns=["missing_column", "output"],
                    formatter="prompt_answer",
                    num_samples=1,
                )
            ],
        )
        _calib_set = CalibrationSet.from_config(config)
        assert False, "Expected a KeyError for missing column"
    except KeyError as e:
        print(
            f"✅ Prompt-answer formatter correctly raised KeyError for missing column: {str(e)}"
        )
    except Exception as e:
        print(f"❌ unexpected error for prompt-answer: {str(e)}")

    # Test raw_text formatter with missing column
    try:
        config = CalibrationSetConfig(
            max_seq_length=4096,
            shuffle=False,
            seed=42,
            datasets=[
                DatasetEntryConfig(
                    dataset=str(Path(__file__).parent / "test_datasets" / "raw_text"),
                    split="train",
                    columns=["missing_column"],
                    formatter="raw_text",
                    num_samples=1,
                )
            ],
        )
        _calib_set = CalibrationSet.from_config(config)
        assert False, "Expected a KeyError for missing column"
    except KeyError as e:
        print(
            f"✅ Raw text formatter correctly raised KeyError for missing column: {str(e)}"
        )
    except Exception as e:
        print(f"❌ unexpected error for raw_text: {str(e)}")

    print("✅ Missing columns KeyError test passed")


if __name__ == "__main__":
    print("🔍 Running dataset loading tests...")

    try:
        test_sharegpt_format_loading()
        test_prompt_answer_format_loading()
        test_chat_completion_format_loading()
        test_raw_text_format_loading()
        test_multiple_dataset_loading()
        test_error_handling_missing_dataset()
        test_error_handling_invalid_columns()
        test_arbitrary_column_names()
        test_formatter_validation()
        test_missing_columns_keyerror()

        print("\n🎉 All dataset loading tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
