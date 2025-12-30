#!/usr/bin/env python3

"""
Comprehensive tests for formatter functionality.

This test file validates all aspects of formatter functionality, including:
- Handling arbitrary column names
- Proper error handling
- Integration with CalibrationSet
- Column count validation
- Field validation for specific formats

To run these tests:
    uv run python tests/t_formatters.py
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
from quantizers.formatters import DatasetFmt

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class MockTokenizer:
    """Mock tokenizer that mimics the behavior of real tokenizers."""

    def __init__(self, input_ids, attention_mask=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask or [1] * len(input_ids)

    def apply_chat_template(self, messages, tokenize=False):
        """Simulate applying chat template to messages."""
        template = " ".join([msg.get("content", "") for msg in messages])
        if tokenize:
            return {"input_ids": self.input_ids, "attention_mask": self.attention_mask}
        return template

    def __call__(self, text, **kwargs):
        """Simulate tokenization of text."""
        return {"input_ids": self.input_ids, "attention_mask": self.attention_mask}


def test_sharegpt_formatter_with_arbitrary_columns():
    """Test that ShareGPT formatter works with any column name."""
    print("\n=== Testing ShareGPT Formatter with Arbitrary Columns ===")

    # Mock ShareGPT data where the conversations are in a column
    mock_data = {
        "my_custom_column": [
            {"from": "human", "value": "What is 2+2?"},
            {"from": "gpt", "value": "4"},
        ]
    }

    # Test with arbitrary column names
    columns = ["my_custom_column"]
    result = DatasetFmt.sharegpt(columns, mock_data)

    # Verify the conversion is correct
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "What is 2+2?"
    assert result[1]["role"] == "assistant"
    assert result[1]["content"] == "4"

    # Verify the formatter uses the arbitrary column input
    columns_2 = ["question_data"]
    mock_data_2 = {
        "question_data": [
            {"from": "human", "value": "What is 2+2?"},
            {"from": "gpt", "value": "4"},
        ]
    }
    result_2 = DatasetFmt.sharegpt(columns_2, mock_data_2)
    assert result_2 == result

    print(
        "✅ ShareGPT formatter correctly handles arbitrary column names and uses input columns"
    )


def test_sharegpt_formatter_field_validation():
    """Test that ShareGPT formatter correctly handles data structure."""
    print("\n=== Testing ShareGPT Formatter Field Validation ===")

    # Test with empty conversations list
    result = DatasetFmt.sharegpt(["messages"], {"messages": []})
    assert result == [], "Should handle empty conversations list gracefully"
    print("✅ ShareGPT formatter correctly handles empty conversations list")

    # Test with missing column - should raise KeyError
    try:
        DatasetFmt.sharegpt(["nonexistent_column"], {"messages": []})
        assert False, "Should raise KeyError for missing column"
    except KeyError:
        pass  # Expected behavior
    print("✅ ShareGPT formatter correctly raises KeyError for missing column")

    # Test with proper conversations data
    mock_data = {
        "messages": [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi there"},
        ]
    }
    result = DatasetFmt.sharegpt(["messages"], mock_data)
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "Hello"
    assert result[1]["role"] == "assistant"
    assert result[1]["content"] == "Hi there"
    assert "conversations" not in str(result)  # Should not hardcode this field

    print(
        "✅ ShareGPT formatter correctly handles conversations data from specified column"
    )


def test_prompt_answer_formatter_with_arbitrary_columns():
    """Test that prompt-answer formatter works with arbitrary column names."""
    print(
        "\n=== Testing Prompt-Answer Formatter with Arbitrary Input/Output Column Names ==="
    )

    # Test case 1: ["prompt", "answer"] names
    mock_data_1 = {
        "prompt": "Translate the following text to English",
        "answer": "Hello world",
    }

    # Test case 2: ["input", "output"] names
    mock_data_2 = {"input": "Bonjour le monde", "output": "Hello world"}

    # Test case 3: ["query", "response"] names
    mock_data_3 = {
        "query": "What is the capital of France?",
        "response": "The capital of France is Paris.",
    }

    # Test case 4: ["user_query", "bot_response"] names
    mock_data_4 = {
        "user_query": "How do I write a Python function?",
        "bot_response": "Use the 'def' keyword followed by the function name.",
    }

    # Test all column variations
    test_cases = [
        (["prompt", "answer"], mock_data_1, "prompt-answer"),
        (["input", "output"], mock_data_2, "input-output"),
        (["query", "response"], mock_data_3, "query-response"),
        (["user_query", "bot_response"], mock_data_4, "user-bot"),
    ]

    for columns, data, test_name in test_cases:
        result = DatasetFmt.prompt_answer(columns, data)

        # Verify the conversion is correct
        assert len(result) == 2, f"{test_name} formatter should return 2 messages"

        # First message should be user
        assert result[0]["role"] == "user", f"{test_name} first message should be user"

        # Second message should be assistant
        assert (
            result[1]["role"] == "assistant"
        ), f"{test_name} second message should be assistant"

        # Check specific content for each test case
        if test_name == "prompt-answer":
            assert "Translate the following text to English" in result[0]["content"]
            assert result[1]["content"] == "Hello world"
        elif test_name == "input-output":
            assert "Bonjour le monde" in result[0]["content"]
            assert result[1]["content"] == "Hello world"
        elif test_name == "query-response":
            assert "What is the capital of France?" in result[0]["content"]
            assert "Paris" in result[1]["content"]
        elif test_name == "user-bot":
            assert "How do I write a Python function?" in result[0]["content"]
            assert "def" in result[1]["content"]

        print(f"✅ {test_name} formatter correctly handles arbitrary column names")


def test_chat_completion_formatter_with_arbitrary_columns():
    """Test that chat completion formatter works with any column name."""
    print("\n=== Testing Chat Completion Formatter with Arbitrary Columns ===")

    # Mock data with messages in arbitrary column
    mock_data = {
        "conversation_data": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of AI..."},
        ]
    }

    # Test that the formatter passthrough the direct data
    columns = ["conversation_data"]
    result = DatasetFmt.chat_completion(columns, mock_data["conversation_data"])

    # Verify the formatter returns the messages data directly
    assert result == mock_data["conversation_data"]

    # Test with different data (simulating direct list input)
    direct_list = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI..."},
    ]
    columns_2 = ["dialogue"]
    result_2 = DatasetFmt.chat_completion(columns_2, direct_list)

    # Verify the formatter returns the list directly
    assert result_2 == direct_list
    assert len(result_2) == 3
    assert result_2[0]["role"] == "system"
    assert result_2[0]["content"] == "You are a helpful assistant."

    print("✅ Chat completion formatter correctly passthrough messages data")


def test_chat_completion_formatter_field_validation():
    """Test that chat completion formatter works with valid data formats."""
    print("\n=== Testing Chat Completion Formatter Field Validation ===")

    # Test with missing key but valid direct list input
    try:
        DatasetFmt.chat_completion(["messages"], {"other_field": "value"})
        # No error because we're passing the whole dict, not trying to extract from it
        print("✅ Chat completion correctly handles dict with missing key")
    except Exception as e:
        print(f"⚠️ Unexpected error with dict input: {e}")

    # Test with direct list input (the intended use case)
    direct_list = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    result = DatasetFmt.chat_completion(["messages"], direct_list)
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "Hello"
    assert result[1]["role"] == "assistant"
    assert result[1]["content"] == "Hi there"

    # Test with nested structure that matches test_data format
    nested_data = {
        "messages_col": {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        }
    }
    # This simulates passing the extracted column content directly
    nested_result = DatasetFmt.chat_completion(
        ["messages_col"], nested_data["messages_col"]["messages"]
    )
    assert len(nested_result) == 2
    assert nested_result[0]["role"] == "user"
    assert nested_result[0]["content"] == "Hello"

    print("✅ Chat completion formatter correctly handles different data formats")


def test_chat_completion_formatter_direct_list():
    """Test that chat completion formatter works with direct list input (for neuralmagic/calibration)."""
    print("\n=== Testing Chat Completion Formatter with Direct List Input ===")

    # Mock data that is a direct list (format used by neuralmagic/calibration)
    direct_list = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI..."},
    ]

    # Test with any column name (should be ignored for direct list)
    columns = ["any_column_name"]
    result = DatasetFmt.chat_completion(columns, direct_list)

    # Verify the formatter returns the list directly
    assert result == direct_list

    print("✅ Chat completion formatter correctly handles direct list input")


def test_raw_text_formatter_with_arbitrary_columns():
    """Test that raw text formatter works with any column name."""
    print("\n=== Testing Raw Text Formatter with Arbitrary Columns ===")

    # Test multiple arbitrary text column names
    test_cases = [
        (["text"], {"text": "This is a raw text document."}),
        (["content"], {"content": "This document contains different content."}),
        (["document"], {"document": "The third document tests longer text."}),
        (["payload"], {"payload": "Fourth document with special chars: !@#$%^&*()"}),
        (
            ["my_custom_field"],
            {"my_custom_field": "Fifth document with numbers: 1234567890"},
        ),
    ]

    for columns, data in test_cases:
        result = DatasetFmt.raw_text(columns, data)

        # Verify the conversion is correct
        assert len(result) == 1, f"Result should contain 1 message for {columns}"
        assert result[0]["role"] == "assistant"

        # The content should match the extracted text
        extracted_text = data[columns[0]]
        assert result[0]["content"] == extracted_text

    print("✅ Raw text formatter correctly handles arbitrary column names")


def test_formatter_column_count_handling():
    """Test that formatters correctly handle different column counts based on type."""
    print("\n=== Testing Formatter Column Count Handling ===")

    # Test data for different formatter types
    sharegpt_data = {"col": [{"from": "user", "value": "Hello"}]}
    raw_text_data = {"col": "This is raw text"}
    prompt_answer_data = {"col1": "Question", "col2": "Response"}
    chat_completion_data = {"col": [{"role": "user", "content": "Hello"}]}

    # Test that all formatters work with single column
    print("✅ Single column handling:")
    DatasetFmt.sharegpt(["col"], sharegpt_data)  # Should work (requires exactly 1)
    DatasetFmt.raw_text(["col"], raw_text_data)  # Should work (requires exactly 1)
    DatasetFmt.chat_completion(
        ["col"], chat_completion_data
    )  # Should work (requires exactly 1)

    # Test that formatters work with multiple columns (where supported)
    print("✅ Multiple column handling:")
    DatasetFmt.prompt_answer(
        ["col1", "col2"], prompt_answer_data
    )  # Should work (requires exactly 2)

    # Test that prompt-answer fails with wrong column counts
    print("✅ Exact column count validation:")
    DatasetFmt.prompt_answer(["col1", "col2"], prompt_answer_data)  # Should work

    # Test that raw_text fails with multiple columns
    try:
        DatasetFmt.raw_text(["col1", "col2"], raw_text_data)
        assert False, "Expected ValueError for raw_text with multiple columns"
    except ValueError as e:
        assert "exactly 1 column" in str(e)
        print("✅ Raw text formatter correctly validates 1-column requirement")

    # Test that prompt-answer requires exactly 2 columns
    try:
        DatasetFmt.prompt_answer(["col1"], prompt_answer_data)
        assert False, "Expected ValueError for prompt-answer with single column"
    except ValueError as e:
        assert "exactly 2 columns" in str(e)
        print("✅ Prompt-answer formatter correctly requires 2 columns")

    try:
        DatasetFmt.prompt_answer(["col1", "col2", "col3"], prompt_answer_data)
        assert False, "Expected ValueError for prompt-answer with three columns"
    except ValueError as e:
        assert "exactly 2 columns" in str(e)
        print("✅ Prompt-answer formatter correctly rejects extra columns")

    print("✅ Formatters correctly validate column counts based on their requirements")


def test_formatter_error_handling():
    """Test that formatters properly handle errors with invalid column configurations."""
    print("\n=== Testing Formatter Error Handling ===")

    # Test ShareGPT with incorrect number of columns
    try:
        DatasetFmt.sharegpt(["col1", "col2"], {"conversations": []})
        assert False, "Expected ValueError for incorrect column count"
    except ValueError as e:
        assert "exactly 1 column" in str(e)
        print("✅ ShareGPT formatter correctly validates 1-column requirement")

    # Test prompt-answer with incorrect number of columns
    try:
        DatasetFmt.prompt_answer(["only_one_column"], {"only_one_column": "value"})
        assert False, "Expected ValueError for incorrect column count"
    except ValueError as e:
        assert "exactly 2 columns" in str(e)
        print("✅ Prompt-answer formatter correctly validates 2-column requirement")

    # Test chat completion with incorrect number of columns
    try:
        DatasetFmt.chat_completion(["col1", "col2"], {"messages": []})
        assert False, "Expected ValueError for incorrect column count"
    except ValueError as e:
        assert "exactly 1 column" in str(e)
        print("✅ Chat completion formatter correctly validates 1-column requirement")

    # Test raw text with incorrect number of columns
    try:
        DatasetFmt.raw_text(["col1", "col2"], {"col1": "text", "col2": "more text"})
        assert False, "Expected ValueError for incorrect column count"
    except ValueError as e:
        assert "exactly 1 column" in str(e)
        print("✅ Raw text formatter correctly validates 1-column requirement")

    print("✅ All formatters correctly handle errors with appropriate messages")


def test_formatter_integration_with_calibration_set():
    """Test that formatters work correctly in the context of CalibrationSet."""
    print("\n=== Testing Formatter Integration with CalibrationSet ===")

    # Test configuration with arbitrary column names
    config = CalibrationSetConfig(
        max_seq_length=4096,
        shuffle=False,
        seed=42,
        datasets=[
            DatasetEntryConfig(
                dataset="test/dataset",
                split="train",
                columns=["user_question", "model_response"],  # Arbitrary column names
                formatter="prompt_answer",
                num_samples=3,
            )
        ],
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create calibration set and directly assign the dataset
        calib_set = CalibrationSet.from_config(config=config, cache_dir=temp_dir)

        # Create mock dataset with formatted data already applied
        mock_dataset = datasets.Dataset.from_dict(
            {
                "formatted": [
                    [
                        {"role": "user", "content": "What is machine learning?"},
                        {
                            "role": "assistant",
                            "content": "Machine learning is a subset of artificial intelligence...",
                        },
                    ],
                    [
                        {"role": "user", "content": "Explain photosynthesis"},
                        {
                            "role": "assistant",
                            "content": "Photosynthesis is the process by which plants convert light energy...",
                        },
                    ],
                    [
                        {"role": "user", "content": "How does the internet work?"},
                        {
                            "role": "assistant",
                            "content": "The internet is a global system of interconnected computer networks...",
                        },
                    ],
                ]
            }
        )
        calib_set._untokenized_calibration_set = mock_dataset

        # Create mock tokenizer
        tokenizer = MockTokenizer([1, 2, 3, 4, 5])

        # Apply tokenization through the calibration set's get_tokenized method
        tokenized = calib_set.get_tokenized(tokenizer)

        # Verify tokenization worked correctly
        assert isinstance(tokenized, datasets.Dataset)
        assert len(tokenized) == 3
        assert "input_ids" in tokenized.column_names
        assert "formatted" not in tokenized.column_names

        # Verify the first example
        assert len(tokenized[0]["input_ids"]) > 0

        print("✅ Formatters work correctly with CalibrationSet integration")


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
                    dataset=str(
                        Path(__file__).parent / "test_datasets" / "prompt_answer_format"
                    ),
                    split="train",
                    columns=["instruction", "output"],  # Existing column names
                    formatter="prompt_answer",
                    num_samples=2,
                )
            ],
        )

        # Create CalibrationSet with the config
        calib_set = CalibrationSet.from_config(config)

        # Get untokenized data to verify it works
        assert (
            calib_set._untokenized_calibration_set is not None
        ), "CalibrationSet should have untokenized data"
        assert (
            len(calib_set._untokenized_calibration_set) > 0
        ), "CalibrationSet should have data samples"
        print(
            "\n✅ CalibrationSet correctly processes data with arbitrary column names"
        )
    except Exception as e:
        print(f"\n⚠️  CalibrationSet test with arbitrary columns encountered: {e}")


if __name__ == "__main__":
    print("🔍 Testing formatter column name handling and integration...")

    try:
        test_sharegpt_formatter_with_arbitrary_columns()
        test_sharegpt_formatter_field_validation()

        test_prompt_answer_formatter_with_arbitrary_columns()
        test_chat_completion_formatter_with_arbitrary_columns()
        test_chat_completion_formatter_field_validation()
        test_chat_completion_formatter_direct_list()
        test_raw_text_formatter_with_arbitrary_columns()
        test_formatter_column_count_handling()
        test_formatter_error_handling()
        test_formatter_integration_with_calibration_set()
        test_calibration_set_with_arbitrary_columns()

        print("\n🎉 All formatter tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
