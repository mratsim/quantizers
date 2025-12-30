"""
Tests for the enhanced raw_text formatter.
"""

from quantizers.formatters import DatasetFmt


class TestRawTextFormatter:
    """Test cases for the raw_text formatter with prefix support."""

    def test_raw_text_without_prefix(self):
        """Test raw_text formatter without prefix (original functionality)."""
        columns = ["text"]
        data = {"text": "This is some sample code"}

        result = DatasetFmt.raw_text(columns, data)

        assert len(result) == 1
        assert result[0] == {"role": "assistant", "content": "This is some sample code"}

    def test_raw_text_with_prefix(self):
        """Test raw_text formatter with prefix."""
        columns = ["text"]
        data = {"text": "This is some sample code"}
        prefix = "explain this code"

        result = DatasetFmt.raw_text(columns, data, prefix=prefix)

        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "explain this code"}
        assert result[1] == {"role": "assistant", "content": "This is some sample code"}

    def test_raw_text_with_different_column(self):
        """Test raw_text formatter with a different column name."""
        columns = ["content"]
        data = {"content": "def hello_world():\n    print('Hello, World!')"}
        prefix = "explain this code"

        result = DatasetFmt.raw_text(columns, data, prefix=prefix)

        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "explain this code"}
        assert result[1] == {"role": "assistant", "content": "def hello_world():\n    print('Hello, World!')"}

    def test_raw_text_prefix_with_language_instruction(self):
        """Test raw_text formatter with language prefix."""
        columns = ["prompt"]
        data = {"prompt": "Write a function to calculate the factorial of a number"}
        prefix = "Solve the following problem using Python"

        result = DatasetFmt.raw_text(columns, data, prefix=prefix)

        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Solve the following problem using Python"}
        assert result[1] == {"role": "assistant", "content": "Write a function to calculate the factorial of a number"}

    def test_raw_text_with_empty_prefix(self):
        """Test raw_text formatter with empty prefix."""
        columns = ["text"]
        data = {"text": "This is some sample code"}
        prefix = ""

        result = DatasetFmt.raw_text(columns, data, prefix=prefix)

        assert len(result) == 1
        assert result[0] == {"role": "assistant", "content": "This is some sample code"}

    def test_raw_text_get_formatter(self):
        """Test getting raw_text formatter through get_formatter method."""
        formatter = DatasetFmt.get_formatter("raw_text")
        assert formatter == DatasetFmt.raw_text

        # Test with formatter call
        columns = ["text"]
        data = {"text": "Sample content"}

        result = formatter(columns, data)
        assert len(result) == 1
        assert result[0] == {"role": "assistant", "content": "Sample content"}

        # Test with formatter call and prefix
        result = formatter(columns, data, prefix="explain this code")
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "explain this code"}
        assert result[1] == {"role": "assistant", "content": "Sample content"}
