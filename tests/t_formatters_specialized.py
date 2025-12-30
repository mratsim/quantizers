"""Tests for specialized dataset format converters.

This module contains tests for formatters that are specific to certain datasets.
"""
# ruff: noqa: E501

import sys
from pathlib import Path

import pytest

# Add the src directory to the path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantizers.formatters import DatasetFmt


class TestDeepMindCodeContestsFormatter:
    """Test cases for the DeepMind Code Contests formatter."""

    def test_formatter_with_valid_data(self):
        """Test the formatter with valid data."""
        # Sample data mimicking the structure of deepmind/code_contests dataset
        sample_data = {
            "description": "Problem description.\nVipul is a hardworking super-hero who maintains the bracket ratio of all the strings in the world.",
            "public_tests": {"input": ["test"], "output": ["output"]},
            "solutions": {
                "language": [1, 1, 1],
                "solution": [
                    "for _ in range(input()):\n    try:\n        eval(raw_input())\n        print 'YES'\n    except TypeError:\n        print 'YES'\n    except:\n        print 'NO'",
                    'for _ in range(input()):\n    stck = []\n    res = "YES"\n    for x in ins:\n        if x == "(":\n            stck.append(x)\n        else:\n            if len(stck)>0:\n                stck.pop()\n            else:\n                res = "NO"\n                break\n    if len(stck) > 0: res = "NO" \n    print res',
                    "for _ in range(input()):\n    try: eval(raw_input()); print 'YES'\n    except TypeError: print 'YES'\n    except: print 'NO'",
                ],
            },
        }

        # The formatter receives a list of columns and a dictionary of extracted data
        columns = ["sample_column"]
        formatted = DatasetFmt.deepmind_code_contests(columns, {"sample_column": sample_data})

        # Check the result
        assert len(formatted) == 2  # Description + Solution

        # Check message roles
        assert formatted[0]["role"] == "user"
        assert formatted[1]["role"] == "assistant"

        # Check message contents
        assert "Problem description" in formatted[0]["content"]
        assert "for _ in range(input()):" in formatted[1]["content"]

    def test_formatter_with_missing_description(self):
        """Test the formatter when description is missing."""
        sample_data = {"description": "", "solutions": {"language": [1], "solution": ['print("Hello")']}}

        columns = ["sample_column"]
        formatted = DatasetFmt.deepmind_code_contests(columns, {"sample_column": sample_data})

        # Should only have one message (assistant)
        assert len(formatted) == 1
        assert formatted[0]["role"] == "assistant"
        assert formatted[0]["content"] == 'print("Hello")'

    def test_formatter_with_no_solutions(self):
        """Test the formatter when solutions are missing."""
        sample_data = {"description": "A problem description", "solutions": {}}

        columns = ["sample_column"]
        formatted = DatasetFmt.deepmind_code_contests(columns, {"sample_column": sample_data})

        # Should only have one message (user)
        assert len(formatted) == 1
        assert formatted[0]["role"] == "user"
        assert formatted[0]["content"] == "A problem description"

    def test_formatter_with_empty_data(self):
        """Test the formatter with empty data."""
        sample_data = {}

        columns = ["sample_column"]
        formatted = DatasetFmt.deepmind_code_contests(columns, {"sample_column": sample_data})

        # Should be empty
        assert len(formatted) == 0

    def test_formatter_with_no_solutions_list(self):
        """Test the formatter when solutions list is empty."""
        sample_data = {"description": "A problem description", "solutions": {"language": [], "solution": []}}

        columns = ["sample_column"]
        formatted = DatasetFmt.deepmind_code_contests(columns, {"sample_column": sample_data})

        # Should only have one message (user)
        assert len(formatted) == 1
        assert formatted[0]["role"] == "user"
        assert formatted[0]["content"] == "A problem description"

    def test_formatter_with_wrong_column_count(self):
        """Test the formatter with wrong number of columns."""
        sample_data = {"description": "A problem description"}

        # Should fail with more than 1 column
        with pytest.raises(ValueError, match="DeepMind Code Contests format requires exactly 1 column"):
            DatasetFmt.deepmind_code_contests(["col1", "col2"], {"col1": sample_data})

        # Should fail with 0 columns
        with pytest.raises(ValueError, match="DeepMind Code Contests format requires exactly 1 column"):
            DatasetFmt.deepmind_code_contests([], {"col1": sample_data})

    def test_formatter_with_get_formatter(self):
        """Test getting the formatter through the get_formatter method."""
        formatter = DatasetFmt.get_formatter("deepmind_code_contests")
        assert formatter == DatasetFmt.deepmind_code_contests

        # Test with invalid formatter name
        with pytest.raises(ValueError, match="Unknown formatter"):
            DatasetFmt.get_formatter("invalid_formatter")

    def test_formatter_with_real_dataset_sample(self):
        """Test the formatter with a sample from the actual deepmind/code_contests dataset."""
        # This sample is extracted from the actual dataset
        sample_data = {
            "name": "brcktsrm",
            "description": "Problem description.\nVipul is a hardworking super-hero who maintains the bracket ratio of all the strings in the world. Recently he indulged himself in saving the string population so much that he lost his ability for checking brackets (luckily, not permanently ).Being his super-hero friend help him in his time of hardship.",
            "public_tests": {"input": ["3\n((()))\n(())()\n()(()"], "output": ["YES\nYES\nNO"]},
            "solutions": {
                "language": [1, 1, 1],
                "solution": [
                    "for _ in range(input()):\n    try:\n        eval(raw_input())\n        print 'YES'\n    except TypeError:\n        print 'YES'\n    except:\n        print 'NO'",
                    'for _ in range(input()):\n    ins = raw_input().strip()\n    stck = []\n    res = "YES"\n    for x in ins:\n        if x == "(":\n            stck.append(x)\n        else:\n            if len(stck)>0:\n                stck.pop()\n            else:\n                res = "NO"\n                break\n    if len(stck) > 0: res = "NO" \n    print res',
                    "for _ in range(input()):\n    try: eval(raw_input()); print 'YES'\n    except TypeError: print 'YES'\n    except: print 'NO'",
                ],
            },
        }

        columns = ["sample_column"]
        formatted = DatasetFmt.deepmind_code_contests(columns, {"sample_column": sample_data})

        # Should include both description and solution
        assert len(formatted) == 2
        assert "Vipul is a hardworking super-hero" in formatted[0]["content"]
        assert "for _ in range(input()):" in formatted[1]["content"]
