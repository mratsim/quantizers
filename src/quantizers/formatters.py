"""
Dataset format converters for quantization calibration.

This module provides a namespace class `DatasetFmt` with static methods for converting
various dataset formats into the chat completion format expected by llm-compressor.

The formatters work with arbitrary column names passed from the calibration set loader.
Each formatter receives:
1. columns: List of column names that were used to extract data from the raw dataset
2. data: The extracted data (could be a single value or a dictionary)

The formatters should use these column names to find the relevant data they need.
"""

import logging
from typing import Any, Callable, Dict, List


class DatasetFmt:
    """
    Namespace class for dataset format converters.

    All methods are static - do not instantiate this class.
    """

    @staticmethod
    def sharegpt(columns: List[str], data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Convert ShareGPT conversation format to chat completion format.

        Convert ShareGPT conversation format to chat completion format.

        Based on original script handling: expects conversations as a list with "from" and "value" keys.

        Args:
            columns: Names of the columns containing the data
            data: Extracted data from the dataset (conversations list)

        Returns:
            List of message dicts with "role" and "content" keys
        """
        # Enforce correct number of columns
        if len(columns) != 1:
            raise ValueError(f"ShareGPT format requires exactly 1 column, got {len(columns)}: {columns}")

        # Extract conversation list from the specified column - mandatory column
        conversations = data[columns[0]]

        role_mapping = {"system": "system", "human": "user", "gpt": "assistant"}
        logger = logging.getLogger(__name__)

        messages = []
        for entry_idx, entry in enumerate(conversations):
            if not isinstance(entry, dict) or "from" not in entry or "value" not in entry:
                logger.warning(f"Skipping invalid conversation entry {entry_idx}: {entry}")
                continue

            role = role_mapping.get(entry.get("from", ""), "user")
            messages.append({"role": role, "content": entry.get("value", "")})

        return messages

    @staticmethod
    def prompt_answer(columns: List[str], data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Convert prompt-answer format to chat completion format.

        Uses arbitrary column names to extract prompt and answer.

        Args:
            columns: Names of the columns containing prompt and answer
            data: Extracted data from the dataset

        Returns:
            List of message dicts with "role" and "content" keys
        """
        # Enforce correct number of columns
        if len(columns) != 2:
            raise ValueError(f"Prompt-answer format requires exactly 2 columns, got {len(columns)}: {columns}")

        # Extract prompt and answer using arbitrary column names - mandatory columns
        prompt = data[columns[0]]
        answer = data[columns[1]]

        messages = []

        # Add user prompt if present
        if prompt:
            messages.append({"role": "user", "content": prompt})

        # Add assistant response if present
        if answer:
            messages.append({"role": "assistant", "content": answer})

        return messages

    @staticmethod
    def chat_completion(columns: List[str], data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Convert chat completion format to chat completion format.

        Extracts messages from the specified column and returns them for apply_chat_template.

        Args:
            columns: Names of the columns containing the data
            data: Extracted data from the dataset

        Returns:
            The messages array extracted from the specified column
        """
        # Enforce correct number of columns
        if len(columns) != 1:
            raise ValueError(f"Chat completion format requires exactly 1 column, got {len(columns)}: {columns}")

        # Extract messages from the specified mandatory column
        return data[columns[0]]

    @staticmethod
    def get_formatter(formatter_name: str) -> Callable:
        """
        Get the appropriate formatter function by name.

        Args:
            formatter_name: Name of the formatter to retrieve

        Returns:
            The formatter function

        Raises:
            ValueError: If formatter_name is not recognized
        """
        formatters = {
            "sharegpt": DatasetFmt.sharegpt,
            "prompt_answer": DatasetFmt.prompt_answer,
            "chat_completion": DatasetFmt.chat_completion,
            "raw_text": DatasetFmt.raw_text,
            "deepmind_code_contests": DatasetFmt.deepmind_code_contests,
        }

        if formatter_name not in formatters:
            raise ValueError(f"Unknown formatter: {formatter_name}")

        return formatters[formatter_name]

    @staticmethod
    def raw_text(columns: List[str], data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Convert raw text to assistant-turn format.

        Extracts text from the specified column and formats it as an assistant response.

        Args:
            columns: Names of the columns containing the text data
            data: Extracted data from the dataset

        Returns:
            List of message dict with "role" and "content" keys
        """
        if len(columns) != 1:
            raise ValueError(f"Raw text format requires exactly 1 column, got {len(columns)}: {columns}")

        # Extract text from the arbitrary column - mandatory column
        text_content = data[columns[0]]

        return [{"role": "assistant", "content": text_content}]

    @staticmethod
    def deepmind_code_contests(columns: List[str], data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Convert DeepMind Code Contests format to chat completion format.

        This formatter creates a conversation between user (problem description) and assistant (solution code).

        Args:
            columns: Names of the columns containing the data
            data: Extracted data from the dataset

        Returns:
            List of message dicts with "role" and "content" keys
        """
        # Enforce correct number of columns
        if len(columns) != 1:
            raise ValueError(f"DeepMind Code Contests format requires exactly 1 column, got {len(columns)}: {columns}")

        # Check if the column value is the entire dataset row (like when using any column name)
        # or just a part of it
        row_data = data[columns[0]]

        # If row_data is a string, it means we selected a column that just contains a string
        # We need to use the entire data instead
        if isinstance(row_data, str):
            row_data = data

        # Get the problem description
        description = row_data.get("description", "")

        # Get the first solution code
        solutions = row_data.get("solutions", {})
        solution_code = ""

        if solutions and "solution" in solutions and len(solutions["solution"]) > 0:
            solution_code = solutions["solution"][0]  # Take the first solution

        messages = []

        # Add problem description as user prompt
        if description:
            messages.append({"role": "user", "content": description})

        # Add solution code as assistant response
        if solution_code:
            messages.append({"role": "assistant", "content": solution_code})

        return messages
