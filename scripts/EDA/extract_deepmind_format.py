#!/usr/bin/env python3
"""
Script to extract and format the top 10 rows from the deepmind/code_contests dataset.

This script demonstrates how to use the new deepmind_code_contests formatter
to convert the dataset into a conversational format suitable for quantization.
"""

import os
import sys

from datasets import load_dataset

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from quantizers.formatters import DatasetFmt


def extract_and_format_dataset():
    """Extract the top 10 rows from deepmind/code_contests and format them."""
    print("Loading deepmind/code_contests dataset...")

    # Load the dataset
    dataset = load_dataset("deepmind/code_contests", split="train")
    print(f"Total rows in dataset: {len(dataset)}")

    # Extract and format the first 10 rows
    print("\nExtracting and formatting first 10 rows...")
    formatted_data = []

    for i in range(min(10, len(dataset))):
        row = dataset[i]

        # Format using the deepmind_code_contests formatter
        # We pass the entire row as if it's from a single column
        try:
            formatted_row = DatasetFmt.deepmind_code_contests(["data"], {"data": row})
            formatted_data.append(formatted_row)

            print(f"\nRow {i + 1}:")
            print(f"Problem name: {row.get('name', 'N/A')}")
            print(f"Difficulty: {row.get('difficulty', 'N/A')}")

            # Print the formatted conversation
            for j, msg in enumerate(formatted_row):
                print(f"\n{msg['role'].upper()}:")
                print(msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"])

        except Exception as e:
            print(f"\nError formatting row {i + 1}: {str(e)}")
            continue

    print("\n" + "=" * 50)
    print("Extraction complete!")
    print(f"Successfully formatted {len(formatted_data)} conversations")

    return formatted_data


if __name__ == "__main__":
    try:
        # Extract and format the dataset
        formatted_data = extract_and_format_dataset()

        # Print summary
        print("\nSummary of extracted data:")
        print(f"Total formatted conversations: {len(formatted_data)}")

        for i, conv in enumerate(formatted_data):
            has_user = any(msg["role"] == "user" for msg in conv)
            has_assistant = any(msg["role"] == "assistant" for msg in conv)

            print(f"\nConversation {i + 1}:")
            print(f"  - User message: {'Yes' if has_user else 'No'}")
            print(f"  - Assistant message: {'Yes' if has_assistant else 'No'}")
            print(f"  - Total messages: {len(conv)}")

    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)
