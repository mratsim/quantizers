#!/usr/bin/env python3
"""
Script to verify the Jinja template modulus fix works correctly.
"""

import re
import sys
from pathlib import Path

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jinja2 import StrictUndefined


class MockRow:
    """Mock row object for testing Jinja templates."""

    def __init__(self, string_value="test"):
        self.string = string_value

    def __str__(self):
        return self.string


def modulus_check(template_str):
    """
    Check if a Jinja template has a list that's big enough for the modulus operation.
    """
    if "[hash(row|string)" in template_str and "]" in template_str:
        # Extract the modulus value from the template
        modulus_match = re.search(r"\[hash\(row\|string\)\s*%\s*(\d+)\]", template_str)
        if modulus_match:
            modulus_val = int(modulus_match.group(1))

            # Find the list to check its length
            list_match = re.search(r"(\[.*?\])\s*\[hash\(row\|string\)\s*%\s*\d+\]", template_str)
            if list_match:
                list_str = list_match.group(1)
                # Count elements in the list
                elements = [item.strip() for item in list_str.strip("[]").split(",") if item.strip()]
                if len(elements) < modulus_val:
                    return False, len(elements), modulus_val
    return True, None, None


def test_yaml_file(yaml_path):
    """Test a YAML file for modulus issues."""
    print(f"\nTesting file: {yaml_path}")

    with open(yaml_path, "r") as f:
        content = f.read()

    # Find all similar templates in this file
    patterns = re.findall(r"\[.*?hash\(row\|string\).*?\]", content)

    issues = []
    for pattern in patterns:
        is_safe, list_len, modulus = modulus_check(pattern)
        if not is_safe:
            issues.append({"pattern": pattern, "list_len": list_len, "modulus": modulus})

    if issues:
        print(f"❌ Found {len(issues)} potential modulus issue(s):")
        for issue in issues:
            print(f"   List has {issue['list_len']} elements but uses modulus {issue['modulus']}")
            print(f"   Pattern: {issue['pattern'][:100]}...")
        return False
    else:
        print("✅ No modulus issues found")
        return True


def test_jinja_templates():
    """Test that templates can be rendered without errors."""
    print("\nTesting Jinja template rendering...")

    # Test 1: Valid template (should work)
    valid_template = (
        "Solve the following problem using {{ ['Python', 'Rust', 'JavaScript', 'Java', 'C++'][hash(row|string) % 5] }}"
    )
    try:
        from jinja2 import Environment

        jinja_env = Environment(undefined=StrictUndefined, autoescape=True)
        # Add Python built-ins to Jinja context to mirror production behavior
        jinja_env.globals.update(
            {
                "hash": hash,
                "len": len,
                "abs": abs,
                "max": max,
                "min": min,
                "sum": sum,
                "sorted": sorted,
                "enumerate": enumerate,
                "zip": zip,
            }
        )

        mock_row = MockRow()
        template = jinja_env.from_string(valid_template)
        result = template.render(row=mock_row)
        print(f"✅ Valid template rendered correctly: {result[:60]}...")
    except Exception as e:
        print(f"❌ Valid template failed to render: {e}")
        return False

    # Test 2: Invalid template (should fail with our validation)
    invalid_template = (
        "Solve the following problem using {{ ['Python', 'Rust', 'JavaScript', 'Java', 'C++'][hash(row|string) % 10] }}"
    )
    is_safe, list_len, modulus = modulus_check(invalid_template)

    if not is_safe:
        print(f"✅ Invalid template correctly flagged as problematic (list: {list_len}, modulus: {modulus})")
    else:
        print("❌ Invalid template not flagged as problematic")
        return False

    return True


def main():
    """Run verification on YAML files"""
    print("Verifying Jinja template modulus fixes...")

    all_good = True

    # Check the test file we fixed
    test_path = Path(__file__).parent / "test_datasets" / "t_calibrate_diverse_columns.yaml"
    if test_path.exists():
        all_good &= test_yaml_file(test_path)
    else:
        print(f"❌ Test file not found: {test_path}")
        all_good = False

    # Check config files
    config_path = Path(__file__).parent.parent / "configs" / "calibration_sets"
    if config_path.exists():
        for yaml_file in config_path.glob("*.yaml"):
            all_good &= test_yaml_file(yaml_file)
    else:
        print(f"❌ Config directory not found: {config_path}")
        all_good = False

    # Test Jinja templates
    all_good &= test_jinja_templates()

    if all_good:
        print("\n🎉 All files passed modulus validation!")
    else:
        print("\n❌ Some files have potential modulus issues.")

    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
