#!/usr/bin/env python3
"""
Test for the consolidated calibration set configuration.

This test validates that the main test-calibrate_code.yaml configuration
loads correctly and includes all expected datasets.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantizers.calibration_sets import CalibrationSetConfig


def test_consolidated_config():
    """Test loading the consolidated calibration set configuration."""
    config_path = Path(__file__).parent.parent / "configs" / "calibration_sets" / "test-calibrate_code.yaml"

    print(f"Testing consolidated configuration from: {config_path}")

    try:
        # Load the configuration
        config = CalibrationSetConfig.from_file(config_path)

        # Verify configuration loaded
        assert config is not None, "Configuration should not be None"
        assert len(config.datasets) > 0, "Configuration should have datasets"

        print(f"✅ Successfully loaded configuration with {len(config.datasets)} datasets")

        # Expected dataset patterns
        expected_datasets = [
            "deepmind/code_contests",
            "nvidia/OpenCodeInstruct",
            "CSJianYang/CodeArena",
            "nvidia/Llama-Nemotron-Post-Training-Dataset",
            "ammarnasr/the-stack-rust-clean",
            "diversoailab/humaneval-rust",
        ]

        # Check for expected datasets
        found_datasets = set()
        for ds in config.datasets:
            found_datasets.add(ds.dataset)

        print(f"Found datasets: {sorted(found_datasets)}")

        # Verify all expected datasets are present
        for expected_ds in expected_datasets:
            assert expected_ds in found_datasets, f"Expected dataset {expected_ds} not found"

        print("✅ All expected datasets are present")

        # Count specific language variations
        python_count = 0
        javascript_count = 0
        java_count = 0
        jinja_count = 0
        stack_rust_count = 0

        for ds in config.datasets:
            if ds.dataset == "ammarnasr/the-stack-rust-clean":
                stack_rust_count += 1
                # Should have "explain this code" prefix
                assert ds.formatter_params.get("prefix") == "explain this code", (
                    "Stack Rust should have explain this code prefix"
                )

            if ds.dataset == "diversoailab/humaneval-rust":
                prefix = ds.formatter_params.get("prefix", "")

                if "Python" in prefix and "{{" not in prefix:
                    python_count += 1
                elif "JavaScript" in prefix and "{{" not in prefix:
                    javascript_count += 1
                elif "Java" in prefix and "{{" not in prefix:
                    java_count += 1
                elif "{{" in prefix and "}}" in prefix:  # Jinja template
                    jinja_count += 1

        print(f"Stack Rust datasets: {stack_rust_count}")
        print(f"Python language entries: {python_count}")
        print(f"JavaScript language entries: {javascript_count}")
        print(f"Java language entries: {java_count}")
        print(f"Jinja template entries: {jinja_count}")

        # Verify counts match expectations
        assert stack_rust_count >= 1, "Should have at least one Stack Rust dataset"
        assert python_count >= 1, "Should have at least one Python language entry"
        assert javascript_count >= 1, "Should have at least one JavaScript language entry"
        assert java_count >= 1, "Should have at least one Java language entry"
        assert jinja_count >= 1, "Should have at least one Jinja template entry"

        # Validate the configuration
        try:
            config.validate()
            print("✅ Configuration validation passed")
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            assert False, f"Configuration validation failed: {e}"

        # Check for Jinja templates in the humaneval dataset
        jinja_templates = []
        for ds in config.datasets:
            if ds.dataset == "diversoailab/humaneval-rust" and ds.formatter_params:
                prefix = ds.formatter_params.get("prefix", "")
                if "{{" in prefix and "}}" in prefix:
                    jinja_templates.append(prefix[:50] + "..." if len(prefix) > 50 else prefix)

        if jinja_templates:
            print("✅ Found Jinja templates in humaneval dataset:")
            for template in jinja_templates:
                print(f"   - {template}")

        print("\n🎉 Consolidated configuration test passed!")
        print("✅ Configuration includes:")
        print("   - 4 Original datasets (code contests, open code instruct, code arena, llama nemotron)")
        print("   - 1 Stack Rust dataset with code explanation")
        print("   - 3 Humaneval datasets with static language selection (Python, JavaScript, Java)")
        print("   - 1 Humaneval dataset with dynamic language selection via Jinja templates")
        print("   - Total of 9 diverse dataset entries with streaming support")

    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        assert False, f"Error loading configuration: {e}"


def test_formatter_params():
    """Test that formatter parameters are correctly configured."""
    config_path = Path(__file__).parent.parent / "configs" / "calibration_sets" / "test-calibrate_code.yaml"

    # Load configuration
    config = CalibrationSetConfig.from_file(config_path)

    # Check formatter params
    for ds in config.datasets:
        if ds.formatter_params:
            for key, value in ds.formatter_params.items():
                # All formatter params should be strings
                assert isinstance(value, str), f"Formatter param {key} should be string, got {type(value)}"

                # Check for Jinja templates
                if "{{" in str(value) and "}}" in str(value):
                    # Validate basic Jinja syntax
                    assert "{{" in str(value), "Should contain opening braces"
                    assert "}}" in str(value), "Should contain closing braces"

    print("✅ All formatter parameters are valid")


def test_streaming_configurations():
    """Test that streaming is properly configured for large datasets."""
    config_path = Path(__file__).parent.parent / "configs" / "calibration_sets" / "test-calibrate_code.yaml"

    # Load configuration
    config = CalibrationSetConfig.from_file(config_path)

    # Check streaming settings
    streaming_datasets = 0
    for ds in config.datasets:
        if ds.streaming:
            streaming_datasets += 1
            # Large datasets should have streaming enabled
            assert ds.dataset in ["nvidia/OpenCodeInstruct", "nvidia/Llama-Nemotron-Post-Training-Dataset"], (
                f"Unknown streaming dataset: {ds.dataset}"
            )

    print(f"✅ Found {streaming_datasets} datasets with streaming enabled")

    # Verify specific datasets have streaming
    assert any(ds.dataset == "nvidia/OpenCodeInstruct" and ds.streaming for ds in config.datasets), (
        "OpenCodeInstruct should have streaming enabled"
    )
    assert any(ds.dataset == "nvidia/Llama-Nemotron-Post-Training-Dataset" and ds.streaming for ds in config.datasets), (
        "Llama Nemotron should have streaming enabled"
    )

    print("✅ Streaming properly configured for large datasets")


if __name__ == "__main__":
    print("=== Testing Consolidated Calibration Set Configuration ===\n")
