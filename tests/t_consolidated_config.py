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
            assert ds.formatter_params.get("prefix") == "explain this code", "Stack Rust should have explain this code prefix"

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
    config.validate()
    print("✅ Configuration validation passed")

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
                    # Jinja template detected - could add more validation here if needed
                    pass

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
            print(f"  - {ds.dataset}")

    print(f"✅ Found {streaming_datasets} datasets with streaming enabled")

    # Verify specific datasets have streaming
    assert any(ds.dataset == "nvidia/OpenCodeInstruct" and ds.streaming for ds in config.datasets), (
        "OpenCodeInstruct should have streaming enabled"
    )
    assert any(ds.dataset == "nvidia/Llama-Nemotron-Post-Training-Dataset" and ds.streaming for ds in config.datasets), (
        "Llama Nemotron should have streaming enabled"
    )

    print("✅ Streaming properly configured for large datasets")


def test_yaml_anchor_config():
    """Test that the YAML anchor configuration loads properly."""
    config_path = Path(__file__).parent.parent / "configs" / "calibration_sets" / "test-calibrate-code-anchor.yaml"

    print(f"Testing YAML anchor configuration from: {config_path}")

    # Load the configuration
    config = CalibrationSetConfig.from_file(config_path)

    # Verify configuration loaded
    assert config is not None, "Configuration should not be None"
    assert len(config.datasets) == 2, f"Configuration should have exactly 2 datasets, got {len(config.datasets)}"

    print(f"✅ Successfully loaded configuration with {len(config.datasets)} datasets")

    # Check that both expected datasets are present
    expected_datasets = ["diversoailab/humaneval-rust", "MathArena/project_euler"]
    found_datasets = set(ds.dataset for ds in config.datasets)

    print(f"Found datasets: {sorted(found_datasets)}")

    # Verify all expected datasets are present
    for expected_ds in expected_datasets:
        assert expected_ds in found_datasets, f"Expected dataset {expected_ds} not found"

    print("✅ Both expected datasets are present")

    # Verify that both use the *language_prefix alias
    jinja_templates = []
    for ds in config.datasets:
        prefix = ds.formatter_params.get("prefix", "")
        assert "{{" in prefix and "}}" in prefix, f"Dataset {ds.dataset} should have Jinja template"
        jinja_templates.append(prefix[:50] + "..." if len(prefix) > 50 else prefix)

    print("✅ Both datasets use Jinja templates with *language_prefix alias")

    # Specifically check that the language list contains all expected languages
    for ds in config.datasets:
        prefix = ds.formatter_params.get("prefix", "")
        # Check if it contains the expected languages
        assert "Python" in prefix, "Should contain Python"
        assert "JavaScript" in prefix, "Should contain JavaScript"
        assert "Rust" in prefix, "Should contain Rust"
        assert "Java" in prefix, "Should contain Java"
        assert "C++" in prefix, "Should contain C++"
        assert "{{" in prefix and "}}" in prefix, "Should contain Jinja template syntax"

    print("✅ All expected languages are present in the Jinja templates")

    # Extract language list from templates
    import re

    language_list_match = re.search(r"\[([^\]]+)\]", prefix)
    assert language_list_match, "Could not find language list in template"

    # Check that all datasets are using the same template content
    # Validate the language list structure

    # Validate that all datasets use the same template
    for ds in config.datasets:
        ds_prefix = ds.formatter_params.get("prefix", "")
        ds_language_match = re.search(r"\[([^\]]+)\]", ds_prefix)  # noqa E501
        assert ds_language_match, f"Dataset {ds.dataset} should have language list"
        assert ds_language_match.group(1) == language_list_match.group(1), (
            f"Dataset {ds.dataset} should use the same language list"
        )

    # Verify that the YAML anchor was resolved (literal anchor syntax should not appear in parsed values)
    for ds in config.datasets:
        prefix = ds.formatter_params.get("prefix", "")
        assert "*language_prefix" not in prefix, "YAML anchor should be resolved, not literal"

    # Count languages
    languages_str = language_list_match.group(1)
    languages = [lang.strip().strip("'\"") for lang in languages_str.split(",")]
    assert len(languages) == 60, f"Should have exactly 60 languages, got {len(languages)}"

    expected_languages = [
        "Python",
        "JavaScript",
        "Rust",
        "Java",
        "C++",
        "Lean",
        "Coq",
        "SML",
        "Agda",
        "Idris",
        "Racket",
        "x86-64 ASM",
        "ARM-64 ASM",
        "CUDA",
        "Vulkan",
        "Metal",
    ]
    for lang in expected_languages:
        assert lang in languages, f"Should contain {lang}"

    print(f"✅ Template contains {len(languages)} languages")

    # Validate the configuration
    config.validate()
    print("✅ Configuration validation passed")

    print("\n🎉 YAML anchor configuration test passed!")


def test_toolace_config():
    """Test loading the ToolACE calibration set configuration."""
    config_path = Path(__file__).parent.parent / "configs" / "calibration_sets" / "test-calibrate_toolace.yaml"

    print(f"Testing ToolACE configuration from: {config_path}")

    # Load the configuration
    config = CalibrationSetConfig.from_file(config_path)

    # Verify configuration loaded
    assert config is not None, "Configuration should not be None"
    assert len(config.datasets) == 1, f"Configuration should have exactly 1 dataset, got {len(config.datasets)}"

    print(f"✅ Successfully loaded configuration with {len(config.datasets)} datasets")

    # Check that the expected dataset is present
    expected_dataset = "tests/test_datasets/toolace"
    found_datasets = [ds.dataset for ds in config.datasets]

    print(f"Found datasets: {found_datasets}")

    # Verify the expected dataset is present
    assert expected_dataset in found_datasets, f"Expected dataset {expected_dataset} not found"

    print("✅ ToolACE dataset is present")

    # Check formatter configuration
    ds = config.datasets[0]
    assert ds.formatter == "chat_completion_with_sysprompt", "Should use chat_completion_with_sysprompt formatter"
    assert ds.columns == ["system", "conversations"], "Should use system and conversations columns"
    assert ds.num_samples == 5, "Should load 5 samples"

    # Validate the configuration
    config.validate()
    print("✅ Configuration validation passed")

    print("\n🎉 ToolACE configuration test passed!")


if __name__ == "__main__":
    print("=== Testing Consolidated Calibration Set Configuration ===\n")

    # Run all tests
    test_consolidated_config()
    print()
    test_formatter_params()
    print()
    test_streaming_configurations()
    print()
    test_yaml_anchor_config()
    print()
    test_toolace_config()
    print()
