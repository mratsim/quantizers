#!/usr/bin/env python3
import sys
from pathlib import Path

from datasets import load_dataset
from jinja2 import Environment, StrictUndefined

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantizers.calibration_sets import CalibrationSetConfig


def test_humaneval_jinja_templates():
    """Test that Jinja templates work with our test dataset."""

    # Load our config from the main test configuration
    config_path = Path(__file__).parent / "test_datasets" / "t_calibrate_diverse_columns.yaml"
    config = CalibrationSetConfig.from_file(config_path)

    # Find the humaneval dataset in the config
    dataset_config = None
    for ds in config.datasets:
        if "diversoailab_humaneval_rust" in ds.dataset:
            dataset_config = ds
            break

    assert dataset_config is not None, "Should find diversoailab_humaneval_rust dataset in config"
    prefix = dataset_config.formatter_params.get("prefix", "")

    # Verify it contains a Jinja template
    assert "{{" in prefix and "}}" in prefix, "Should contain Jinja template"

    # Test template rendering
    jinja_env = Environment(undefined=StrictUndefined, autoescape=True)
    jinja_env.globals.update({"hash": hash})

    # Test with a few different rows
    languages = ["Python", "Rust", "JavaScript", "Java", "C++"]
    found_languages = set()

    for i in range(5):
        sample_row = {"prompt": f"Test prompt {i}"}
        template = jinja_env.from_string(prefix)
        rendered = template.render(row=sample_row)
        # Extract language from the rendered string
        for lang in languages:
            if lang in rendered:
                found_languages.add(lang)
                break

    # Should have found multiple languages
    assert len(found_languages) >= 2, f"Should find multiple languages, found: {found_languages}"

    dataset = load_dataset(str(Path(__file__).parent / "test_datasets" / "diversoailab_humaneval_rust"), split="train")

    # Verify we have 10 samples
    assert len(dataset) == 10, f"Expected 10 samples, got {len(dataset)}"

    print("✅ Jinja templates work correctly with our test dataset")
