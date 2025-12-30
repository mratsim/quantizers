#!/usr/bin/env python3
"""
Calibration-based quantization entry point.

Usage:
    uv run scripts/do_oneshot.py --config configs/quantize_qwen3-4b-awq.yaml
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from llmcompressor import oneshot

from quantizers.calibration_sets import CalibrationSet
from quantizers.config import load_quantization_config
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup environment variables to avoid warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault(
    "PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512"
)


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """Configure Python logging."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))  # type: ignore

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calibration-based quantization entry point"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to quantization configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Output directory for quantized model (default: model-AWQ)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Directory to cache calibration sets (default: ./cache)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (default: stdout only)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_name: str, revision: str = "main"):
    """Load model and tokenizer for quantization."""
    logging.info(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)

    return model, tokenizer


def main():
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_file, log_level)

    # Load configuration
    logging.info(f"Loading configuration from: {args.config}")
    config = load_quantization_config(args.config)

    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        # Generate output directory based on model name and recipe
        model_name = config.model.name
        if "/" in model_name:
            model_name = model_name.split("/")[1]

        # Extract recipe name from path to make output directory dynamic
        recipe_name = Path(config.quantization.recipe).stem.replace("recipe_", "")
        output_dir = f"outputs/{model_name}-{recipe_name}"

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        config.model.name, config.model.revision
    )

    # Load calibration set
    logging.info("Loading calibration set")

    # First try to load from cache
    start_time = time.time()
    # Check if calibration set is cached
    if CalibrationSet.is_cached(
        config.calibration_set_config, cache_dir=args.cache_dir
    ):
        calib_set = CalibrationSet.from_cache(
            config.calibration_set_config, cache_dir=args.cache_dir
        )
        cache_time = time.time() - start_time
        logging.info(f"Loaded calibration set from cache in {cache_time:.2f} seconds")
    else:
        logging.info("Cache miss, building calibration set from raw data")
        start_time = time.time()
        calib_set = CalibrationSet.from_config(
            config.calibration_set_config,
            cache_dir=args.cache_dir,
        )
        build_time = time.time() - start_time
        logging.info(f"Built calibration set from raw data in {build_time:.2f} seconds")

        # Save to cache immediately after building
        start_time = time.time()
        calib_set.save_to_cache()
        save_time = time.time() - start_time
        logging.info(f"Saved calibration set to cache in {save_time:.2f} seconds")

    # Get the tokenized dataset
    logging.info("Getting tokenized calibration dataset")
    start_time = time.time()
    tokenized_dataset = calib_set.get_tokenized(tokenizer)
    tokenize_time = time.time() - start_time
    logging.info(f"Tokenized calibration dataset in {tokenize_time:.2f} seconds")

    # Load quantization recipe from YAML file
    recipe_path = config.quantization.recipe
    if not Path(recipe_path).is_absolute() and args.config:
        # Try relative to config file first
        candidate = Path(args.config).parent / recipe_path
        if not candidate.exists():
            # Fall back to recipes subdirectory
            candidate = Path(args.config).parent / "recipes" / recipe_path
        recipe_path = str(candidate)

    # Verify recipe file exists
    if not Path(recipe_path).exists():
        raise ValueError(f"Recipe file not found: {recipe_path}")

    # Run quantization
    logging.info("Starting quantization")
    start_time = time.time()
    oneshot(
        model=model,
        recipe=recipe_path,
        dataset=tokenized_dataset,
        max_seq_length=config.calibration_set_config.max_seq_length,
        num_calibration_samples=len(tokenized_dataset),
    )
    oneshot_time = time.time() - start_time
    logging.info(f"Completed quantization in {oneshot_time:.2f} seconds")

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # Save quantized model
    logging.info(f"Saving quantized model to: {output_dir}")
    start_time = time.time()
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)
    save_time = time.time() - start_time
    logging.info(f"Saved quantized model in {save_time:.2f} seconds")

    logging.info(f"SUCCESS: Quantized model saved in {output_dir}")


if __name__ == "__main__":
    main()
