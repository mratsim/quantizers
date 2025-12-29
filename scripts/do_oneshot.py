#!/usr/bin/env python3
"""
do_oneshot.py - Entry point for calibration-based quantization.

Usage:
    python scripts/do_oneshot.py --config configs/quantize_<model>-<method>.yaml

This script:
1. Loads YAML configuration
2. Sets up logging with timestamped files
3. Loads and caches calibration datasets
4. Runs oneshot quantization via llm-compressor
5. Saves quantized model to outputs/

Supports methods: GPTQ, AWQ, FP8, NVFP4A16
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add src to path for quantizer imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

from quantizer.config import (
    load_quantization_config,
    load_recipe,
    QuantizationRunConfig,
)
from quantizer.cache import CalibrationSetCache
from quantizer.logging_ import setup_logging, log_recipe_summary
from quantizer.formatters import DatasetFmt


def create_recipe_from_config(config: QuantizationRunConfig, recipe_data: dict):
    """
    Create llm-compressor recipe modifier from configuration.

    Args:
        config: QuantizationRunConfig with method and recipe info
        recipe_data: Loaded recipe YAML dictionary

    Returns:
        Initialized llm-compressor recipe modifier
    """
    method = config.quantization.method.lower()

    if method == "gptq":
        return create_gptq_recipe(recipe_data, config)
    elif method == "awq":
        return create_awq_recipe(recipe_data, config)
    elif method == "fp8":
        return create_fp8_recipe(recipe_data, config)
    elif method in ("nvfp4", "nvfp4a16"):
        return create_nvfp4_recipe(recipe_data, config)
    else:
        raise ValueError(f"Unsupported quantization method: {method}")


def create_gptq_recipe(recipe_data: dict, config: QuantizationRunConfig):
    """Create GPTQ recipe modifier."""
    modifiers_cfg = recipe_data.get("modifiers", [recipe_data])[0]

    ignore_patterns = modifiers_cfg.get("ignore", [])
    damp_percent = modifiers_cfg.get("damp_percent", 0.01)
    block_size = modifiers_cfg.get("block_size", 128)

    return GPTQModifier(
        targets=modifiers_cfg.get("targets", ["Linear"]),
        ignore=ignore_patterns,
        damp_percent=damp_percent,
        block_size=block_size,
        offload_device=torch.device("cpu"),
    )


def create_awq_recipe(recipe_data: dict, config: QuantizationRunConfig):
    """Create AWQ recipe modifier."""
    modifiers_cfg = recipe_data.get("modifiers", [recipe_data])[0]

    ignore_patterns = modifiers_cfg.get("ignore", [])
    config_groups = modifiers_cfg.get("config_groups", {})
    duo_scaling = modifiers_cfg.get("duo_scaling", True)

    return AWQModifier(
        targets=config_groups.get("group_0", {}).get("targets", ["Linear"]) if config_groups else ["Linear"],
        ignore=ignore_patterns,
        duo_scaling=duo_scaling,
        offload_device=torch.device("cpu"),
        config_groups=config_groups,
    )


def create_fp8_recipe(recipe_data: dict, config: QuantizationRunConfig):
    """Create FP8 recipe modifier using QuantizationModifier."""
    modifiers_cfg = recipe_data.get("modifiers", [recipe_data])[0]

    ignore_patterns = modifiers_cfg.get("ignore", [])

    # Determine scheme based on method variant
    scheme = "FP8_BLOCK" if config.quantization.method.lower() == "fp8_block" else "W8A8_FP8"

    return QuantizationModifier(
        targets=modifiers_cfg.get("targets", ["Linear"]),
        scheme=scheme,
        ignore=ignore_patterns,
    )


def create_nvfp4_recipe(recipe_data: dict, config: QuantizationRunConfig):
    """Create NVFP4 recipe modifier using QuantizationModifier."""
    modifiers_cfg = recipe_data.get("modifiers", [recipe_data])[0]

    ignore_patterns = modifiers_cfg.get("ignore", [])

    # Determine scheme based on method variant
    scheme = "NVFP4A16" if config.quantization.method == "nvfp4a16" else "NVFP4"

    return QuantizationModifier(
        targets=modifiers_cfg.get("targets", ["Linear"]),
        scheme=scheme,
        ignore=ignore_patterns,
    )


def process_calibration_datasets(
    config: QuantizationRunConfig,
    tokenizer,
    cache_dir: str = "./cache",
    overwrite_cache: bool = False
):
    """
    Load and process calibration datasets with caching.

    Args:
        config: QuantizationRunConfig with calibration_set or calibration_sets
        tokenizer: HuggingFace tokenizer
        cache_dir: Directory for dataset cache
        overwrite_cache: If True, ignore existing cache

    Returns:
        Concatenated dataset ready for calibration
    """
    cache = CalibrationSetCache(cache_dir=cache_dir, overwrite=overwrite_cache)

    # Get calibration configurations
    if config.calibration_set:
        # Single calibration set file reference
        from quantizer.config import load_calibration_set
        calib_config = load_calibration_set(config.calibration_set)
        configs = [calib_config]
    elif config.calibration_sets:
        # Inline list of calibration sets (legacy support)
        configs = [cs.__dict__ for cs in config.calibration_sets]
    else:
        raise ValueError("No calibration configuration found")

    # Build format function if specified
    format_fn = None
    text_column = "messages"

    # Check for reformatting_fn in first config (shared across configs)
    reformatting_fn_name = configs[0].get("reformatting_fn") if configs else None
    if reformatting_fn_name:
        format_fn = _get_reformatting_function(reformatting_fn_name)
        # Use appropriate text column based on format
        if reformatting_fn_name in ("chat_completion", "sharegpt", "alpaca", "prompt_answer"):
            text_column = "messages"
        else:
            text_column = "text"
    
    # Load all calibration sets
    dataset = cache.load_multiple(
        calibration_configs=configs,
        format_fn=format_fn,
        text_column=text_column,
    )
    
    return dataset


def _get_reformatting_function(fn_name: str):
    """
    Get reformatting function by name.
    
    Maps function names to DatasetFmt static methods.
    """
    mapping = {
        "sharegpt": DatasetFmt.sharegpt,
        "alpaca": DatasetFmt.alpaca,
        "raw_text": DatasetFmt.raw_text,
        "chat_completion": DatasetFmt.chat_completion,
        "prompt_answer": DatasetFmt.prompt_answer,
        "input_output": DatasetFmt.input_output,
        "gpt_style": DatasetFmt.gpt_style,
    }
    
    fn_name_lower = fn_name.lower().replace("-", "_")
    
    if fn_name_lower in mapping:
        return mapping[fn_name_lower]
    
    raise ValueError(f"Unknown reformatting function: {fn_name}")


def load_model_and_tokenizer(config: QuantizationRunConfig):
    """
    Load model and tokenizer from HuggingFace.
    
    Args:
        config: QuantizationRunConfig with model info
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {config.model.name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=config.model.trust_remote_code,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
    )
    
    return model, tokenizer


def run_quantization(
    config: QuantizationRunConfig,
    model,
    tokenizer,
    dataset,
    recipe,
    output_dir: str = "./outputs",
):
    """
    Run oneshot quantization.
    
    Args:
        config: QuantizationRunConfig
        model: Loaded model
        tokenizer: Loaded tokenizer
        dataset: Calibration dataset
        recipe: Initialized recipe modifier
        output_dir: Directory for outputs
    """
    max_seq = config.inference.max_seq_length
    num_samples = config.calibration.num_samples if config.calibration else dataset.num_rows
    
    print(f"Running oneshot quantization...")
    print(f"  Max sequence length: {max_seq}")
    print(f"  Calibration samples: {num_samples}")
    
    start_time = time.time()
    
    oneshot(
        model=model,
        recipe=recipe,
        dataset=dataset,
        max_seq_length=max_seq,
        num_calibration_samples=num_samples,
    )
    
    elapsed = time.time() - start_time
    print(f"Quantization completed in {elapsed:.2f}s")
    
    # Save model
    model_name = config.model.name.split("/")[-1]
    method = config.quantization.method.upper()
    output_name = f"{model_name}-{method}"
    output_path = Path(output_dir) / output_name
    
    print(f"Saving to: {output_path}")
    model.save_pretrained(output_path, save_compressed=True)
    tokenizer.save_pretrained(output_path)
    
    return output_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run calibration-based quantization with llm-compressor"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to quantization config YAML file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs",
        help="Output directory for quantized model"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Directory for dataset cache"
    )
    
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Ignore existing cache and regenerate"
    )
    
    parser.add_argument(
        "--recipe-arg",
        type=str,
        action="append",
        default=[],
        help="Override recipe arguments (key=value format)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Build launch command for logging
    launch_command = f"python {' '.join(sys.argv)}"
    
    # Load configuration
    print(f"Loading configuration: {args.config}")
    config = load_quantization_config(args.config)
    
    # Setup logging
    model_name = config.model.name.split("/")[-1]
    method = config.quantization.method.upper()
    
    with setup_logging(
        log_dir="./logs",
        model_name=model_name,
        method=method,
        launch_command=launch_command,
    ) as logger:
        
        logger.log_model_info(config.model.name, config.model.revision)
        logger.log_quantization_info(
            config.quantization.method,
            config.quantization.scheme
        )
        
        if config.calibration:
            logger.log_calibration_info(
                num_samples=config.calibration.num_samples,
                shuffle=config.calibration.shuffle,
                seed=config.calibration.seed,
            )
        
        # Log calibration set info
        if config.calibration_set:
            logger.log(f"Calibration set: {config.calibration_set}")
        elif config.calibration_sets:
            for cs in config.calibration_sets:
                ds_name = cs.dataset[0] if isinstance(cs.dataset, tuple) else cs.dataset
                logger.log(f"  Dataset: {ds_name}, samples: {cs.num_samples}")
        
        # Load recipe
        recipe_path = config.quantization.recipe
        logger.log(f"Loading recipe: {recipe_path}")
        recipe_data = load_recipe(recipe_path)
        log_recipe_summary(logger, recipe_data)
        
        # Create recipe modifier
        recipe = create_recipe_from_config(config, recipe_data)
        
        # Load model and tokenizer
        logger.log_step("Loading model")
        model, tokenizer = load_model_and_tokenizer(config)
        
        # Process calibration datasets
        logger.log_step("Processing calibration datasets")
        dataset = process_calibration_datasets(
            config,
            tokenizer,
            cache_dir=args.cache_dir,
            overwrite_cache=args.overwrite_cache,
        )
        
        logger.log(f"Total calibration samples: {dataset.num_rows}")
        
        # Run quantization
        logger.log_step("Running quantization")
        try:
            output_path = run_quantization(
                config,
                model,
                tokenizer,
                dataset,
                recipe,
                output_dir=args.output,
            )
            logger.log(f"SUCCESS: Model saved to {output_path}")
        except Exception as e:
            logger.log_exception(e, context="quantization")
            raise


if __name__ == "__main__":
    main()