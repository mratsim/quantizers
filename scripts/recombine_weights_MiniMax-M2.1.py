#!/usr/bin/env python3
"""
Recombine weights from FP8 and compressed-tensors models into a hybrid model.
This is specialized for MiniMax-M2.1

This script merges:
- Most layers: FP8 weights from the source model
- MoE expert layers: W4A16 pack-quantized weights from the compressed model
- Smoothing layers: Weights from the compressed model
- scale_inv: Renamed to weight_scale for certain layers
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

# Configuration
DEVICE = "cuda:0"
CHUNK_SIZE = 2  # Number of safetensor files to keep in memory at once


def natural_sort_key(s: str) -> list:
    """Sort strings with numbers in a human-friendly way."""
    return [int(t) if i & 1 else t.casefold() for i, t in enumerate(re.split(r"(\d+)", s))]


def is_moe_expert_weight(weight_name: str) -> bool:
    """Check if this weight is a MoE expert weight (w1, w2, or w3)."""
    return re.match(r".*block_sparse_moe\.experts\.\d+\.(w1|w2|w3)\.weight$", weight_name) is not None


def is_smoothing_layer(weight_name: str) -> bool:
    """Check if this is a smoothing layer (post_attention_layernorm)."""
    return "post_attention_layernorm" in weight_name and weight_name.endswith("weight")


def is_renamed_scale_inv(weight_name: str) -> bool:
    """Check if this is a scale_inv that should be renamed to weight_scale."""
    return weight_name.endswith("_proj.weight_scale_inv")


def is_scale_inv(weight_name: str) -> bool:
    """Check if this is a scale_inv parameter (any)."""
    return weight_name.endswith("_scale_inv")


def get_pack_quantized_param_names(base_weight_name: str) -> list:
    """Get the pack-quantized parameter names for a base weight name."""
    if not base_weight_name.endswith(".weight"):
        return []
    base = base_weight_name[:-7]  # Remove .weight suffix
    return [
        f"{base}.weight_packed",
        f"{base}.weight_scale",
        f"{base}.weight_shape",
        f"{base}.weight_zero_point",
        f"{base}.weight_g_idx",
    ]


def get_output_file_name(fp8_file: str) -> str:
    """Transform FP8 file name to output file name (e.g., 00130 -> 00125).
    This is specialized for MiniMax-M2.1
    """
    return fp8_file.replace("00130", "00125")


def create_mixed_precision_config(fp8_config_path: str, compressed_config_path: str) -> dict:
    """Create a mixed-precision compressed-tensors config combining FP8 and W4A16."""
    with open(fp8_config_path) as f:
        fp8_config = json.load(f)
    with open(compressed_config_path) as f:
        compressed_config = json.load(f)

    quantization_config = {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {  # FP8 for most layers
                "targets": ["Linear"],
                "weights": {
                    "type": "float",
                    "num_bits": 8,
                    "strategy": "block",
                    "block_structure": [128, 128],
                    "symmetric": True,
                    "dynamic": False,
                },
                "input_activations": {
                    "type": "float",
                    "num_bits": 8,
                    "strategy": "token",
                    "symmetric": True,
                    "dynamic": True,
                },
                "format": "float-quantized",
            },
            "group_1": {  # W4A16 for MoE experts
                "format": "pack-quantized",
                "input_activations": None,
                "output_activations": None,
                "targets": [
                    "Linear",
                    r"re:.*block_sparse_moe\.experts\.\d+\.(w1|w2|w3)$",
                ],
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": 32,
                    "num_bits": 4,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "group",
                    "symmetric": True,
                    "type": "int",
                },
            },
        },
        "ignore": compressed_config.get("quantization_config", {}).get("ignore", []),
        "kv_cache_scheme": None,
        "global_compression_ratio": None,
        "sparsity_config": {},
        "transform_config": {},
        "version": "0.13.1.dev0+g797d301.d20251228",
    }

    hybrid = fp8_config.copy()
    hybrid["quantization_config"] = quantization_config
    return hybrid


@dataclass
class MergeStats:
    """Statistics for the merge operation."""

    pack_quantized_replaced: int = 0
    smoothing_layers_replaced: int = 0
    scale_inv_copied: int = 0
    scale_inv_skipped: int = 0
    total_tensors: int = 0
    total_size: int = 0

    def log(self, dry_run: bool) -> None:
        """Log the statistics."""
        prefix = "[DRY-RUN] Would" if dry_run else "✓"
        action = "do" if dry_run else "done"
        print(f"{prefix} {action} the following:")
        print(f"  - Replaced {self.pack_quantized_replaced} tensors with pack-quantized versions")
        print(f"  - Replaced {self.smoothing_layers_replaced} smoothing layers")
        print(f"  - Skipped {self.scale_inv_skipped} scale_inv parameters")
        print(f"  - Copied and renamed {self.scale_inv_copied} scale_inv to weight_scale")
        print(f"  - Total tensors: {self.total_tensors}")


class ModelMerger:
    """
    Manages the merging of FP8 and compressed-tensors models.

    This class handles:
    - Loading and caching of model files
    - Weight processing and replacement
    - Output file generation
    """

    def __init__(
        self,
        fp8_path: str,
        compressed_path: str,
        output_path: str,
        dry_run: bool = False,
    ):
        self.fp8_path = Path(fp8_path)
        self.compressed_path = Path(compressed_path)
        self.output_path = Path(output_path)
        self.dry_run = dry_run
        self.stats = MergeStats()
        self.fp8_loaded: dict = {}
        self.compressed_loaded: dict = {}
        self.compressed_index: dict = {}
        self.compressed_weight_map: dict = {}

    def _load_index(self) -> dict:
        """Load the model index files for both models."""
        with open(self.fp8_path / "model.safetensors.index.json") as f:
            fp8_index = json.load(f)
        with open(self.compressed_path / "model.safetensors.index.json") as f:
            self.compressed_index = json.load(f)
        self.compressed_weight_map = self.compressed_index["weight_map"]
        return fp8_index

    def _get_output_file(self, fp8_file: str) -> Path:
        """Get the output file path for a given FP8 file."""
        return self.output_path / get_output_file_name(fp8_file)

    def _get_file_path(self, model_path: Path, file_name: str) -> Path:
        """Get the full file path for a safetensor file."""
        return model_path / file_name

    def _load_safetensor(self, model_path: Path, file_name: str, cache: dict) -> dict:
        """Load a safetensor file, using the cache if available."""
        if file_name in cache:
            return cache[file_name]
        file_path = self._get_file_path(model_path, file_name)
        state_dict = load_file(file_path, device=DEVICE)
        cache[file_name] = state_dict
        return state_dict

    def _evict_old_files(self, cache: dict) -> None:
        """Evict old files from the cache to limit memory usage."""
        if len(cache) > CHUNK_SIZE:
            oldest = next(iter(cache))
            del cache[oldest]
        torch.cuda.empty_cache()

    def _is_in_compressed(self, param_name: str) -> bool:
        """Check if a parameter exists in the compressed model."""
        return param_name in self.compressed_weight_map

    def _get_compressed_param(self, param_name: str) -> Optional[torch.Tensor]:
        """Get a parameter from the compressed model, or None if not found."""
        if not self._is_in_compressed(param_name):
            return None
        file_name = self.compressed_weight_map[param_name]
        if file_name not in self.compressed_loaded:
            self.compressed_loaded[file_name] = self._load_safetensor(self.compressed_path, file_name, self.compressed_loaded)
        return self.compressed_loaded[file_name][param_name]

    def _get_all_compressed_params(self, base_weight_name: str) -> dict:
        """Get all pack-quantized parameters for a base weight name."""
        result = {}
        for param_name in get_pack_quantized_param_names(base_weight_name):
            param = self._get_compressed_param(param_name)
            if param is not None:
                result[param_name] = param
        return result

    def _process_file(self, fp8_file: str) -> Optional[str]:
        """
        Process a single FP8 file and return the output file name.

        This is the main processing logic that:
        1. Skips _scale_inv parameters (renames or skips)
        2. Replaces MoE expert weights with pack-quantized versions
        3. Replaces smoothing layers
        4. Keeps all other weights from FP8
        """
        output_file = str(self._get_output_file(fp8_file))
        fp8_state_dict = self._load_safetensor(self.fp8_path, fp8_file, self.fp8_loaded)
        merged: dict = {}

        for weight_name, weight_tensor in fp8_state_dict.items():
            if is_renamed_scale_inv(weight_name):
                # Rename _scale_inv to _weight_scale for projection layers
                new_name = weight_name.replace("weight_scale_inv", "weight_scale")
                merged[new_name] = weight_tensor
                self.stats.scale_inv_copied += 1
            elif is_scale_inv(weight_name):
                # Skip other _scale_inv parameters
                self.stats.scale_inv_skipped += 1
            elif is_moe_expert_weight(weight_name):
                # Replace with all pack-quantized parameters
                pack_params = self._get_all_compressed_params(weight_name)
                if pack_params:
                    merged.update(pack_params)
                    self.stats.pack_quantized_replaced += 1
            elif is_smoothing_layer(weight_name):
                # Try to get from compressed, fall back to FP8
                compressed_tensor = self._get_compressed_param(weight_name)
                if compressed_tensor is not None:
                    merged[weight_name] = compressed_tensor
                    self.stats.smoothing_layers_replaced += 1
                else:
                    merged[weight_name] = weight_tensor
            else:
                # Keep from FP8
                merged[weight_name] = weight_tensor

        # Update stats for index metadata
        self.stats.total_tensors += len(merged)
        self.stats.total_size += sum(t.numel() * t.element_size() for t in merged.values())

        # Evict old files to save memory
        self._evict_old_files(self.fp8_loaded)
        self._evict_old_files(self.compressed_loaded)

        if not self.dry_run:
            save_file(merged, output_file)
        return output_file

    def _print_dry_run_info(self, fp8_index: dict) -> None:
        """Print debugging information for dry-run mode."""
        print("[DRY-RUN] Sample parameters in FP8 model:")
        for param in list(fp8_index["weight_map"].keys())[:20]:
            print(f"  {param}")

        print("\n[DRY-RUN] Sample parameters in compressed-tensors model:")
        for param in list(self.compressed_weight_map.keys())[:20]:
            print(f"  {param}")

        # Check for MoE expert weights
        moe_weights = [w for w in fp8_index["weight_map"] if is_moe_expert_weight(w)]
        print(f"\n[DRY-RUN] Found {len(moe_weights)} MoE expert weights in FP8 model")
        if not moe_weights:
            similar = [w for w in fp8_index["weight_map"] if "experts" in w and any(x in w for x in ("w1", "w2", "w3"))]
            print("[DRY-RUN] Found these expert-related weights:")
            for w in similar[:10]:
                print(f"  {w}")

    def run(self) -> None:
        """Run the full merge process."""
        if not self.dry_run:
            self.output_path.mkdir(parents=True, exist_ok=True)

        # Load model indices
        fp8_index = self._load_index()
        fp8_files = sorted(set(fp8_index["weight_map"].values()), key=natural_sort_key)

        if self.dry_run:
            self._print_dry_run_info(fp8_index)
            print(f"\n[DRY-RUN] Would process {len(fp8_files)} safetensor files")
        else:
            print(f"Processing {len(fp8_files)} safetensor files")

        # Process all files
        output_files = []
        for fp8_file in tqdm(fp8_files, desc="Merging files"):
            output_file = self._process_file(fp8_file)
            if output_file:
                output_files.append(output_file)

        # Create and save the model index
        if not self.dry_run:
            weight_map = {}
            for file in output_files:
                state = load_file(file, device="cpu")
                for name in state.keys():
                    weight_map[name] = Path(file).name

            model_index = {
                "metadata": {
                    "total_size": self.stats.total_size,
                    "format": "pt",
                },
                "weight_map": weight_map,
            }
            with open(self.output_path / "model.safetensors.index.json", "w") as f:
                json.dump(model_index, f, indent=2)

        # Create and save the config
        if not self.dry_run:
            config = create_mixed_precision_config(
                str(self.fp8_path / "config.json"),
                str(self.compressed_path / "config.json"),
            )
            with open(self.output_path / "config.json", "w") as f:
                json.dump(config, f, indent=2)

        # Log final summary
        if self.dry_run:
            print("\n[DRY-RUN] Summary of what would be done:")
            self.stats.log(dry_run=True)
        else:
            print(f"✓ Hybrid model saved to: {self.output_path}")
            self.stats.log(dry_run=False)


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Merge FP8 and compressed-tensors models into a hybrid model")
    parser.add_argument("--fp8-path", required=True, help="Path to the FP8 model")
    parser.add_argument("--compressed-path", required=True, help="Path to the compressed-tensors model")
    parser.add_argument("--output-path", required=True, help="Path for the output model")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    args = parser.parse_args()

    merger = ModelMerger(
        fp8_path=args.fp8_path,
        compressed_path=args.compressed_path,
        output_path=args.output_path,
        dry_run=args.dry_run,
    )
    merger.run()


if __name__ == "__main__":
    main()
