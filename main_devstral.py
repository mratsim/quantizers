# uv run --env-file=.env main_glm4.py\
# Note: This used to work in May 2025, since then llmcompressor
#       internals have significantly changed and calculate_offload_device_map has been removed

from typing import Dict, Union, Type
import os

import psutil, shutil

import torch
from accelerate import init_empty_weights, infer_auto_device_map
from transformers import AutoModelForCausalLM, AutoTokenizer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy
from huggingface_hub import hf_hub_download

from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.quantization.gptq import GPTQModifier
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.awq.mappings import AWQMapping
from llmcompressor import oneshot
from llmcompressor.transformers.compression.helpers import calculate_offload_device_map, hessian_memory_requirements, quantization_memory_requirement

CALIBRATION_DATASET="nvidia/OpenCodeInstruct"
DATASET_SPLIT="train"
SHUFFLE_SEED=42

MODEL_ID="mistralai/Devstral-Small-2505"
tekken_file = hf_hub_download(repo_id=MODEL_ID, filename="tekken.json")

# GPTQ
# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
# 2048 samples + 4096 seq length require about 107GB of CPU RAM
# and 8 hours for the GPTQ part on a RTX 5090
# they seem to also avoid hessian warnings
# NUM_CALIBRATION_SAMPLES = 2048
# MAX_SEQUENCE_LENGTH = 4096
# MODEL_OUT = MODEL_ID.split("/")[1] + ".w4a16-gptq"
# DAMPENING_FRAC=0.005

# AWQ
# See slides https://minjiazhang.github.io/courses/fall24-resource/slides/awq.pdf
# "AWQ: Calibration set"
# RAM usage grows linearly with calibration and sequence length, 128 and 2048 requires over 1TB.
# use swapfile if necessary, on a NVMe disk.
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 4096
MODEL_OUT = MODEL_ID.split("/")[1] + ".w4a16-awq"

def calculate_offload_device_map2(
    model_stub: str,
    reserve_for_hessians=False,
    num_gpus: int = 1,
    torch_dtype: torch.dtype = torch.float16,
    model_cls: Type = AutoModelForCausalLM,
    **model_kwargs,
) -> Dict[Union[int, str], Union[int, str]]:
    """
    Calculates the optimal gpu mappings for model_stub stored as torch_dtype. Takes
    into account extra memory required for quantization and (optionally) GPTQ hessians

    :param model_stub: local path or HF stub to calculate mapping for
    :param reserve_for_hessians: whether to reserve memory for GPTQ
    :param num_gpus: number of gpus to utilize
    :param model_cls: model class to use when initializing model structure,
        default is AutoModelForCausalLM
    :param model_kwargs: keyword arguments to pass to model initializer
    :return: memory mapping for layers of model_stub to be passed to from_pretrained()
    """
    max_cpu_memory = psutil.virtual_memory().available
    max_gpu_memory = torch.cuda.mem_get_info(0)[0]
    print(f'max_gpu_memory: {max_gpu_memory}')
    max_gpu_memory = int(0.90 * max_gpu_memory)
    print(f'new_max_gpu_memory: {max_gpu_memory}')
    available_gpus = torch.cuda.device_count()
    if available_gpus < num_gpus:
        raise ValueError(
            f"Requested {num_gpus} GPUs but only {available_gpus} are available."
        )
    max_gpu_memory = [max_gpu_memory] * num_gpus

    device_map = {}
    with init_empty_weights():
        dummy_model = model_cls.from_pretrained(
            model_stub, torch_dtype=torch_dtype, **model_kwargs
        )

        reserved_memory = 0
        if reserve_for_hessians:
            reserved_memory = hessian_memory_requirements(dummy_model)
        reserved_memory += quantization_memory_requirement(dummy_model)

        memory_limits = {
            idx: (max_memory - reserved_memory)
            for idx, max_memory in enumerate(max_gpu_memory)
        }
        memory_limits["cpu"] = max_cpu_memory

        device_map = infer_auto_device_map(
            dummy_model,
            max_memory=memory_limits,
            no_split_module_classes=dummy_model._no_split_modules,
        )
        del dummy_model

    return device_map

def get_calib_dataset(tokenizer):
    from datasets import load_dataset

    ds = load_dataset(
        CALIBRATION_DATASET,
        split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES*100}]",
    )
    
    print(f'Loaded "{CALIBRATION_DATASET}" dataset of size {len(ds)}')

    def preprocess(example):
        return {
                "input_ids": tokenizer.encode(
                    example["input"].strip(),
                    bos=True,
                    eos=True,
                )[:MAX_SEQUENCE_LENGTH]
        }

    ds = (
        ds.shuffle(seed=SHUFFLE_SEED)
        .map(preprocess, remove_columns=ds.column_names)
        # .filter(lambda example: len(example["input_ids"]) >= MAX_SEQUENCE_LENGTH)
        .select(range(NUM_CALIBRATION_SAMPLES))
    )
    
    print(f'Filtered "{CALIBRATION_DATASET}" dataset of size {len(ds)}')

    return ds

# The MistralTokenizer only has encode_chat_completion instead of "encode".
tekkenizer = MistralTokenizer.from_file(tekken_file)
tokenizer = tekkenizer.instruct_tokenizer.tokenizer
tokenizer.special_token_policy = SpecialTokenPolicy.IGNORE

device_map = calculate_offload_device_map2(MODEL_ID, reserve_for_hessians=True, num_gpus=1, torch_dtype=torch.bfloat16)
# with init_empty_weights():
#     model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
#     device_map = infer_auto_device_map(
#             model,
#             max_memory={0: "27GiB"},
#             no_split_module_classes=model._no_split_modules)
#     del model

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map=device_map, torch_dtype=torch.bfloat16,
    trust_remote_code=True)

ignore_layers = ["lm_head"]

print(model)

# AWQ
# recipe = [
#     AWQModifier(
#         # Read input->output from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/glm4/modeling_glm4.py
#         # which are somewhat easier than vllm ones as it's all in a single file
#         mappings=[
#             AWQMapping("re:.*input_layernorm", ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"]),
#             AWQMapping("re:.*v_proj", ["re:.*o_proj"]),
#             AWQMapping("re:.*post_attention_layernorm", ["re:.*gate_proj", "re:.*up_proj"]),
#             AWQMapping("re:.*up_proj", ["re:.*down_proj"]),
#         ],
#         ignore=ignore_layers,
#         config_groups={
#             "group_0": QuantizationScheme(
#                 targets=["Linear"],
#                 weights=QuantizationArgs(
#                     num_bits=4,
#                     type=QuantizationType.INT,
#                     dynamic=False,
#                     symmetric=False,
#                     strategy=QuantizationStrategy.GROUP,
#                     group_size=128,
#                 ),
#             ),
#         },
#     )
# ]

# GPTQ
recipe = [
    GPTQModifier(
        dampening_frac=DAMPENING_FRAC,
        ignore=ignore_layers,
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=4,
                    type=QuantizationType.INT,
                    dynamic=False,
                    symmetric=False,
                    strategy=QuantizationStrategy.GROUP,
                    group_size=128,
                ),
            ),
        },
    ),
]

# W4A8 
# recipe = [
#         GPTQModifier(
#             ignore=ignore_layers,
#             scheme="W4A8",
#             kv_cache_scheme=QuantizationArgs(
#                 num_bits=8,
#                 type=QuantizationType.FLOAT,
#                 strategy=QuantizationStrategy.TENSOR,
#                 dynamic=False,
#                 symmetric=True,
#             )
#         )
# ]

oneshot(
    model=model,
    tokenizer=tekkenizer,
    # dataset=CALIBRATION_DATASET,
    dataset=get_calib_dataset(tokenizer=tokenizer),
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    output_dir=MODEL_OUT,
)

# Save quantized model
model.save_pretrained(MODEL_OUT)

# Save tokenizer file
if not os.path.exists(MODEL_OUT):
    os.makedirs(MODEL_OUT)
dst_tekken_file = os.path.join(MODEL_OUT, "tekken.json")
shutil.copyfile(tekken_file, dst_tekken_file)

print(f'SUCCESS: files saved in {MODEL_OUT}')