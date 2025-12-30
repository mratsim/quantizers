# uv run --env-file=.env main_glm4.py
# Note: This used to work in May 2025, since then llmcompressor
#       internals have significantly changed and calculate_offload_device_map has been removed

from transformers import AutoModelForCausalLM, AutoTokenizer
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
from llmcompressor.transformers.compression.helpers import calculate_offload_device_map

CALIBRATION_DATASET="mit-han-lab/pile-val-backup"
DATASET_SPLIT="validation"
SHUFFLE_SEED=42

MODEL_ID="THUDM/GLM-4-32B-0414"

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
MAX_SEQUENCE_LENGTH = 2048
MODEL_OUT = MODEL_ID.split("/")[1] + ".w4a16-awq"

def get_calib_dataset(tokenizer):
    from datasets import load_dataset

    ds = load_dataset(
        CALIBRATION_DATASET,
        split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES*100}]",
    )

    def preprocess(example):
        return {
                "input_ids": tokenizer.encode(example["text"].strip())[:MAX_SEQUENCE_LENGTH]
        }

    ds = (
        ds.shuffle(seed=SHUFFLE_SEED)
        .map(preprocess, remove_columns=ds.column_names)
        .filter(lambda example: len(example["input_ids"]) >= MAX_SEQUENCE_LENGTH)
        .select(range(NUM_CALIBRATION_SAMPLES))
    )

    return ds

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

device_map = calculate_offload_device_map(MODEL_ID, reserve_for_hessians=True, num_gpus=1)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map=device_map, torch_dtype="auto",
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
#             AWQMapping("re:.*post_attention_layernorm", ["re:.*gate_up_proj"]),
#             AWQMapping("re:.*gate_up_proj", ["re:.*down_proj"]),
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
    tokenizer=tokenizer,
    # dataset=CALIBRATION_DATASET,
    dataset=get_calib_dataset(tokenizer=tokenizer),
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    save_compressed=True,
    trust_remote_code_model=True,
    output_dir=MODEL_OUT,
)

