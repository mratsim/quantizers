from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.utils import dispatch_for_generation
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)

CALIBRATION_DATASET="HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT="train_sft"
SHUFFLE_SEED=42
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 4096

MODEL_ID = "ByteDance-Seed/Seed-OSS-36B-Instruct"
MODEL_OUT = MODEL_ID.split("/")[1] + "-FP8-KV"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model.generation_config.do_sample=True

# Dataset processing
ds = load_dataset(CALIBRATION_DATASET, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

def process_and_tokenize(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(text, padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)

ds = ds.map(process_and_tokenize, remove_columns=ds.column_names)

recipe = [
    QuantizationModifier(
        ignore=["lm_head"],
        # DeepSeek V3 style block quantization + dynamic per token quantization
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.FLOAT,
                    dynamic=False,
                    symmetric=True,
                    strategy=QuantizationStrategy.BLOCK,
                    block_structure=[128, 128],
                ),
                input_activations=QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.FLOAT,
                    strategy=QuantizationStrategy.GROUP,
                    symmetric=True,
                    dynamic=True,
                    observer=None,
                    group_size=128,
                ),
            ),
        },
        kv_cache_scheme=QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            dynamic=False,
            symmetric=True,
            strategy=QuantizationStrategy.TENSOR,
        ),
    )
]

oneshot(
    # pipeline="basic",
    model=model,
    recipe=recipe,
    dataset=ds,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save to disk in compressed-tensors format.
model.save_pretrained(MODEL_OUT, save_compressed=True)
tokenizer.save_pretrained(MODEL_OUT)
print(f'SUCCESS: files saved in {MODEL_OUT}')

# Testing
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")
