import torch
import pandas as pd

from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)

SHUFFLE_SEED=42
MAX_SEQUENCE_LENGTH = 8192

MODEL_ID = "ByteDance-Seed/Seed-OSS-36B-Instruct"
MODEL_OUT = MODEL_ID.split("/")[1] + "-FP8-KV8"

# Datasets config
# ================================================================
def gpt_to_chat_completion_format(data):
    chat_completion = []
    role_mapping = {
        "system": "system",
        "human": "user",
        "gpt": "assistant"
    }
    for entry in data:
        role = role_mapping.get(entry["from"], "user")  # default to "user" if unknown
        chat_completion.append({
            "role": role,
            "content": entry["value"]
        })
    return chat_completion

def input_output_to_chat_completion_format(input_col, output_col, row):
    chat_completion = []
    chat_completion.append({
        "role": "user",
        "content": row[input_col]
    })
    chat_completion.append({
        "role": "assistant",
        "content": row[output_col]
    })
    return chat_completion

COL_NAMES = ["dataset", "split", "num_samples", "column", "reformatting_fn"]
DATASETS = [
    (["neuralmagic/calibration", "LLM"], "train", 256, "messages", None),
    ("HuggingFaceH4/ultrachat_200k", "train_sft", 256, "messages", None),
    ("nvidia/OpenCodeInstruct", "train", 16, None, lambda entry: input_output_to_chat_completion_format(
        "input", "output", entry)),
    ("CSJianYang/CodeArena", "test", 32, "messages", None),
    ("nvidia/OpenScienceReasoning-2", "train", 16, None, lambda entry: input_output_to_chat_completion_format(
        "input", "output", entry)),
    ("MegaScience/MegaScience", "train", 16, None, lambda entry: input_output_to_chat_completion_format(
        "question", "answer", entry)),
    ("Gryphe/Opus-WritingPrompts", "train", 32, "conversations", gpt_to_chat_completion_format),
    (["ServiceNow-AI/M2Lingual", "full_data"], "train", 256, "conversation", None),
    ("anthracite-org/stheno-filtered-v1.1", "train", 32, "conversations", gpt_to_chat_completion_format),
    ("zerofata/Roleplay-Anime-Characters", "train", 16, "messages", None),
    ("zerofata/Instruct-Anime", "train", 16, "messages", None),
    ("zerofata/Instruct-Anime-CreativeWriting", "train", 16, "messages", None),
    ("sam-paech/gutenberg3-generalfiction-scifi-fantasy-romance-adventure-dpo", "train", 16, "chosen", None),
    # ("nvidia/OpenMathReasoning", "cot", 16, None, lambda entry: input_output_to_chat_completion_format(
    #     "problem", "generated_solution", entry)),
    ("nvidia/OpenMathInstruct-2", "train", 16, None, lambda entry: input_output_to_chat_completion_format(
        "problem", "generated_solution", entry)),
    ("fka/awesome-chatgpt-prompts", "train", 203, "prompt", lambda entry: [{"role": "user", "content": entry}]),
    ("databricks/databricks-dolly-15k", "train", 256, None, lambda entry: input_output_to_chat_completion_format(
        "instruction", "response", entry)),
    ("FreedomIntelligence/SocraticChat", "train", 32, "conversations", gpt_to_chat_completion_format),
    ("ruggsea/stanford-encyclopedia-of-philosophy_instruct", "train", 32, None, lambda entry: input_output_to_chat_completion_format(
        "question", "answer", entry)),
    ("mlfoundations-dev/stackexchange_philosophy", "train", 32, "conversations", gpt_to_chat_completion_format),
    ("theoldmandthesea/17k_business_book", "train", 64, None, lambda entry: input_output_to_chat_completion_format(
        "question", "answer", entry)),
    ("anthracite-org/nopm_claude_writing_fixed", "train", 32, "conversations", gpt_to_chat_completion_format),
    # ("nvidia/Llama-Nemotron-Post-Training-Dataset", "chat", 256, "input", None),
    # ("nvidia/Llama-Nemotron-Post-Training-Dataset", "code", 16, "input", None),
    # ("nvidia/Llama-Nemotron-Post-Training-Dataset", "math", 16, "input", None),
    # ("nvidia/Llama-Nemotron-Post-Training-Dataset", "science", 16, "input", None),
    ("PJMixers/grimulkan_physical-reasoning-ShareGPT", "train", 16, "conversations", gpt_to_chat_completion_format),
    ("PJMixers/grimulkan_theory-of-mind-ShareGPT", "train", 16, "conversations", gpt_to_chat_completion_format),
    ("HuggingFaceH4/no_robots", "train", 16, "messages", None),
    ("nvidia/HelpSteer", "train", 16, None, lambda entry: input_output_to_chat_completion_format(
        "prompt", "response", entry)),
    ("garage-bAInd/Open-Platypus", "train", 16, None, lambda entry: input_output_to_chat_completion_format(
        "instruction", "output", entry)),
    ("AquaV/US-Army-Survival-Sharegpt", "train", 8, "conversations", gpt_to_chat_completion_format),
    ("AquaV/Interrogation-Sharegpt", "train", 8, "conversations", gpt_to_chat_completion_format),
    ("AquaV/Multi-Environment-Operations-Sharegpt", "train", 8, "conversations", gpt_to_chat_completion_format),
    ("AquaV/Resistance-Sharegpt", "train", 8, "conversations", gpt_to_chat_completion_format),
    ("PocketDoc/Dans-Kinomaxx-VanillaBackrooms", "train", 8, "conversations", gpt_to_chat_completion_format),
    ("PocketDoc/Dans-Prosemaxx-Adventure", "train", 8, "conversations", gpt_to_chat_completion_format),
]

DATASETS=pd.DataFrame(dict(zip(COL_NAMES, zip(*DATASETS))))
NUM_CALIBRATION_SAMPLES=DATASETS['num_samples'].sum()
print(f'DATASETS:\n{DATASETS}')

# Load model.
# ================================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    # device_map="sequential",
    # max_memory={0: "85GiB", 1: "85GiB", "cpu": "100GiB"},
    offload_folder = "./offload/",
    offload_state_dict=False,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model.generation_config.do_sample=True

# Load datasets
# ================================================================
def process_and_tokenize(row, col, reformatting_fn):
    if reformatting_fn and col:
        text = tokenizer.apply_chat_template(reformatting_fn(row[col]), tokenize=False)
    elif reformatting_fn:
        text = tokenizer.apply_chat_template(reformatting_fn(row), tokenize=False)
    else:
        text = tokenizer.apply_chat_template(row[col], tokenize=False)
    return tokenizer(text, padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)

dss = []
for i in range(DATASETS.shape[0]):
    ds_desc = DATASETS.iloc[i]
    print(f'DATASET: {ds_desc}')
    # Some datasets have multiple config like neuralmagic/calibration with LLM and VLM
    if isinstance(ds_desc["dataset"], str):
        ds = load_dataset(ds_desc["dataset"], split=ds_desc["split"])
    else:
        ds = load_dataset(*ds_desc["dataset"], split=ds_desc["split"])
    ds = ds.shuffle(seed=SHUFFLE_SEED)
    ds = ds.filter(lambda e, i: i<ds_desc["num_samples"], with_indices=True)
    ds = ds.map(
            lambda row: process_and_tokenize(row, ds_desc["column"], ds_desc["reformatting_fn"]),
            remove_columns=ds.column_names)
    dss.append(ds)
    del(ds)

dss = concatenate_datasets(dss)

# Visual inspection
dss.to_csv('dataset-calibration-' + MODEL_OUT + '.csv')
# acc = 0
# for i in range(DATASETS.shape[0]):
#     ds_desc = DATASETS.iloc[i]
#     print(f'DATASET: {ds_desc} - at row {acc}')
#     print("---------------------")
#     print(tokenizer.decode(dss[int(acc)]['input_ids']))
#     print("=====================")
#     acc += ds_desc["num_samples"]

# Randomize order
dss = dss.shuffle(seed=SHUFFLE_SEED)

# Quantization recipe
# ================================================================
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
    dataset=dss,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save to disk in compressed-tensors format.
model.save_pretrained(MODEL_OUT, save_compressed=True)
tokenizer.save_pretrained(MODEL_OUT)
print(f'SUCCESS: files saved in {MODEL_OUT}')
