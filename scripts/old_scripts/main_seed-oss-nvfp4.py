from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

MODEL_ID = "ByteDance-Seed/Seed-OSS-36B-Instruct"
MODEL_OUT = MODEL_ID.split("/")[1] + "-NVFP4"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model.generation_config.do_sample=True

CALIBRATION_DATASET="mit-han-lab/pile-val-backup"
DATASET_SPLIT="validation"
SHUFFLE_SEED=42

# Select number of samples
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 4096

# Load dataset and preprocess.

def get_calib_dataset(tokenizer):
    from datasets import load_dataset

    ds = load_dataset(
        CALIBRATION_DATASET,
        split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES*100}]",
    )

    print(f'Loaded "{CALIBRATION_DATASET}" dataset of size {len(ds)}')

    def preprocess(example):
        return {
                "input_ids": tokenizer.encode(example["text"].strip())[:MAX_SEQUENCE_LENGTH]
        }

    ds = (
        ds.shuffle(seed=SHUFFLE_SEED)
        .map(preprocess, remove_columns=ds.column_names)
        # .filter(lambda example: len(example["input_ids"]) >= MAX_SEQUENCE_LENGTH)
        .select(range(NUM_CALIBRATION_SAMPLES))
    )

    print(f'Filtered "{CALIBRATION_DATASET}" dataset of size {len(ds)}')

    return ds

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp4 with per group 16 via ptq
#   * calibrate a global_scale for activations, which will be used to
#       quantize activations to fp4 on the fly
recipe = QuantizationModifier(
    targets="Linear", scheme="NVFP4", ignore=["lm_head"]
)

# Apply quantization.
# We see `calibrate_moe_context` to True to update all `Qwen3MoeSparseMoeBlock`
# during calibration.
# Feel free to update the definition under
# llm-compressor/src/llmcompressor/modeling/qwen3_moe.py` to play around with
# this behaviour and evaluate its impact on quantization performance
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
oneshot(
    # pipeline="basic",
    model=model,
    tokenizer=tokenizer,
    dataset=get_calib_dataset(tokenizer=tokenizer),
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    calibrate_moe_context=True,
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
