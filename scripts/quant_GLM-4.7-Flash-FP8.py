import os

from llmcompressor import model_free_ptq

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")

MODEL_ID = "zai-org/GLM-4.7-Flash"
MODEL_OUT = MODEL_ID.split("/")[1] + "-FP8"

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=MODEL_OUT,
    scheme="FP8_BLOCK",
    ignore=[
        "lm_head",
        "re:.*mlp\\.gate$",  # MoE router
        "re:.*kv_a_proj_with_mqa$",
        "re:.*q_a_proj$",
        "model.embed_tokens",
    ],
    max_workers=16,
    device="cuda:0",
)

print(f"SUCCESS: files saved in {MODEL_OUT}")
