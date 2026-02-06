# Quantization tips & tricks

This document accumulates knowledge to improve quantization quality.
Ideally that knowledge should be sourced.

## Quantization speed

Quantization can be accelerated by using Tensor Cores, see https://docs.pytorch.org/docs/stable/notes/cuda.html
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

The tradeoff is 7x faster speed but 56x larger relative error.
For quantization, a one-time cost we prefer not to use tensor cores.

## Layers to quantize

Quantization should be focused on Linear layer (also called Dense or Fully-Connected layers i.e. MatMul+Bias)
In particular quantizing LayerNorm/RMSnorm layer is strongly discouraged, see [1]
> LayerNorm in Quantization. Kovaleva et al. (2021); Wei et al. (2022) find that outliers in the
> LayerNorm parameters of BERT (Devlin et al., 2019) cause difficulties in model compression.
> Given the importance of LayerNorm, all the quantization methods we discuss above leave LayerNorm unquantized.

This is also reported in Intel and Nvidia repo:
- https://github.com/intel/neural-compressor/issues/1963#issuecomment-2274873441
- https://github.com/NVIDIA/TensorRT/issues/4084#issuecomment-2294513950

In LLM compressor this can be handled by making sure the `targets` of Quantization modifier is `Linear`.
Note that for MoE, the Linear part of individual experts may be masked and they would be left unquantized.

## Tensors to up-quantize

If there is enough bits, down projections should be prioritized.

According to [4]
> Fig. 3: Maximum absolute value over layers for a LLaMA3-8B.
> Each color represent a different projection and we clearly see that down_proj has the biggest
> spikes in input and output. We also observe that RMSNorm propagate spikes through the entire model

According to [5]
> Figure 5(a) illustrates the extremal ratio across layers and modules in LLaMA2-7B, highlighting
> that weight outliers are concentrated in the down-projection matrices Wdown
> ℓ of the second layer and
> the last two layers. Figures 5(b) and 5(c) provide detailed visualizations of these outliers in the last
> two layers.

## Mixture-of-Experts quantization (MoE)

Mixture-of-Experts require specific quantization techniques.

### Mixed-precision quantization

Some layers have a higher impact on LLM performance.
According to [2], spending more bits in attention layers results in large gain compared to spending them in FFN layers.
According to [3] on 2-bit quantization:
- quantizing expert FFN layers do not seriously impact model quality
- quantizing cross-attention has some impact
- quantizing self-attention has a large impact
- quantizing dense FFN has a very significant impact

Hence to preserve model quality we choose not to quantize dense FFN layers (i.e. shared experts) and self-attention layers.

We notice that:
- official MXFP4 weights of gpt-oss-120b from OpenAI keep self-attention in BF16:
  - https://huggingface.co/openai/gpt-oss-120b/blob/main/model.safetensors.index.json
- NVFP4 weights of DeepSeek-R1 quantized by Nvidia also keep self-attention in BF16:
  - https://huggingface.co/nvidia/DeepSeek-R1-0528-FP4/blob/main/model.safetensors.index.json

### Layers with high-impact

According to [2], giving more bits to the first `k` blocks have a significantly higher impact on model quality than for the same last `k` blocks.

### Non-Linear layers

Experts layers might not be stored as a `Linear` layer, meaning they might be skipped if using `llmcompressor` with a `Linear` target.
This requires adding the proper layer type as a target, or implementing a per LLM-architecture patch to `llmcompressor`.

### Activation quantization

When quantizing MoE, quantizing activations is tricky as only a subset of experts are activated per request. This can be alleviated the following ways:
- Not quantizing activation (i.e. NVFP4A16 vs NVFP4)
- Forcing experts to be activated, this requires implementing a per LLM-architecture patch to the quantizing library
  - https://github.com/vllm-project/llm-compressor/blob/0.8.1/examples/quantization_w4a4_fp4/README.md#quantizing-moes
  - https://github.com/vllm-project/llm-compressor/tree/0.8.1/src/llmcompressor/modeling
- While NVFP4 requires few samples for calibration (20 in LLM compressor example)
  to ensure all experts see a large enough number of samples we need a large and varied dataset.

### Expert quantization

When quantizing MoE, quantizing activations is tricky as only a subset of experts are activated per request. You have to make sure all experts are calibrated.

<details>
<summary>Visual showcase of why ensuring quantization of all MoE experts is important</summary>

- Source: https://avtc.github.io/aquarium-side-by-side/
- Context: https://github.com/ModelCloud/GPTQModel/pull/2235

![image](https://cdn-uploads.huggingface.co/production/uploads/67f26fd2c7b14380431d1f5a/BDc3-0m3_WLl3ZmbBMhmd.png)

</details>

## References

1. Why Do Some Inputs Break Low-Bit LLM Quantization? (2025)\
  Ting-Yun Chang, Muru Zhang, Jesse Thomason, Robin Jia\
  https://arxiv.org/pdf/2506.12044

2. Examining Post-Training Quantization for Mixture-of-Experts: A Benchmark (2024)\
  Pingzhi Li, Xiaolong Jin, Yu Cheng, Tianlong Chen\
  https://arxiv.org/pdf/2406.08155v1

3. Mixture of Quantized Experts (MoQE): Complementary Effect of Low-bit Quantization and Robustness (2023)\
  Young Jin Kim, Raffy Fahim, Hany Hassan Awadalla\
  https://arxiv.org/pdf/2310.02410


4. Precision Where It Matters: A Novel Spike\
   Aware Mixed-Precision Quantization Strategy for\
   LLaMA-based Language Models (2025)\
   Lucas Maisonnave, Cyril Moineau, Olivier Bichler, and Fabrice Rastello\
   https://arxiv.org/pdf/2504.21553

5. Systematic Outliers in Large Language Models (2025)\
   Yongqi An, Xu Zhao, Tao Yu, Ming Tang, Jinqiao Wang\
   https://arxiv.org/pdf/2502.06415v2