# Quantization Strategy Selection for Production LLM Deployment

## bitsandbytes vs GPTQ
`bitsandbytes` (BnB) performs **post-training, runtime quantization** (PTQ) via `load_in_4bit` / `load_in_8bit`. It requires no calibration dataset and is very fast to apply, but it is tightly coupled to HuggingFace Transformers and CUDA, making it **CPU-incompatible**. Conversely, `GPTQ` performs **offline, one-shot weight quantization** using second-order Hessian information and a small calibration dataset. It produces a static quantized checkpoint. While GPTQ requires an offline calibration step (taking hours for large models), it achieves higher quality at equivalent bit-widths compared to BnB. GPTQ is preferred when you can afford calibration and need maximum throughput at serving, whereas BnB is better for rapid experimentation on GPU stacks.

## AWQ vs GPTQ
`AWQ` (Activation-aware Weight Quantization) identifies and protects the **1% of salient weights** most sensitive to quantization, achieving better quality than GPTQ at the same bit-width (especially at 4-bit). AWQ quantization is faster to compute since it doesn't require the full Hessian. While GPTQ is widely supported across serving frameworks (e.g., vLLM, TGI), AWQ support is growing rapidly and becoming the standard for modern high-performance serving. In short, AWQ is preferred for quality-critical 4-bit deployments on modern GPUs, while GPTQ offers broader legacy compatibility.

## GGUF vs bitsandbytes
`GGUF` (the llama.cpp format) supports both **CPU and GPU inference**, making it the only practical option for CPU-only machines or heterogeneous hardware. It packages multiple quantization schemes (Q2_K through Q8_0) into a single, highly portable ecosystem that runs on macOS (Metal), Windows, Linux, and edge devices without framework dependencies. `bitsandbytes`, on the other hand, is **GPU-exclusive**, requires CUDA, and lives entirely within the HuggingFace Python stack. GGUF is chosen for portability and CPU compatibility, whereas BnB is strictly for HuggingFace-based GPU deployments.

## Production Recommendation
For this assignment's hardware constraints (CPU-capable, local, no cloud), **GGUF + Ollama / llama.cpp** is the clear winner—it has zero GPU dependency, offers simple deployment, and provides proven throughput. For a GPU-equipped production stack with strict quality SLAs, **AWQ at 4-bit** served on `vLLM` provides the best throughput-to-quality ratio. For rapid HuggingFace-based prototyping with a GPU, **bitsandbytes** (`load_in_4bit`) is the best choice for zero-friction quantization.
