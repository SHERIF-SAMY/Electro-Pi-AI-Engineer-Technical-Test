# Section 3 — Quantization Benchmark

## Overview
This repository contains a complete benchmark suite for evaluating the impact of quantization on Large Language Models. The benchmark compares two variants of the `qwen2.5:0.5b` model served via Ollama to measure the trade-offs between model size, memory footprint, inference throughput, and output quality.

## Quantization Approach
The default `qwen2.5:0.5b` model on Ollama is already quantized using the GGUF `Q4_K_M` (4-bit) format, as Ollama does not natively distribute true FP16 or FP32 unquantized models. To demonstrate a valid and realistic quantization trade-off in an Ollama-only environment, this benchmark compares:
- **Baseline (High Quality):** `qwen2.5:0.5b` (Q4_K_M, 4-bit)
- **Quantized (Aggressive):** `qwen2.5:0.5b-instruct-q2_k` (Q2_K, 2-bit)

This comparison clearly illustrates the performance and quality impacts of moving to a lower bit-width.

## Project Structure
```text
section3_quantization/
├── models/                  # Reserved for manual GGUF downloads
├── scripts/                 # Source code for the benchmark
│   ├── benchmark.py         # Main benchmark execution script
│   ├── metrics.py           # RAM, latency, and throughput tracking utilities
│   ├── quality_eval.py      # ROUGE-L quality evaluation
│   └── generate_report.py   # Renders the final comparison table
├── benchmarks/
│   └── raw_results/         # JSON outputs from benchmark runs
├── prompts/
│   └── prompts.json         # 5 canonical prompts covering reasoning, coding, QA
├── results/
│   ├── comparison_table.md  # Final generated metrics table
│   └── writeup.md           # Discussion on quantization techniques
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Environment Setup
1. **Requires Python 3.10+**. Create a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/Mac:
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install [Ollama](https://ollama.com/) and pull the models:
   ```bash
   ollama pull qwen2.5:0.5b
   ollama pull qwen2.5:0.5b-instruct-q2_k
   ```

## Running the Benchmark
1. Ensure the Ollama server is running (`ollama serve` or via the desktop app).
2. Execute the benchmark script:
   ```bash
   python scripts/benchmark.py --baseline qwen2.5:0.5b --quantized qwen2.5:0.5b-instruct-q2_k
   ```
   *This will run inference for both models on all 5 prompts and save the raw JSON results in `benchmarks/raw_results/`.*
3. Generate the markdown comparison table:
   ```bash
   python scripts/generate_report.py
   ```
   *The table will be saved to `results/comparison_table.md`.*

## Limitations
- **"Full Precision" vs Quantized:** As noted, true FP32 is not tested here. The comparison uses Q4_K_M as the high-quality baseline proxy.
- **CPU Benchmarking:** If run without a GPU, memory metrics capture system RAM, and throughput may have higher variance due to OS background tasks.
- **Memory Tracking:** The `psutil` tracker polls the `ollama` server process to estimate memory overhead. Peak RAM reported represents the overall server footprint increase during generation.

## References
- [GPTQ Paper](https://arxiv.org/abs/2210.17323)
- [AWQ Paper](https://arxiv.org/abs/2306.00978)
- [bitsandbytes Repository](https://github.com/TimDettmers/bitsandbytes)
- [llama.cpp (GGUF format)](https://github.com/ggerganov/llama.cpp)
