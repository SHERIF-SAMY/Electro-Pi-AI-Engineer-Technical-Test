# Section 3 — Quantization: Implementation Plan

> **Scope**: Quantize `qwen2.5:0.5b` via Ollama, benchmark full-precision vs quantized variants, produce a comparison table and a half-page write-up.
> **Target**: Mid-Level AI Engineer technical assessment — completable in one working day.

---

## Open Questions

> [!IMPORTANT]
> **Design Decision — "Full Precision" Baseline**: Ollama distributes `qwen2.5:0.5b` as a GGUF file, which is already quantized. The raw `qwen2.5:0.5b` tag uses **Q4_K_M** by default (~397 MB). True FP32/FP16 is not available through Ollama.
>
> **Recommended resolution**: Use two *different* GGUF quantization levels that *are* available on Ollama as the "precision comparison":
> - **Baseline** → `qwen2.5:0.5b` (Q4_K_M, 397 MB) — treated as the "full quality reference"
> - **Quantized** → `qwen2.5:0.5b-instruct-q2_k` or `qwen2.5:0.5b-q8_0` — treated as the quantized variant
>
> This is the *only correct, non-invented workflow* for an Ollama-only setup. The plan will clearly document this constraint in the write-up.

---

## Phase 1 — Project Structure

```
section3_quantization/
│
├── models/                  # (empty) reserved for any local model artifacts
├── scripts/
│   ├── benchmark.py         # Core benchmark runner (both models)
│   ├── metrics.py           # Memory, latency, throughput measurement helpers
│   ├── quality_eval.py      # Output quality scoring (BERTScore / rouge)
│   └── generate_report.py   # Build the markdown comparison table + write-up
├── benchmarks/
│   └── raw_results/         # JSON outputs from each benchmark run
├── prompts/
│   └── prompts.json         # The five canonical prompts (versioned, immutable)
├── results/
│   ├── comparison_table.md  # Final rendered table
│   └── writeup.md           # Half-page quantization discussion
├── README.md
└── requirements.txt
```

### Folder Purpose

| Folder / File | Purpose |
|---|---|
| `models/` | Placeholder for any manually downloaded GGUF files; unused in Ollama-only flow but keeps the structure extensible |
| `scripts/benchmark.py` | Orchestrates the full benchmark: loads prompts, calls Ollama API, records timings, delegates to `metrics.py` |
| `scripts/metrics.py` | Isolated measurement utilities: RAM sampling, token counting, throughput calculation |
| `scripts/quality_eval.py` | Computes automated quality scores (ROUGE-L, optional BERTScore) against reference answers |
| `scripts/generate_report.py` | Reads raw JSON results and renders the Markdown comparison table and write-up skeleton |
| `benchmarks/raw_results/` | Immutable JSON dumps for reproducibility and audit trail |
| `prompts/prompts.json` | Single source of truth for the five prompts — both runs use the same file |
| `results/` | Final deliverables: table and write-up |
| `README.md` | Setup + reproduction instructions |
| `requirements.txt` | Pinned dependency list |

---

## Phase 2 — Environment Setup

### Python Version
- **Python 3.10+** (3.11 preferred for performance; avoid 3.12 due to occasional dependency compatibility gaps)

### Required Libraries

| Library | Version | Why Needed |
|---|---|---|
| `ollama` | ≥0.2.0 | Official Python client for the Ollama REST API — handles streaming, model management |
| `psutil` | ≥5.9.0 | Cross-platform RAM measurement (`Process.memory_info()`) |
| `rouge-score` | ≥0.1.2 | ROUGE-L quality metric (no GPU needed, pure Python) |
| `bert-score` | ≥0.3.13 | Optional — BERTScore for semantic similarity; CPU-capable but slow on large models |
| `pandas` | ≥2.0.0 | Tabular aggregation of benchmark results before rendering |
| `tabulate` | ≥0.9.0 | Renders pandas DataFrames as Markdown tables |
| `tqdm` | ≥4.66.0 | Progress bars during multi-prompt runs |
| `python-dotenv` | ≥1.0.0 | Environment variable management for Ollama host config |
| `pytest` | ≥8.0.0 | Optional — unit tests for metric utilities |

### Ollama Requirements

| Requirement | Detail |
|---|---|
| Ollama version | ≥0.1.40 (supports `/api/generate` streaming with timing fields) |
| Models to pull | `ollama pull qwen2.5:0.5b` and `ollama pull qwen2.5:0.5b-instruct-q2_k` |
| Host | Default `http://localhost:11434` |
| GPU | Optional — benchmark runs on CPU if no CUDA/Metal GPU detected |

### Installation Steps

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Pull both Ollama models
ollama pull qwen2.5:0.5b
ollama pull qwen2.5:0.5b-instruct-q2_k

# 4. Verify Ollama is running
ollama ps
```

> [!NOTE]
> `bert-score` is marked **optional** in `requirements.txt` because it downloads a RoBERTa checkpoint on first use (~500 MB). Comment it out for lightweight runs.

---

## Phase 3 — Baseline Model

### Model Selection
- **Baseline**: `qwen2.5:0.5b` (Ollama default tag = Q4_K_M GGUF, 397 MB)
- This is the "higher quality, larger" reference in the Ollama-constrained context.

### Loading / Inference Approach
Ollama exposes a REST API. The Python `ollama` client wraps it:

```python
import ollama
response = ollama.generate(model="qwen2.5:0.5b", prompt=prompt)
```

For streaming (required for token-level timing):
```python
stream = ollama.generate(model="qwen2.5:0.5b", prompt=prompt, stream=True)
```

### How to Measure Metrics

#### Memory (RAM)
- **Library**: `psutil`
- **Method**: Sample `psutil.Process(os.getpid()).memory_info().rss` before and after the Ollama call.
- **Limitation**: This captures the *client* process RSS, not Ollama server heap. To capture server-side memory, poll `psutil` on the `ollama` OS process by name, or use `ollama ps` output parsing.
- **Recommended**: PID-based polling of the `ollama` server process via `psutil.process_iter()`.

#### Latency
- **Method**: `time.perf_counter()` wall-clock delta from first token requested to last token received.
- Ollama also returns `eval_duration` (nanoseconds) in the non-streaming response — use this as the authoritative server-side latency.

#### Throughput (tokens/sec)
- Ollama response includes `eval_count` (number of output tokens) and `eval_duration` (nanoseconds).
- **Formula**: `throughput = eval_count / (eval_duration / 1e9)`
- This is the most reliable metric as it's server-reported, not subject to network/client overhead.

#### Output Quality
- Collect the full text response for each prompt.
- Compare against reference answers using **ROUGE-L** (mandatory) and optionally **BERTScore**.

---

## Phase 4 — Quantized Model

### Correct Approach for Ollama-Based Quantization Demo

> [!IMPORTANT]
> **Key architectural constraint**: Ollama manages its own model registry. You cannot inject arbitrary GGUF files into its serving pipeline via the standard `ollama run` flow without a Modelfile. The correct production-realistic approaches are:

#### Option A — Use a Different Ollama Tag (Recommended)
Pull a lower-bit GGUF variant that Ollama distributes for the same model family:

```bash
ollama pull qwen2.5:0.5b-instruct-q2_k
```

- `Q2_K` is a 2-bit quantization → significantly smaller, faster, lower quality.
- This creates a *real, valid, supported* quantization comparison.
- Both models are served by the identical Ollama runtime, making the comparison fair.

#### Option B — Custom GGUF via Modelfile (Advanced)
Download a GGUF directly from HuggingFace (e.g., `Qwen/Qwen2.5-0.5B-Instruct-GGUF`) and register it via an Ollama `Modelfile`:

```
FROM ./models/qwen2.5-0.5b-q8_0.gguf
```
```bash
ollama create qwen2.5-q8-custom -f ./models/Modelfile
ollama run qwen2.5-q8-custom
```

- This gives control over the exact quantization level (Q8_0 ≈ 8-bit, near FP16 quality).
- Recommended if the assessor requires demonstrating a "near full precision" vs "aggressive quantization" contrast.

#### Recommended Pairing for Maximum Demonstration Value
| Role | Model Tag | GGUF Type | Size |
|---|---|---|---|
| Baseline ("full precision") | `qwen2.5:0.5b` | Q4_K_M | ~397 MB |
| Quantized | `qwen2.5:0.5b-instruct-q2_k` | Q2_K | ~270 MB |

> The plan will explicitly state in the write-up that "full precision" is approximated within the Ollama ecosystem and explain what true FP16/FP32 would require (llama.cpp direct, bitsandbytes, etc.).

---

## Phase 5 — Benchmark Design

### The Five Canonical Prompts

All stored in `prompts/prompts.json` as a versioned, immutable array.

---

#### Prompt 1 — Reasoning
```
"Prompt": "A bat and a ball together cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost? Explain your reasoning step by step."
```
**Why**: The classic CRT (Cognitive Reflection Test) problem. Tests multi-step arithmetic reasoning and tendency to resist the intuitive-but-wrong answer ($0.10). Produces measurable divergence between model quality levels.

---

#### Prompt 2 — Coding
```
"Prompt": "Write a Python function that accepts a list of integers and returns the two numbers that sum closest to zero. Include docstring, type hints, and one usage example."
```
**Why**: Tests code generation correctness, structure, and adherence to Python conventions. Output can be automatically validated by running it. Quality difference between quantization levels is often visible in edge-case handling.

---

#### Prompt 3 — Summarization
```
"Prompt": "Summarize the following paragraph in exactly two sentences:\n\n'The transformer architecture, introduced in the 2017 paper Attention Is All You Need, replaced recurrent networks with self-attention mechanisms, enabling parallelization during training and dramatically improving performance on sequence-to-sequence tasks such as translation and text summarization.'"
```
**Why**: Tests information compression and instruction following (exact two-sentence constraint). Reference answer is easy to define, making ROUGE-L scoring meaningful and reliable.

---

#### Prompt 4 — Translation
```
"Prompt": "Translate the following sentence from English to French, then back to English, and note any meaning changes:\n\n'The early bird catches the worm, but the second mouse gets the cheese.'"
```
**Why**: Tests cross-lingual capability and semantic preservation. The idiom creates a natural challenge for lower-precision models. Round-trip translation also exposes subtle degradation.

---

#### Prompt 5 — Factual QA
```
"Prompt": "What is the difference between supervised learning and unsupervised learning? Give one real-world example of each."
```
**Why**: Tests factual recall and definitional clarity on a well-defined ML concept. Both domain relevance (appropriate for an AI engineer assessment) and easy qualitative scoring make this ideal.

---

### Prompt Configuration Schema (`prompts.json`)
```json
[
  {
    "id": 1,
    "category": "reasoning",
    "prompt": "...",
    "reference_answer": "..."
  },
  ...
]
```
Including reference answers enables automated ROUGE-L scoring.

---

## Phase 6 — Performance Metrics

### 1. Memory Footprint

| Approach | Method |
|---|---|
| Client RAM | `psutil.Process(pid).memory_info().rss` before/after call |
| **Server RAM (recommended)** | Find the `ollama` server PID via `psutil.process_iter(['name', 'pid'])`, poll `.memory_info().rss` every 500ms during inference |
| Peak memory | Track the maximum sampled value across the polling window |

**Reported value**: Peak RSS of Ollama server process during inference (MB).

---

### 2. Latency (Time-to-Last-Token)

- **Source**: Ollama response field `eval_duration` (nanoseconds, server-side)
- **Formula**: `latency_ms = eval_duration / 1_000_000`
- Additionally record `load_duration` (model load time) to distinguish cold vs warm inference.
- **Per-run**: Average over 5 prompts, report mean ± std dev.

---

### 3. Tokens/sec Throughput

- **Source**: Ollama response fields `eval_count` (output tokens) and `eval_duration` (ns)
- **Formula**: `tokens_per_sec = eval_count / (eval_duration / 1e9)`
- This is the definitive metric — server-side, not affected by network overhead.
- **Per-run**: Average over 5 prompts.

---

### 4. Output Quality

#### Primary — ROUGE-L (mandatory, CPU, fast)
- Compare model output against the `reference_answer` in `prompts.json`
- Report per-prompt ROUGE-L F1 and macro-average across 5 prompts
- Library: `rouge_score.rouge_scorer.RougeScorer(['rougeL'])`

#### Secondary — BERTScore (optional, CPU-capable, slower)
- Semantic similarity via contextual embeddings
- Library: `bert_score.score(candidates, references, lang='en')`
- Report F1 score averaged across 5 prompts

#### Tertiary — Qualitative Flag (human, mandatory)
- For each prompt, note any factual errors, hallucinations, or instruction violations
- Capture as a binary `quality_flag` (✓ / ✗) in the results JSON

---

## Phase 7 — Comparison Table

### Final Table Design

| Metric | Baseline (Q4_K_M) | Quantized (Q2_K) |
|---|---|---|
| **Precision** | Q4_K_M (4-bit) | Q2_K (2-bit) |
| **Model Size (disk)** | 397 MB | ~270 MB |
| **Peak RAM Usage** | _measured_ MB | _measured_ MB |
| **Avg Latency (ms)** | _measured_ | _measured_ |
| **Avg Throughput (tok/s)** | _measured_ | _measured_ |
| **ROUGE-L (avg F1)** | _measured_ | _measured_ |
| **BERTScore F1 (avg)** | _measured_ (optional) | _measured_ (optional) |
| **Quality Flags Passed** | _X/5_ | _X/5_ |

### How Each Value Is Obtained

| Column | Source |
|---|---|
| Precision | Ollama model tag / GGUF quantization type in metadata |
| Model Size | `ollama show <model> --verbose` → file size field; or `os.path.getsize()` if local GGUF |
| Peak RAM | `psutil` server process polling (Phase 6, Method 2) |
| Avg Latency | Mean of `eval_duration` across 5 prompts, converted to ms |
| Avg Throughput | Mean of `eval_count / (eval_duration/1e9)` across 5 prompts |
| ROUGE-L | `rouge_score` library against reference answers |
| BERTScore | `bert_score` library against reference answers |
| Quality Flags | Human review per prompt output |

---

## Phase 8 — Write-up Outline

### Structure (half-page ≈ 300–400 words)

**Title**: *Quantization Strategy Selection for Production LLM Deployment*

---

**§1 — bitsandbytes vs GPTQ** (≈80 words)

*Key points*:
- `bitsandbytes` (BnB) performs **post-training, runtime quantization** (PTQ) via `load_in_4bit` / `load_in_8bit`. No calibration dataset needed. Very fast to apply. Tightly coupled to HuggingFace Transformers + CUDA. **CPU-incompatible**.
- `GPTQ` performs **offline, one-shot weight quantization** using second-order Hessian information and a small calibration dataset. Produces a static quantized checkpoint. Requires a calibration step (~hours for large models) but achieves higher quality at equivalent bit-width vs BnB.
- **Decision axis**: GPTQ is preferred when you can afford an offline calibration step and need maximum throughput at serving time. BnB is preferred for rapid experimentation with HuggingFace-compatible GPU stacks.

---

**§2 — AWQ vs GPTQ** (≈80 words)

*Key points*:
- `AWQ` (Activation-aware Weight Quantization) identifies and protects the **1% of salient weights** most sensitive to quantization, achieving better quality than GPTQ at the same bit-width, especially at 4-bit.
- AWQ quantization is faster to compute than GPTQ (no full Hessian needed).
- GPTQ is more widely supported across serving frameworks (vLLM, TGI). AWQ is increasingly supported but has a narrower ecosystem.
- **Decision axis**: AWQ for quality-critical 4-bit deployments on modern GPUs. GPTQ for compatibility breadth.

---

**§3 — GGUF vs bitsandbytes** (≈80 words)

*Key points*:
- `GGUF` (llama.cpp format) supports both **CPU and GPU inference**, making it the only practical option for CPU-only machines. It packs multiple quantization schemes (Q2_K through Q8_0) into a single ecosystem.
- `bitsandbytes` is **GPU-exclusive**, requires CUDA, and lives entirely within the HuggingFace stack.
- GGUF is more **portable**: runs on macOS (Metal), Windows, Linux, edge devices, without framework dependencies.
- **Decision axis**: GGUF for any heterogeneous or CPU-capable deployment. BnB only for GPU-homogeneous HuggingFace deployments.

---

**§4 — Production Selection** (≈80 words)

*Key points*:
- For this assignment's hardware constraints (CPU-capable, local, no cloud): **GGUF + Ollama / llama.cpp** is the clear winner — zero GPU dependency, simple deployment, proven throughput.
- For a GPU-equipped production stack with quality SLAs: **AWQ at 4-bit** on `vLLM` gives the best throughput/quality ratio.
- For rapid HuggingFace-based prototyping with GPU available: **bitsandbytes** (load_in_4bit) for zero-friction quantization.
- **Production recommendation**: GGUF (Q4_K_M) via llama.cpp/Ollama for CPU-heavy or mixed deployments; AWQ-4bit on vLLM for pure-GPU high-throughput serving.

---

## Phase 9 — README Sections

```markdown
# Section 3 — Quantization Benchmark

## Overview
Brief explanation of the task, the model used, and what "full precision vs quantized" means in the Ollama context.

## Quantization Approach
Explain GGUF Q4_K_M vs Q2_K, why true FP32 is unavailable via Ollama, and how the comparison remains valid and instructive.

## Project Structure
Annotated directory tree.

## Environment Setup
Step-by-step: Python version, venv creation, `pip install`, `ollama pull`.

## Running the Benchmark
```bash
python scripts/benchmark.py --baseline qwen2.5:0.5b --quantized qwen2.5:0.5b-instruct-q2_k
```
Explain all CLI flags.

## Results
Embed the comparison table from `results/comparison_table.md`.

## Limitations
Honest statement of CPU benchmarking caveats and Ollama constraints.

## References
Papers and links for GPTQ, AWQ, bitsandbytes, GGUF/llama.cpp.
```

---

## Phase 10 — Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| **"Full precision" is not FP32** | The baseline is already Q4_K_M; true precision trade-off is compressed | Explicitly document this in write-up; use Q2_K vs Q4_K_M as the valid proxy |
| **CPU benchmarking variability** | RAM, background OS processes, thermal throttling all affect throughput | Run 3+ warm-up passes, discard first result, report mean ± std dev |
| **Ollama server memory isolation** | `psutil` client-side RAM ≠ actual model memory footprint | Poll the Ollama server PID directly; note limitation in results |
| **Small model (0.5B)** | Quality degradation from quantization is less dramatic than on 7B+ models | Explicitly note this; mention that findings scale differently for larger models |
| **No GPU baseline** | Cannot demonstrate VRAM savings — a core GPTQ/AWQ use case | Frame the comparison as RAM and throughput on CPU; note VRAM metrics as N/A |
| **ROUGE-L is surface-level** | Doesn't capture reasoning correctness | Supplement with qualitative human review flags |
| **Single machine run** | Results are not statistically robust across hardware configurations | Note reproducibility constraints; include system info in benchmark output |
| **Ollama model availability** | Q2_K tag may not be available for all Ollama model versions | Verify tag availability before the benchmark; fall back to Q2_K via custom Modelfile if needed |

---

## Development Order & Execution Sequence

| Step | Phase | Task | Estimated Time |
|---|---|---|---|
| 1 | Setup | Create folder structure, init git, write `requirements.txt` | 20 min |
| 2 | Setup | Install dependencies, pull Ollama models, verify connectivity | 20 min |
| 3 | Prompts | Write and validate `prompts/prompts.json` with reference answers | 25 min |
| 4 | Metrics | Implement `scripts/metrics.py` (RAM polling, latency, throughput) | 45 min |
| 5 | Benchmark | Implement `scripts/benchmark.py` (core runner, CLI args, JSON output) | 60 min |
| 6 | Quality | Implement `scripts/quality_eval.py` (ROUGE-L, optional BERTScore) | 30 min |
| 7 | Dry Run | Run baseline benchmark, verify JSON output is correct | 20 min |
| 8 | Dry Run | Run quantized benchmark, verify JSON output | 20 min |
| 9 | Report | Implement `scripts/generate_report.py` (table rendering) | 30 min |
| 10 | Results | Execute full benchmark, generate `results/comparison_table.md` | 30 min |
| 11 | Write-up | Write `results/writeup.md` (half-page discussion) | 30 min |
| 12 | README | Write `README.md` | 20 min |
| 13 | Polish | Review, clean code, add docstrings, final commit | 20 min |
| **Total** | | | **~6 hours** |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `qwen2.5:0.5b-instruct-q2_k` tag not found on Ollama Hub | Medium | High | Pre-verify with `ollama search qwen2.5`; fall back to custom Modelfile with GGUF from HuggingFace |
| `psutil` cannot locate Ollama server PID | Low | Medium | Fall back to client-side RSS measurement with explicit caveat in results |
| BERTScore download fails / too slow on CPU | Medium | Low | Mark as optional in `requirements.txt`; skip gracefully with a flag |
| High benchmark variance on CPU | High | Medium | Use 3 warm runs + 5 measured runs; report std dev alongside mean |
| Ollama not installed / version too old | Low | High | Add version check in benchmark script startup; document minimum version in README |
| ROUGE-L scores are misleadingly low due to paraphrasing | Medium | Medium | Include qualitative review as a parallel metric; note ROUGE limitation in write-up |

---

> [!TIP]
> **Quick start after approval**: The most time-critical path is Steps 4 → 5 → 7 → 8. Get the benchmark runner and metrics module working first before touching quality evaluation or report generation. A working benchmark loop with rough output is more valuable than a perfect quality evaluator.
