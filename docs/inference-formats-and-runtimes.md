# LLM artifacts, formats, and inference stacks

This document explains how large language models are **stored**, **quantized**, and **served** — with emphasis on **GGUF vs Hugging Face layouts**, **Ollama vs vLLM**, and the numeric concepts (**FP16**, **quantization**, etc.) that show up when you deploy on **Cloud Run** or elsewhere.

---

## 1. What you are actually “deploying”

A runnable LLM is not a single magic file; it is mostly:

| Piece | Role |
|--------|------|
| **Weights** | Billions of numbers (parameters) that define the network’s behavior after training. |
| **Architecture config** | Layer counts, hidden sizes, attention type — tells the runtime how to arrange those weights. |
| **Tokenizer** | Maps text ↔ tokens (integer IDs); must match how the model was trained. |
| **Chat / instruction template** | How “system / user / assistant” turns into a single prompt string (critical for quality and for **tool calling** in some stacks). |

**Training** produces high-precision weights (often **FP32** during optimization). **Inference** usually uses **lower precision** or **quantization** to save memory and speed up math on GPU/CPU.

---

## 2. Precision and dtypes (FP32, FP16, BF16)

Parameters are stored as **floating-point** or **integer** values.

| Format | Bits | Typical use |
|--------|------|-------------|
| **FP32** | 32 | Training reference; high memory; rarely needed for serving huge models. |
| **FP16** | 16 | Half precision; good GPU support; can hurt stability on some ops vs BF16. |
| **BF16** (“brain float”) | 16 | Popular on GPUs (e.g. NVIDIA A100/L4); wider exponent range than FP16 → often **more stable** for inference/training at scale. |
| **INT8 / INT4 / etc.** | 8, 4, … | After **quantization** — weights (and sometimes activations) use fewer bits. |

**Takeaway:** “Full precision” serving often means **FP16 or BF16** (16-bit), not FP32. Quantized formats shrink footprint further.

---

## 3. Quantization (what “Q4_K_M”, “Q8_0”, etc. mean)

**Quantization** maps high-precision weights (or activations) to **fewer bits**, using scaling/zero-points so the network stays usable.

### Why it exists

- **VRAM / RAM:** A 70B model at BF16 is roughly **140 GB** of weights alone — impractical on one consumer GPU.
- **Throughput:** Integer/tensor-core paths can be faster than FP16 for some kernels.

### Common GGUF / llama.cpp naming patterns

You will see tags like **Q4_K_M**, **Q5_K_S**, **Q8_0**:

- **Q** = quantized.
- **4 / 5 / 8** = target bits per weight (conceptually; exact packing differs by scheme).
- **K** variants (“k-quants”) = mixed strategies (some layers higher precision than others) — common tradeoff between quality and size.
- **_M / _S** etc. = variants within that family (different balance).

**Higher** quantization (e.g. toward **Q8**) → usually **better quality**, **larger** files. **Lower** (e.g. **Q4**) → smaller/faster, more risk of degradation.

### Weight-only vs full quantization

- **Weight-only quantization (WQ):** Only weights are INT4/INT8; activations stay FP16/BF16 — common and relatively safe.
- **More aggressive schemes** may quantize activations or use grouped scales — behavior depends on runtime.

---

## 4. GGUF vs “Hugging Face” (HF) format

These answer **different packaging questions** for **the same underlying transformer**.

### 4.1 Hugging Face Hub layout (“HF format” colloquially)

What people usually mean:

- A **model repository** on [Hugging Face Hub](https://huggingface.co/models) containing:
  - **`config.json`** — architecture, vocab size, etc.
  - **Weight shards:** often **`model-*.safetensors`** (preferred) or **`pytorch_model.bin`**
  - **Tokenizer files:** `tokenizer.json`, merges, etc.
  - Optional **`generation_config.json`**, **`tokenizer_config.json`**, etc.

**Safetensors:** A **safe, mmap-friendly** tensor storage format (no arbitrary Python pickle execution risk like older `.bin` loads).

**Primary consumers:** PyTorch **Transformers**, **vLLM**, **Text Generation Inference (TGI)**, training/finetuning scripts. Servers typically download by **`organization/model-name`** and load shards into GPU RAM.

### 4.2 GGUF

**GGUF** (GGML Universal Format, evolved from older GGML/JIT ecosystems) is a **single-file (or few-file)** format optimized for **llama.cpp**-family runtimes:

- Holds **weights + metadata** for efficient loading.
- Strong ecosystem for **CPU** and **GPU** inference via **llama.cpp**.
- **Ollama** builds on this stack for many (not all) bundled models.

**Primary consumers:** **Ollama**, **llama.cpp**, **LM Studio**, many desktop apps.

### 4.3 GGUF vs HF — conceptual comparison

| Aspect | HF Hub (`safetensors` + configs) | GGUF |
|--------|----------------------------------|------|
| Typical layout | Many files + JSON metadata | Often **one** `.gguf` per variant |
| Ecosystem | Training, research, **vLLM/TGI** | **llama.cpp**, **Ollama** |
| Quantization | Various pipelines (GPTQ, AWQ, etc.) | Built into specific **GGUF quant** variants |
| “Pull model” UX | Model ID + runtime download | **`ollama pull name:tag`** |

They are **not** rival “qualities” of the model — they are **different containers** for similar mathematical objects. Converters exist (community tooling) to move between worlds, but **production pipelines usually pick one stack**.

---

## 5. How models are “generated” (lifecycle sketch)

1. **Pre-training:** Train on large corpora → base model (often released as HF checkpoints).
2. **Instruction tuning / alignment:** SFT, RLHF/DPO-style steps → “chat” behavior.
3. **Distillation:** Large “teacher” trains smaller “student” (e.g. some **DeepSeek-R1** distilled variants).
4. **Quantization / conversion:** FP/BF16 checkpoints → **GGUF** Q4/Q8 builds, or **GPTQ/AWQ** for GPU servers.
5. **Packaging:** Uploaded to **Hub**, **Ollama library**, `mradermacher`-style mirrors, etc.

**Important:** Marketing (“supports function calling”) describes **intent or benchmarks**. What your runtime accepts — e.g. Ollama **`tools`** on `/api/chat` — also depends on **template / runtime wiring**, not weights alone.

---

## 6. Ollama vs vLLM

Short answer: they overlap, but they are not the same type of product.

- **Ollama** is closer to a **model runtime + packaging UX** (`pull`, tags, Modelfiles, easy local workflows).
- **vLLM** is primarily a **serving engine** (high-throughput inference server with OpenAI-compatible APIs).

So if someone asks “is vLLM an orchestrator like Ollama?”, the practical answer is: **not really**. vLLM focuses on inference performance; Ollama includes more model-tag lifecycle ergonomics.

### 6.1 Ollama

- **Role:** Developer-friendly **local/remote** runner; **`ollama pull`** for curated tags; **REST API** (`/api/chat`, `/api/generate`) + **OpenAI-compatible** `/v1` in many setups.
- **Weights:** Often **GGUF** behind the scenes for library models.
- **Strengths:** Simple ops, good for laptops/small servers, huge model catalog.
- **Limits:** **Native tool calling** is only reliable when **that hub image + template** supports it — your **`curl` + `tools`** test is ground truth.

Official entry points:

- [https://ollama.com/](https://ollama.com/)
- [https://github.com/ollama/ollama](https://github.com/ollama/ollama)

### 6.2 vLLM

- **Role:** **Throughput-oriented** inference server for datacenter GPUs; **OpenAI-compatible HTTP API** (`/v1/chat/completions`, etc.).
- **Weights:** Typically **Hugging Face model IDs** / safetensors-class checkpoints (not `ollama pull`).
- **Strengths:** Batching,PagedAttention-class optimizations, strong fit for **multi-user** API serving on big GPUs.
- **Limits/tradeoffs:** Less “model app” ergonomics than Ollama; you usually manage model IDs, auth, and deployment knobs more explicitly.

Official entry points:

- [https://docs.vllm.ai/](https://docs.vllm.ai/)
- [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)

### When to prefer which (rule of thumb)

| Goal | Often choose |
|------|----------------|
| Fast path + GGUF + **`ollama pull`** | **Ollama** |
| Cloud GPU API with **OpenAI-style tools** + HF models | **vLLM** (or **TGI**) |
| Minimum moving parts on a workstation | **Ollama** |
| Max **RPS**/batching on datacenter GPU | **vLLM** |

### Typical workflow comparison

| Step | Ollama-style workflow | vLLM-style workflow |
|------|------------------------|---------------------|
| Pick model | `ollama.com` tag (e.g. `qwen3:8b`) | HF model ID (e.g. `DeepHat/DeepHat-V1-7B`) |
| Build/deploy | Pull tag in image build | Deploy vLLM server image; point `MODEL_ID` to HF repo |
| API path | `/api/chat` (or Ollama `/v1` shim) | Native OpenAI-style `/v1/chat/completions` |
| Tool calling reliability | Depends on model tag + template recipe | Depends on model behavior + OpenAI tool schema support |
| Best fit | Local/dev simplicity, GGUF flows | Cloud serving, throughput, multi-user API |

---

## 7. Other concepts that appear in model cards

| Term | Meaning |
|------|---------|
| **Context length / window** | Max tokens (input + output budget) the stack is configured for; larger ≠ always feasible on your GPU RAM. |
| **Parameters (7B, 70B, …)** | Rough scale of the network; drives memory needs. |
| **MoE (Mixture of Experts)** | Only subset of “experts” active per token — big total params, lower active cost (depends on implementation). |
| **License** | Llama Community License, Apache 2.0, MIT, etc. — affects commercial use. |
| **System prompt / template** | Fixed instructions wrapping user content — affects safety **and** tool-call formatting. |

---

## 8. Curated external references

These are stable, high-signal starting points (verify dates on fast-moving projects):

1. **Ollama library & capabilities**  
   - Models: [https://ollama.com/search](https://ollama.com/search)  
   - Tool calling (docs): [https://docs.ollama.com/capabilities/tool-calling](https://docs.ollama.com/capabilities/tool-calling)

2. **Hugging Face — model hub & Safetensors**  
   - Hub: [https://huggingface.co/docs/hub/index](https://huggingface.co/docs/hub/index)  
   - Safetensors: [https://huggingface.co/docs/safetensors/index](https://huggingface.co/docs/safetensors/index)

3. **Quantization (survey / concepts)**  
   - HF quantization overview (quantization space moves quickly): search **“Hugging Face quantization”** in official docs: [https://huggingface.co/docs](https://huggingface.co/docs)

4. **GGUF / llama.cpp ecosystem**  
   - llama.cpp (upstream): [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)

5. **vLLM**  
   - Documentation: [https://docs.vllm.ai/](https://docs.vllm.ai/)

6. **Google Cloud Run + GPUs** (your deployment context)  
   - Cloud Run GPU: [https://cloud.google.com/run/docs/configuring/services/gpu](https://cloud.google.com/run/docs/configuring/services/gpu)

---

## 9. One-page mental model

```
Training → FP/BF16 (or mixed) checkpoint on Hub (safetensors + config + tokenizer)
              │
              ├─► Quantize / convert ──► GGUF ──► Ollama / llama.cpp / LM Studio
              │
              └─► Serve as-is or with server-side quant ──► vLLM / TGI / Transformers
```

**Precision** answers “how many bits per number.”  
**Quantization** answers “how aggressively we compress those numbers.”  
**GGUF vs HF** answers “which box we ship the tensors in.”  
**Ollama vs vLLM** answers “which engine loads that box and exposes which API.”

---

*Document generated for the GCP Cloud Run + Ollama workflow; adapt numbers and links as vendors update docs.*
