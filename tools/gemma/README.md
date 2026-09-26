# Phase G0: Gemma 3 4B IT (QAT) -> Q4_0 GGUF, on the desktop

`g0_pipeline.py` downloads the QAT checkpoint, builds llama.cpp at the pinned tag, converts the text model
to a BF16 GGUF, quantizes it to Q4_0, compares it with Google's official Q4_0 GGUF, measures perplexity,
runs the 20 golden assistant prompts with BF16 and Q4_0 (greedy), and writes a report. You run it on your
desktop; nothing here runs on the phone yet.

## Decide first: which QAT checkpoint

The project spec names `google/gemma-3-4b-it-qat-int4-unquantized`. Google publishes two unquantized QAT
checkpoints for the 4B IT model, and they target different formats:

| repo | QAT target (Google's description) |
|---|---|
| `google/gemma-3-4b-it-qat-q4_0-unquantized` | Q4_0, the GGUF format used by llama.cpp / Ollama |
| `google/gemma-3-4b-it-qat-int4-unquantized` | general int4 schemes (other runtimes) |

We quantize to Q4_0 with llama.cpp, so the checkpoint trained for Q4_0 is `...-q4_0-unquantized`. The
official GGUF we cross-check against also comes from the Q4_0 QAT. **The default is still the repo named in the
spec** (`gemma_sources.json` -> `source_repo`). To use the Q4_0 checkpoint instead, change that value (so the
choice is recorded in git) or pass `--source-repo google/gemma-3-4b-it-qat-q4_0-unquantized`.

## Prerequisites

* About **30 GB free disk** for `--work-dir` (HF download ~8.6 GB, BF16 GGUF ~7.8 GB, official GGUF ~3.2 GB,
  comparison variant ~3.2 GB, build ~1 GB). Our Q4_0 (~2.2 GB) goes to `tools/reference/gemma/`.
* **16 GB RAM** recommended: the BF16 runs map a 7.8 GB file.
* Git, CMake 3.18+, a C++17 compiler, Python 3.10+ (3.12 recommended, 64-bit).
  * Windows: "Visual Studio 2022 Build Tools" with the "Desktop development with C++" workload (includes
    CMake), Git for Windows, Python 3.12 from python.org.
  * Linux: `build-essential cmake git python3-venv`. macOS: Xcode command line tools + `brew install cmake`.
* A Hugging Face account that has **accepted the Gemma terms** on these pages (open each one while logged in):
  * https://huggingface.co/google/gemma-3-4b-it-qat-int4-unquantized (or the q4_0 one, see above)
  * https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf
* A **read** token from https://huggingface.co/settings/tokens, given to the script through the `HF_TOKEN`
  environment variable only. It is never written to disk or into any output.

## Run

From the repo root. Windows (PowerShell):

```powershell
py -3.12 -m venv tools\gemma\.venv
tools\gemma\.venv\Scripts\python.exe -m pip install -r tools\gemma\requirements.txt
$env:HF_TOKEN = "hf_..."          # this shell only
tools\gemma\.venv\Scripts\python.exe tools\gemma\g0_pipeline.py --work-dir D:\gemma-work
```

Linux / macOS:

```bash
python3.12 -m venv tools/gemma/.venv
tools/gemma/.venv/bin/pip install -r tools/gemma/requirements.txt
export HF_TOKEN=hf_...
tools/gemma/.venv/bin/python tools/gemma/g0_pipeline.py --work-dir ~/gemma-work
```

Rough CPU-only timing on an 8-core desktop: build 5-15 min, downloads depend on your line, convert ~5 min,
quantize ~2 min, inspect ~5 min, perplexity 30-60 min (BF16 is the slow one), golden prompts 45-90 min.
Every step is recorded in `<work-dir>/state.json`; if something stops, fix it and run the same command again,
finished steps are skipped. Run a single step with e.g. `g0_pipeline.py perplexity`, redo one with `--force`.
Logs of every tool run are in `<work-dir>/logs/`.

Useful options: `--threads N` (default: physical cores), `--ppl-chunks N` (default 150 chunks of 512 tokens),
`--source-revision <sha>` / `--official-revision <sha>` (pin a Hugging Face commit),
`--skip-variant`, `--llama-cpp-dir <checkout at the pinned commit>`.

## Steps and what they check

| step | does | fails loudly if |
|---|---|---|
| check | Python, git, cmake, packages, free disk, HF_TOKEN | anything missing |
| llama | clones llama.cpp at the tag in `llama_cpp_pin.json`, builds `llama-quantize`, `llama-perplexity` and `runner/gemma_runner` in one CPU-only static build | the checkout is not at the pinned commit |
| download | resolves `main` to a commit sha, downloads `*.json`, `*.safetensors`, `tokenizer.model` at that sha | Gemma terms not accepted; any file's SHA-256 differs from Hugging Face's |
| convert | `convert_hf_to_gguf.py --outtype bf16` (text model only; no `--mmproj`) | architecture is not gemma3, any vision tensor, any tensor not BF16/F32 |
| quantize | `llama-quantize --pure --token-embedding-type q4_0 <bf16> <out> Q4_0` | any tensor that is not Q4_0 (weights) or F32 (norms), any K-quant, imatrix metadata |
| official | downloads `gemma-3-4b-it-q4_0.gguf` only (never the mmproj file) | file missing (lists the repo's .gguf files), SHA-256 mismatch |
| variant | if the official file stores `token_embd` in another type (e.g. F16), also builds our Q4_0 with that `token_embd` type, for comparison only | - |
| inspect | tensor-type diff ours vs official, metadata diff, tokenizer fingerprints, grid exactness vs BF16 | - |
| perplexity | `llama-perplexity` on wikitext-2 test (same file for all; its SHA-256 is recorded), ctx 512 | no final estimate in the log |
| golden | builds the 20 prompts (`golden_llm/tasks.json`) in the Phase 7a format, checks llama.cpp's template against the GGUF's Jinja template token by token, then greedy generation with BF16 and our Q4_0 | a model tokenizes the prompt differently; an expected line was cut by truncation |
| report | `gemma_manifest.json`, `g0_report.md/.json`, `golden_llm/side_by_side.md` | - |

Why `--pure --token-embedding-type q4_0`: Gemma 3 ties the token embedding to the output layer, and with
plain `Q4_0` llama-quantize (b11201, `src/llama-quant.cpp`) stores that tensor as **Q6_K**, a K-quant. The spec
says Q4_0 with no K-quants, so every quantizable tensor is Q4_0. Google's official file is expected to keep
`token_embd` at higher precision; the report shows the difference, and the `variant` file measures how much
of any quality gap comes from that one tensor.

Grid exactness: a QAT checkpoint's weights can be reproduced exactly by Q4_0 only if the quantizer picks the
same per-block scales that QAT trained against. `llama-quantize` picks `scale = max / -8` per block of 32;
reports on Gemma 4 QAT found that plain llama.cpp Q4_0 reproduces only about a quarter of QAT weights exactly.
The report measures this for our file and the official one, so a perplexity gap can be traced to its cause.

## Prompt format (shared with Phase 7a)

`prompt_builder.py` builds the messages exactly as the phone will (texts and thresholds from
`tools/reference/gemma/gemma_config.json`): lines numbered `[L1] ... [Ln]` in reading order, blank lines
skipped (with a map back to the OCR line), `(low confidence)` when confidence < 0.80, the question, and
truncation from the end in reading order when the prompt would exceed `n_ctx - max_new_tokens`, with a note
saying how many lines were left out. The chat template is applied by llama.cpp's `llama_chat_apply_template`
with the template stored in the GGUF. Gemma 3 has no system role: llama.cpp (b11201,
`src/llama-chat.cpp`, `LLM_CHAT_TEMPLATE_GEMMA`) trims the system text and puts it at the start of the first
user turn, followed by a blank line:

```
<bos><start_of_turn>user
{system text}

Document lines:
[L1] ...
Question: ...<end_of_turn>
<start_of_turn>model
```

(`<bos>` is added by the tokenizer, not by the template.) The golden step also renders every prompt with the
GGUF's own Jinja template and checks that both give the same token ids.

## Outputs

```
tools/reference/gemma/
  gemma_config.json              committed now: runtime + prompt settings, each with its source
  golden_llm/tasks.json          committed now: the 20 tasks
  gemma-3-4b-it-qat-q4_0.gguf    written by the run; NOT committed (*.gguf is git-ignored)
  gemma_manifest.json            SHA-256, bytes, quant type, llama.cpp tag/commit, HF revision
  g0_report.md / g0_report.json  all numbers
  golden_llm/prompts.json        fully built prompts + token ids (fixtures for Phase 7a and G1)
  golden_llm/outputs_bf16.json   greedy outputs + token ids + timings
  golden_llm/outputs_q4_0.json
  golden_llm/side_by_side.md     BF16 vs Q4_0 answers for reading
```

After a run, commit everything in that folder except the `.gguf` (and `g0_report.json` if you prefer; it
contains local paths). Keep the `.gguf`: G1 imports that exact file on the phone and checks it against the
manifest.

## Offline smoke test

`tests/smoke_test_pipeline.py` runs every step on a tiny random Gemma 3 model built locally (no Hugging Face
access), including the official-file and variant logic. Its numbers are meaningless; it only proves the
plumbing works:

```bash
python tools/gemma/tests/smoke_test_pipeline.py --work-dir /tmp/g0-smoke
python -m unittest discover -s tools/gemma/tests          # prompt builder + golden task checks
```

## Privacy

The prompts are built from the Phase 0 golden OCR text of blank public forms. The only network access is the
Hugging Face downloads. Nothing is uploaded.
