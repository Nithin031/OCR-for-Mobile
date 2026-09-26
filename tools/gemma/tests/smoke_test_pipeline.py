"""Offline end-to-end run of g0_pipeline.py on a tiny random Gemma 3 model. No Hugging Face access.

    python tools/gemma/tests/smoke_test_pipeline.py --work-dir <tmp dir> [--llama-cpp-dir <pinned checkout>]

Runs every step: build, convert (multimodal checkpoint -> text-only BF16 GGUF), quantize (--pure Q4_0),
an "official" stand-in (Q4_0 with F16 token_embd, like Google's file), the matching variant, inspect,
perplexity, golden prompts (runner + Jinja cross-check) and the report, then checks the outputs.
The model is random, so every number it prints is meaningless; this only tests the plumbing.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEMMA = HERE.parent
REPO = GEMMA.parents[1]
sys.path.insert(0, str(GEMMA))
sys.path.insert(0, str(HERE))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--llama-cpp-dir", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)

    from gemma_common import find_binary, load_json, sha256_file  # noqa: PLC0415
    from make_tiny_gemma3 import build  # noqa: PLC0415

    cfg = load_json(REPO / "tools/reference/gemma/gemma_config.json")
    text = work / "english_text.txt"
    parts = [cfg["prompt"]["system_text"]["value"]]
    for f in sorted((REPO / "tools/reference/golden/max960/paddle").glob("*.json")):
        parts += [l["text"] for l in load_json(f)["lines"]]
    text.write_text("\n".join(parts) * 3 + "\n", encoding="utf-8")

    tiny = work / "tiny_hf"
    if not (tiny / "config.json").is_file():
        print("building tiny random Gemma 3 checkpoint ...", flush=True)
        build(tiny, text)

    pipe_work, out = work / "pipeline_work", work / "out"
    official = work / "official_standin.gguf"
    common = ["--work-dir", pipe_work, "--out-dir", out, "--local-source-dir", tiny, "--local-official-gguf", official,
              "--perplexity-text", text, "--skip-disk-check", "--ppl-chunks", "4", "--threads", str(args.threads)]
    if args.llama_cpp_dir:
        common += ["--llama-cpp-dir", args.llama_cpp_dir]
    run = lambda steps: subprocess.run([sys.executable, GEMMA / "g0_pipeline.py", *steps, *map(str, common)], check=True)  # noqa: E731

    run(["check", "llama", "download", "convert", "quantize"])
    quantize = find_binary(pipe_work / "build", "llama-quantize")
    subprocess.run([str(quantize), "--pure", "--token-embedding-type", "f16", str(pipe_work / "gemma-3-4b-it-qat-bf16.gguf"),
                    str(official), "Q4_0", "2"], check=True, capture_output=True)
    run(["official", "variant", "inspect", "perplexity", "golden", "report"])

    # ---- checks
    state = load_json(pipe_work / "state.json")["steps"]
    manifest = load_json(out / "gemma_manifest.json")
    q4 = out / "gemma-3-4b-it-qat-q4_0.gguf"
    assert manifest["sha256"] == sha256_file(q4), "manifest sha256 does not match our Q4_0"
    assert manifest["token_embd_type"] == "Q4_0", manifest["token_embd_type"]
    assert set(manifest["tensor_types"]) == {"Q4_0", "F32"}, manifest["tensor_types"]
    assert state["variant"]["token_embd_type"] == "F16", state["variant"]
    ins = load_json(pipe_work / "inspect.json")
    assert ins["summaries"]["bf16"]["tensor_types"].keys() == {"BF16", "F32"}
    assert not any(n.startswith(("v.", "mm.")) for n in ins["ours_vs_official_types"]["only_in_a"])
    assert ins["ours_vs_official_types"]["type_difference_counts"] == {"Q4_0 -> F16": 1}, ins["ours_vs_official_types"]
    # the variant differs from the official stand-in only by the quantizer run -> must be bit-identical tensors
    assert ins["variant_vs_official_types"]["type_differences"] == []
    assert sha256_file(Path(state["variant"]["path"])) != "" and state["variant"]["sha256"]
    ge = ins["grid_exactness_vs_bf16"]
    assert abs(ge["variant"]["overall_exact_fraction"] - ge["official"]["overall_exact_fraction"]) < 1e-12
    prompts = load_json(out / "golden_llm/prompts.json")["prompts"]
    assert len(prompts) == 20
    assert all(p["jinja_crosscheck"]["identical_token_ids"] for p in prompts), \
        [p["id"] for p in prompts if not p["jinja_crosscheck"]["identical_token_ids"]]
    for name in ("bf16", "q4_0"):
        o = load_json(out / f"golden_llm/outputs_{name}.json")
        assert len(o["outputs"]) == 20 and o["peak_rss_bytes"], name
    ppl = state["perplexity"]["results"]
    assert set(ppl) == {"bf16", "ours", "official", "variant"}, ppl.keys()
    report = (out / "g0_report.md").read_text(encoding="utf-8")
    assert "## Perplexity" in report and "## Golden prompts" in report
    print("\nSMOKE TEST PASSED")
    print(json.dumps({k: {"ppl": v["ppl"], "peak_rss": v["peak_rss_bytes"]} for k, v in ppl.items()}, indent=1))


if __name__ == "__main__":
    main()
