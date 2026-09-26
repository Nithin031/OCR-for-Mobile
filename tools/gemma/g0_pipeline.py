#!/usr/bin/env python3
"""Phase G0: quantize Gemma 3 4B IT (QAT) to Q4_0 GGUF with the pinned llama.cpp, and measure it.

Steps (run all, or name some):  check llama download convert quantize official variant
                                inspect perplexity golden report
Every step records what it did in <work>/state.json and is skipped next time unless --force.
Read tools/gemma/README.md first. The Hugging Face token is read from the HF_TOKEN environment
variable and is never written anywhere.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gemma_common import (  # noqa: E402
    CONFIG_FILE, GEMMA_TOOLS, PIN_FILE, REF_GEMMA, REPO_ROOT, SOURCES_FILE, TASKS_FILE,
    Runner, config_value, find_binary, human_bytes, load_json, physical_cores, run_measured,
    sha256_file, sha256_text, write_json,
)
from golden_tasks import quick_actions_from_config, resolve_tasks  # noqa: E402
from prompt_builder import PromptConfig, build_prompt  # noqa: E402

STEPS = ["check", "llama", "download", "convert", "quantize", "official", "variant",
         "inspect", "perplexity", "golden", "report"]
MIN_FREE_BYTES = 30 * 10**9
OUR_Q4_NAME = "gemma-3-4b-it-qat-q4_0.gguf"
PPL_RE = re.compile(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)")


class G0Error(SystemExit):
    pass


def fail(msg: str) -> None:
    raise G0Error(f"\nG0 STOPPED: {msg}\n")


def log(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- context

class Ctx:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.work: Path = args.work_dir.resolve()
        self.logs = self.work / "logs"
        self.state_file = self.work / "state.json"
        self.state: dict = load_json(self.state_file) if self.state_file.is_file() else {"steps": {}}
        self.pin = load_json(PIN_FILE)
        self.sources = load_json(SOURCES_FILE)
        self.config = load_json(CONFIG_FILE)
        self.source_repo = args.source_repo or self.sources["source_repo"]
        self.llama_dir = (args.llama_cpp_dir or self.work / "llama.cpp").resolve()
        self.build_dir = self.work / "build"
        self.src_dir = self.work / "hf" / self.source_repo.replace("/", "__")
        self.bf16 = self.work / "gemma-3-4b-it-qat-bf16.gguf"
        self.official_dir = self.work / "official"
        self.ppl_dir = self.work / "perplexity"
        self.threads = args.threads or physical_cores()
        self.out = (args.out_dir or REF_GEMMA).resolve()
        self.our_q4 = self.out / OUR_Q4_NAME
        self.golden_out = self.out / "golden_llm"

    def token(self) -> str | None:
        return os.environ.get("HF_TOKEN") or None

    def step(self, name: str) -> dict:
        return self.state["steps"].get(name, {})

    def done(self, name: str) -> bool:
        return bool(self.step(name).get("done")) and not self.args.force

    def save(self, name: str, data: dict) -> None:
        data = dict(data)
        data["done"] = True
        data["finished"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        self.state["steps"][name] = data
        write_json(self.state_file, self.state)

    def bin(self, name: str) -> Path:
        return find_binary(self.build_dir, name)

    def runtime(self, key: str):
        return config_value(self.config, "runtime", key)

    def runner(self, model: Path, log_name: str, vocab_only: bool = False) -> Runner:
        return Runner(
            self.bin("gemma_runner"), model, self.logs / log_name, vocab_only=vocab_only,
            n_ctx=self.runtime("n_ctx"), n_batch=self.runtime("n_batch"), n_ubatch=self.runtime("n_ubatch"),
            threads=self.threads, threads_batch=self.threads, flash_attn=self.runtime("flash_attn"),
            swa_full=self.runtime("swa_full"), load_mode=self.runtime("load_mode"),
            extra_bufts=self.runtime("use_extra_bufts"))

    def variant_path(self) -> Path | None:
        v = self.step("variant")
        return Path(v["path"]) if v.get("path") else None


def run_or_fail(cmd: list, log_file: Path, what: str, **kw):
    log(f"{what}: {' '.join(str(c) for c in cmd[:3])} ...  (log: {log_file})")
    res = run_measured(cmd, log_file, **kw)
    if res.returncode != 0:
        fail(f"{what} failed with exit code {res.returncode}. See the end of {log_file}")
    return res


def _hf():
    try:
        import huggingface_hub  # noqa: PLC0415
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # noqa: PLC0415
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError  # noqa: PLC0415
    except ImportError:
        fail("huggingface_hub is not installed; pip install -r tools/gemma/requirements.txt")
    return huggingface_hub, HfApi, hf_hub_download, snapshot_download, GatedRepoError, RepositoryNotFoundError


def _check_access(api, repo: str, token: str | None, GatedRepoError, RepositoryNotFoundError, repo_type: str = "model"):
    try:
        api.auth_check(repo, repo_type=repo_type, token=token)
    except GatedRepoError:
        fail(f"no access to {repo}. Open https://huggingface.co/{repo} while logged in, accept the Gemma "
             f"terms, and make sure HF_TOKEN belongs to that account.")
    except RepositoryNotFoundError:
        fail(f"{repo} was not found (or HF_TOKEN is missing/invalid).")


def _verify_sibling(path: Path, sib) -> dict:
    digest = sha256_file(path)
    rec = {"file": sib.rfilename, "bytes": path.stat().st_size, "sha256": digest}
    lfs = getattr(sib, "lfs", None)
    if lfs is not None:
        want = lfs.sha256 if hasattr(lfs, "sha256") else lfs["sha256"]
        if digest != want:
            fail(f"{sib.rfilename}: sha256 {digest} does not match Hugging Face ({want})")
        rec["verified_against"] = "huggingface lfs sha256"
    return rec


# ---------------------------------------------------------------- steps

def step_check(c: Ctx) -> None:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
            "physical_cores": physical_cores(), "threads_used": c.threads}
    if sys.version_info < (3, 10):
        fail("Python 3.10+ is required (3.12 recommended)")
    for tool in ("git", "cmake"):
        if shutil.which(tool) is None:
            fail(f"'{tool}' is not on PATH. See tools/gemma/README.md (prerequisites)")
    info["cmake"] = subprocess.run(["cmake", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    versions = {}
    for mod in ("torch", "transformers", "numpy", "sentencepiece", "google.protobuf", "huggingface_hub", "jinja2",
                "safetensors"):
        try:
            m = __import__(mod, fromlist=["__version__"])
            versions[mod] = getattr(m, "__version__", "?")
        except ImportError:
            fail(f"Python package '{mod}' is missing; pip install -r tools/gemma/requirements.txt")
    info["python_packages"] = versions
    try:
        import psutil  # noqa: PLC0415
        info["ram_total_bytes"] = psutil.virtual_memory().total
    except ImportError:
        info["ram_total_bytes"] = None
    c.work.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(c.work).free
    info["work_dir"] = str(c.work)
    info["work_dir_free_bytes"] = free
    if free < MIN_FREE_BYTES and not c.args.skip_disk_check:
        fail(f"only {human_bytes(free)} free at {c.work}; G0 needs about {human_bytes(MIN_FREE_BYTES)}. "
             f"Use --work-dir on a bigger drive (or --skip-disk-check if you know it fits).")
    offline = c.args.local_source_dir and c.args.local_official_gguf and c.args.perplexity_text
    if not c.token() and not offline:
        fail("HF_TOKEN is not set. Create a read token at https://huggingface.co/settings/tokens and set it "
             "in this shell (PowerShell: $env:HF_TOKEN = \"hf_...\"; bash: export HF_TOKEN=hf_...).")
    info["hf_token_present"] = bool(c.token())
    log(f"check ok: Python {info['python']}, {info['physical_cores']} physical cores, "
        f"{human_bytes(free)} free in {c.work}")
    c.save("check", info)


def step_llama(c: Ctx) -> None:
    pin = c.pin
    if not (c.llama_dir / "CMakeLists.txt").is_file():
        if c.args.llama_cpp_dir:
            fail(f"--llama-cpp-dir {c.llama_dir} is not a llama.cpp checkout")
        log(f"cloning llama.cpp {pin['tag']} into {c.llama_dir}")
        r = subprocess.run(["git", "clone", "--depth", "1", "--branch", pin["tag"], pin["repo"], str(c.llama_dir)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            fail(f"git clone failed:\n{r.stderr}")
    head = subprocess.run(["git", "-C", str(c.llama_dir), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != pin["commit"]:
        fail(f"llama.cpp checkout is at {head}, but {PIN_FILE.name} pins {pin['tag']} = {pin['commit']}. "
             f"Delete {c.llama_dir} and re-run (or check the tag was not moved upstream).")
    cfg_cmd = ["cmake", "-S", GEMMA_TOOLS / "runner", "-B", c.build_dir, f"-DLLAMA_CPP_DIR={c.llama_dir}",
               "-DCMAKE_BUILD_TYPE=Release"]
    run_or_fail(cfg_cmd, c.logs / "build.log", "cmake configure")
    build_cmd = ["cmake", "--build", c.build_dir, "--config", "Release", "--target", "llama-quantize",
                 "llama-perplexity", "gemma_runner", "--parallel", str(c.args.jobs or os.cpu_count() or 4)]
    res = run_or_fail(build_cmd, c.logs / "build.log", "build llama.cpp tools + gemma_runner")
    bins = {n: str(c.bin(n)) for n in ("llama-quantize", "llama-perplexity", "gemma_runner")}
    cache = (c.build_dir / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace")
    def cache_val(key: str) -> str | None:
        m = re.search(rf"^{re.escape(key)}:[A-Z]+=(.*)$", cache, re.M)
        return m.group(1) if m else None
    c.save("llama", {"tag": pin["tag"], "commit": head, "dir": str(c.llama_dir), "binaries": bins,
                     "build_seconds": round(res.wall_s, 1), "cmake_generator": cache_val("CMAKE_GENERATOR"),
                     "cxx_compiler": cache_val("CMAKE_CXX_COMPILER"), "ggml_native": cache_val("GGML_NATIVE")})
    log(f"llama.cpp {pin['tag']} ({head[:10]}) built")


def step_download(c: Ctx) -> None:
    if c.args.local_source_dir:
        src = c.args.local_source_dir.resolve()
        files = [{"file": str(p.relative_to(src)), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
                 for p in sorted(src.rglob("*")) if p.is_file()]
        c.save("download", {"repo": c.source_repo, "revision": "LOCAL (not from Hugging Face)", "dir": str(src),
                            "files": files, "local": True})
        log(f"using local source model at {src} (test mode)")
        return
    hub, HfApi, _, snapshot_download, GatedRepoError, RepositoryNotFoundError = _hf()
    api = HfApi()
    token = c.token()
    _check_access(api, c.source_repo, token, GatedRepoError, RepositoryNotFoundError)
    info = api.model_info(c.source_repo, revision=c.args.source_revision, files_metadata=True, token=token)
    patterns = c.sources["source_allow_patterns"]
    wanted = [s for s in info.siblings if any(fnmatch.fnmatch(s.rfilename, p) for p in patterns)]
    if not any(s.rfilename.endswith(".safetensors") for s in wanted):
        fail(f"{c.source_repo}@{info.sha} has no .safetensors files: {[s.rfilename for s in info.siblings]}")
    total = sum((s.size or 0) for s in wanted)
    log(f"downloading {c.source_repo} @ {info.sha} ({len(wanted)} files, {human_bytes(total)})")
    snapshot_download(repo_id=c.source_repo, revision=info.sha, allow_patterns=patterns,
                      local_dir=str(c.src_dir), token=token)
    records = [_verify_sibling(c.src_dir / s.rfilename, s) for s in wanted]
    cfg = load_json(c.src_dir / "config.json")
    c.save("download", {"repo": c.source_repo, "revision": info.sha, "requested_revision": c.args.source_revision,
                        "dir": str(c.src_dir), "files": records, "architectures": cfg.get("architectures"),
                        "huggingface_hub": hub.__version__})
    log(f"downloaded and verified {len(records)} files")


def step_convert(c: Ctx) -> None:
    from gguf_inspect import GgufFile, check_bf16, import_gguf, vision_tensors  # noqa: PLC0415
    src = Path(c.step("download")["dir"])
    env = dict(os.environ)
    env.pop("NO_LOCAL_GGUF", None)  # make the converter use the pinned gguf-py
    cmd = [sys.executable, c.llama_dir / "convert_hf_to_gguf.py", src, "--outtype", "bf16", "--outfile", c.bf16]
    res = run_or_fail(cmd, c.logs / "convert.log", "convert_hf_to_gguf (BF16, text only)", env=env)
    import_gguf(c.llama_dir)
    g = GgufFile(c.bf16)
    if g.field("general.architecture") != "gemma3":
        fail(f"converted architecture is {g.field('general.architecture')}, expected gemma3")
    vis = vision_tensors(g)
    if vis:
        fail(f"vision tensors found in the text model: {vis[:5]}")
    problems = check_bf16(g)
    if problems:
        fail("BF16 GGUF has unexpected tensor types:\n  " + "\n  ".join(problems[:20]))
    c.save("convert", {"path": str(c.bf16), "bytes": c.bf16.stat().st_size, "sha256": sha256_file(c.bf16),
                       "summary": g.summary(), "peak_rss_bytes": res.peak_rss_bytes, "seconds": round(res.wall_s, 1),
                       "command": ["convert_hf_to_gguf.py", "<source snapshot dir>", "--outtype", "bf16",
                                   "--outfile", c.bf16.name]})
    log(f"BF16 GGUF: {human_bytes(c.bf16.stat().st_size)}, {g.summary()['n_params']:,} parameters")


def _quantize(c: Ctx, out: Path, token_embd_type: str, log_name: str) -> dict:
    from gguf_inspect import GgufFile, check_q4_0, import_gguf  # noqa: PLC0415
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [c.bin("llama-quantize"), "--pure", "--token-embedding-type", token_embd_type.lower(),
           c.bf16, out, "Q4_0", str(c.threads)]
    res = run_or_fail(cmd, c.logs / log_name, f"llama-quantize Q4_0 (token_embd {token_embd_type})")
    import_gguf(c.llama_dir)
    g = GgufFile(out)
    problems = check_q4_0(g, token_embd_type=token_embd_type.upper())
    if problems:
        fail(f"{out.name} does not have the expected tensor types:\n  " + "\n  ".join(problems[:20]))
    return {"path": str(out), "bytes": out.stat().st_size, "sha256": sha256_file(out), "summary": g.summary(),
            "command": ["llama-quantize"] + [str(x) if not isinstance(x, Path) else x.name for x in cmd[1:]],
            "peak_rss_bytes": res.peak_rss_bytes, "seconds": round(res.wall_s, 1)}


def step_quantize(c: Ctx) -> None:
    rec = _quantize(c, c.our_q4, "Q4_0", "quantize.log")
    c.save("quantize", rec)
    log(f"our Q4_0: {human_bytes(rec['bytes'])}, sha256 {rec['sha256'][:16]}...")


def step_official(c: Ctx) -> None:
    from gguf_inspect import GgufFile, import_gguf, vision_tensors  # noqa: PLC0415
    fname = c.sources["official_gguf_file"]
    repo = c.sources["official_gguf_repo"]
    if c.args.local_official_gguf:
        path = c.args.local_official_gguf.resolve()
        rec = {"repo": repo, "revision": "LOCAL (not from Hugging Face)", "file": fname, "path": str(path),
               "bytes": path.stat().st_size, "sha256": sha256_file(path), "local": True}
    else:
        hub, HfApi, hf_hub_download, _, GatedRepoError, RepositoryNotFoundError = _hf()
        api = HfApi()
        token = c.token()
        _check_access(api, repo, token, GatedRepoError, RepositoryNotFoundError)
        info = api.model_info(repo, revision=c.args.official_revision, files_metadata=True, token=token)
        sib = next((s for s in info.siblings if s.rfilename == fname), None)
        if sib is None:
            ggufs = [s.rfilename for s in info.siblings if s.rfilename.endswith(".gguf")]
            fail(f"{fname} is not in {repo}@{info.sha}. GGUF files there: {ggufs}. "
                 f"Pick the text-model file and set official_gguf_file in {SOURCES_FILE.name}.")
        log(f"downloading official {repo}/{fname} @ {info.sha} ({human_bytes(sib.size)})")
        path = Path(hf_hub_download(repo_id=repo, filename=fname, revision=info.sha,
                                    local_dir=str(c.official_dir), token=token))
        rec = _verify_sibling(path, sib)
        rec.update({"repo": repo, "revision": info.sha, "path": str(path)})
    import_gguf(c.llama_dir)
    g = GgufFile(path)
    if vision_tensors(g):
        fail(f"{fname} contains vision tensors; it is not the text-only model")
    rec["summary"] = g.summary()
    c.save("official", rec)
    log(f"official Q4_0: {human_bytes(rec['bytes'])}, token_embd {rec['summary']['token_embd_type']}")


def step_variant(c: Ctx) -> None:
    if c.args.skip_variant:
        c.save("variant", {"skipped": "--skip-variant"})
        return
    t = c.step("official")["summary"]["token_embd_type"]
    if t == "Q4_0":
        c.save("variant", {"skipped": "official token_embd is Q4_0 like ours; no variant needed"})
        log("variant not needed (official token_embd is Q4_0)")
        return
    out = c.work / f"gemma-3-4b-it-qat-q4_0-embd-{t.lower()}.gguf"
    rec = _quantize(c, out, t, "quantize_variant.log")
    rec["token_embd_type"] = t
    rec["purpose"] = ("comparison only: our Q4_0 with token_embd stored like the official file, to separate the "
                      "effect of the embedding type from the effect of the quantizer's scale choice")
    c.save("variant", rec)
    log(f"variant (token_embd {t}): {human_bytes(rec['bytes'])}")


def step_inspect(c: Ctx) -> None:
    from gguf_inspect import (GgufFile, compare_metadata, compare_types, grid_exactness,  # noqa: PLC0415
                              import_gguf, metadata, tokenizer_fingerprint)
    import_gguf(c.llama_dir)
    files = {"bf16": c.bf16, "ours": c.our_q4, "official": Path(c.step("official")["path"])}
    if c.variant_path():
        files["variant"] = c.variant_path()
    g = {k: GgufFile(p) for k, p in files.items()}
    out: dict = {"files": {k: str(p) for k, p in files.items()}}
    out["metadata"] = {k: metadata(v) for k, v in g.items()}
    out["tokenizer_fingerprint"] = {k: tokenizer_fingerprint(v) for k, v in g.items()}
    out["ours_vs_official_types"] = compare_types(g["ours"], g["official"])
    out["ours_vs_official_metadata"] = compare_metadata(g["ours"], g["official"])
    if "variant" in g:
        out["variant_vs_official_types"] = compare_types(g["variant"], g["official"])
    out["summaries"] = {k: v.summary() for k, v in g.items()}
    log("grid exactness vs BF16 (dequantizing every quantized tensor; takes a few minutes)")
    out["grid_exactness_vs_bf16"] = {}
    for k in [x for x in ("ours", "variant", "official") if x in g]:
        t0 = time.monotonic()
        res = grid_exactness(g[k], g["bf16"])
        out["grid_exactness_vs_bf16"][k] = res
        log(f"  {k}: exact {res['overall_exact_fraction']:.4%}, rel RMSE {res['overall_rel_rmse']:.3e} "
            f"({time.monotonic() - t0:.0f} s)")
    write_json(c.work / "inspect.json", out)
    fp = out["tokenizer_fingerprint"]
    c.save("inspect", {"inspect_file": str(c.work / "inspect.json"),
                       "tokenizer_same_bf16_ours": fp["bf16"] == fp["ours"],
                       "tokenizer_same_ours_official": fp["ours"] == fp["official"]})


def _perplexity_text(c: Ctx) -> tuple[Path, dict]:
    if c.args.perplexity_text:
        p = c.args.perplexity_text.resolve()
        return p, {"source": "LOCAL (test mode)", "path": str(p), "sha256": sha256_file(p)}
    spec = c.sources["perplexity_text"]
    target = c.ppl_dir / Path(spec["member"]).name
    hub, HfApi, hf_hub_download, *_ = _hf()
    m = re.match(r"https://huggingface.co/datasets/([^/]+/[^/]+)/resolve/([^/]+)/(.+)$", spec["url"])
    repo, rev, fname = m.group(1), m.group(2), m.group(3)
    info = HfApi().dataset_info(repo, revision=rev)
    if not target.is_file():
        log(f"downloading perplexity text {repo}/{fname} @ {info.sha}")
        z = Path(hf_hub_download(repo_id=repo, repo_type="dataset", filename=fname, revision=info.sha,
                                 local_dir=str(c.ppl_dir)))
        with zipfile.ZipFile(z) as zf:
            target.write_bytes(zf.read(spec["member"]))
    return target, {"source": spec["url"], "dataset_revision": info.sha, "member": spec["member"],
                    "path": str(target), "sha256": sha256_file(target), "bytes": target.stat().st_size}


def step_perplexity(c: Ctx) -> None:
    text, text_info = _perplexity_text(c)
    models = {"bf16": c.bf16, "ours": c.our_q4, "official": Path(c.step("official")["path"])}
    if c.variant_path():
        models["variant"] = c.variant_path()
    params = {"ctx": c.args.ppl_ctx, "batch": c.args.ppl_ctx, "chunks": c.args.ppl_chunks, "threads": c.threads,
              "flash_attn": "off", "n_gpu_layers": 0, "fit": "off"}
    results = {}
    for name, model in models.items():
        log_file = c.logs / f"perplexity_{name}.log"
        if log_file.exists():
            log_file.unlink()
        cmd = [c.bin("llama-perplexity"), "-m", model, "-f", text, "-c", params["ctx"], "-b", params["batch"],
               "--chunks", params["chunks"], "-t", params["threads"], "-fa", "off", "-ngl", "0", "-fit", "off"]
        res = run_or_fail(cmd, log_file, f"perplexity {name}")
        m = PPL_RE.search(log_file.read_text(encoding="utf-8", errors="replace"))
        if not m:
            fail(f"no 'Final estimate' line in {log_file}")
        results[name] = {"model": str(model), "ppl": float(m.group(1)), "ppl_err": float(m.group(2)),
                         "peak_rss_bytes": res.peak_rss_bytes, "seconds": round(res.wall_s, 1)}
        log(f"  {name}: PPL {m.group(1)} +/- {m.group(2)}  ({res.wall_s:.0f} s, peak {human_bytes(res.peak_rss_bytes)})")
    base = results["bf16"]["ppl"]
    for r in results.values():
        r["vs_bf16_percent"] = (r["ppl"] - base) / base * 100.0
    flag = results["ours"]["vs_bf16_percent"] > 5.0
    c.save("perplexity", {"text": text_info, "params": params, "results": results,
                          "flag_ours_more_than_5_percent_worse": flag})
    if flag:
        log(f"FLAG: our Q4_0 perplexity is {results['ours']['vs_bf16_percent']:.2f}% worse than BF16 (> 5%)")


def _jinja_render(template: str, messages: list[dict], bos: str) -> str:
    import jinja2  # noqa: PLC0415
    from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: PLC0415

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template).render(messages=messages, bos_token=bos, add_generation_prompt=True)


def _first_diff(a: list, b: list) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def step_golden(c: Ctx) -> None:
    from gguf_inspect import GgufFile, import_gguf  # noqa: PLC0415
    ins = c.step("inspect")
    if not ins.get("tokenizer_same_bf16_ours"):
        fail("BF16 and Q4_0 files have different tokenizers; prompts would differ between them")
    cfg = PromptConfig.from_config_file(CONFIG_FILE)
    tasks = resolve_tasks(TASKS_FILE, REPO_ROOT, cfg, quick_actions_from_config(CONFIG_FILE))
    import_gguf(c.llama_dir)
    vocab_tokens = GgufFile(c.our_q4).field("tokenizer.ggml.tokens")
    max_new = config_value(c.config, "generation", "max_new_tokens")

    log(f"building {len(tasks)} prompts (token counts from the model tokenizer)")
    prompts = []
    with c.runner(c.our_q4, "golden_vocab.log", vocab_only=True) as rv:
        template = rv.ready["chat_template"]
        # the BOS llama.cpp actually prepends (llama_vocab_bos), not a metadata key that may be absent
        bos_text = vocab_tokens[rv.ready["bos_token"]]
        for t in tasks:
            built = build_prompt(t.lines, t.question, cfg, rv.count_tokens)
            rendered = rv.render(built.messages)
            ids = rv.tokenize(rendered, add_special=True, parse_special=True)
            jtext = _jinja_render(template, built.messages, bos_text)
            jids = rv.tokenize(jtext, add_special=False, parse_special=True)
            kept = set(range(1, built.n_lines_kept + 1))
            missing = [x for x in t.expected_line_ids if x not in kept]
            if missing:
                fail(f"{t.id}: expected lines {missing} were cut by truncation")
            prompts.append({
                "id": t.id, "type": t.type, "document": t.document, "question": t.question, "checks": t.checks,
                "prompt_version": built.prompt_version, "messages": built.messages, "rendered": rendered,
                "n_prompt_tokens": len(ids), "prompt_token_ids": ids, "token_budget": built.token_budget,
                "n_lines_total": built.n_lines_total, "n_lines_kept": built.n_lines_kept,
                "n_lines_omitted": built.n_lines_omitted, "line_map_golden_index": built.line_map,
                "expected_line_ids": t.expected_line_ids, "expected_text": t.expected_text,
                "expect_not_in_document": t.expect_not_in_document,
                "jinja_crosscheck": {"identical_token_ids": ids == jids, "first_diff_index": _first_diff(ids, jids),
                                     "jinja_rendered_sha256": sha256_text(jtext)},
            })
    c.golden_out.mkdir(parents=True, exist_ok=True)
    write_json(c.golden_out / "prompts.json", {
        "_note": "Fully built G0 prompts (Phase 7a format). messages -> llama_chat_apply_template (GGUF template) "
                 "-> llama_tokenize(add_special, parse_special) gives prompt_token_ids. The Kotlin prompt builder "
                 "must reproduce 'messages' exactly; G1/7d replay these.",
        "gemma_config_version": c.config["config_version"], "prompt_version": cfg.version,
        "llama_cpp": {"tag": c.pin["tag"], "commit": c.pin["commit"]},
        "chat_template_sha256": sha256_text(template), "bos_token": bos_text, "prompts": prompts})
    bad = [p["id"] for p in prompts if not p["jinja_crosscheck"]["identical_token_ids"]]
    if bad:
        log(f"WARNING: llama.cpp template and the GGUF Jinja template give different tokens for {bad}")

    outputs = {}
    for name, model in (("bf16", c.bf16), ("q4_0", c.our_q4)):
        log(f"greedy generation, {name}, {len(prompts)} prompts, max {max_new} new tokens")
        t_start = time.monotonic()
        runner = c.runner(model, f"golden_{name}.log")
        rows = []
        try:
            for p in prompts:
                r = runner.generate(p["messages"], max_new)
                if r["prompt_tokens"] != p["prompt_token_ids"]:
                    fail(f"{p['id']}: {name} tokenized the prompt differently from the fixture")
                rows.append({"id": p["id"], "tokens": r["tokens"], "text": r["text"], "n_tokens": r["n_tokens"],
                             "stop_reason": r["stop_reason"], "n_prompt_tokens": r["n_prompt_tokens"],
                             "timings": r["timings"]})
                log(f"  {name} {p['id']}: {r['n_tokens']} tokens, prompt {r['timings']['prompt_tokens_per_s']:.1f} t/s, "
                    f"decode {r['timings']['decode_tokens_per_s']:.2f} t/s")
        finally:
            peak = runner.close()
        outputs[name] = {
            "model": str(model), "model_sha256": c.step("quantize")["sha256"] if name == "q4_0" else c.step("convert")["sha256"],
            "llama_cpp": {"tag": c.pin["tag"], "commit": c.pin["commit"]},
            "runtime": {"n_ctx": c.runtime("n_ctx"), "n_batch": c.runtime("n_batch"), "n_ubatch": c.runtime("n_ubatch"),
                        "threads": c.threads, "flash_attn": c.runtime("flash_attn"), "swa_full": c.runtime("swa_full"),
                        "load_mode": c.runtime("load_mode"), "use_extra_bufts": c.runtime("use_extra_bufts"),
                        "sampler": "greedy", "max_new_tokens": max_new},
            "load_model_ms": runner.ready["load_model_ms"], "init_context_ms": runner.ready["init_context_ms"],
            "system_info": runner.ready["system_info"], "peak_rss_bytes": peak,
            "seconds": round(time.monotonic() - t_start, 1), "outputs": rows}
        write_json(c.golden_out / f"outputs_{name}.json", outputs[name])
    c.save("golden", {"prompts_file": str(c.golden_out / "prompts.json"),
                      "jinja_mismatch": bad,
                      "peak_rss_bytes": {k: v["peak_rss_bytes"] for k, v in outputs.items()},
                      "seconds": {k: v["seconds"] for k, v in outputs.items()}})


_CITE = re.compile(r"\[L(\d+)\]")


def _answer_checks(text: str, p: dict) -> dict:
    cited = [int(x) for x in _CITE.findall(text)]
    valid = [x for x in cited if 1 <= x <= p["n_lines_kept"]]
    return {"citations": cited, "invalid_citations": sorted(set(cited) - set(valid)),
            "cites_expected_line": bool(set(cited) & set(p["expected_line_ids"])) if p["expected_line_ids"] else None,
            "expected_text_verbatim": {s: (s in text) for s in p["expected_text"]},
            "general_guidance_marked": "General guidance:" in text}


def _side_by_side(prompts: list[dict], bf16: dict, q4: dict) -> tuple[str, list[dict]]:
    by_b = {r["id"]: r for r in bf16["outputs"]}
    by_q = {r["id"]: r for r in q4["outputs"]}
    lines = ["# G0 golden prompts: BF16 vs our Q4_0 (greedy)", "",
             "Hints (`expected_*`) are for hand review, not pass/fail. First differing token: index into the "
             "generated tokens (none = identical).", ""]
    rows = []
    for p in prompts:
        b, q = by_b[p["id"]], by_q[p["id"]]
        fd = _first_diff(b["tokens"], q["tokens"])
        cb, cq = _answer_checks(b["text"], p), _answer_checks(q["text"], p)
        rows.append({"id": p["id"], "first_diff_token": fd, "bf16_tokens": b["n_tokens"], "q4_0_tokens": q["n_tokens"],
                     "bf16_checks": cb, "q4_0_checks": cq})
        lines += [f"## {p['id']} · {p['type']} · {p['document']}", "",
                  f"**Question:** {p['question']}  ", f"**Checks:** {p['checks']}  ",
                  f"**Prompt:** {p['n_prompt_tokens']} tokens, {p['n_lines_kept']}/{p['n_lines_total']} lines kept  ",
                  f"**Expected lines:** {p['expected_line_ids'] or '-'} · **expected text:** "
                  f"{p['expected_text'] or ('(not in document)' if p['expect_not_in_document'] else '-')}  ",
                  f"**First differing token:** {'none (identical)' if fd is None else fd}", "",
                  "| | BF16 | our Q4_0 |", "|---|---|---|",
                  f"| tokens / stop | {b['n_tokens']} / {b['stop_reason']} | {q['n_tokens']} / {q['stop_reason']} |",
                  f"| citations | {cb['citations']} | {cq['citations']} |",
                  f"| invalid citations | {cb['invalid_citations'] or '-'} | {cq['invalid_citations'] or '-'} |",
                  f"| cites an expected line | {cb['cites_expected_line']} | {cq['cites_expected_line']} |",
                  f"| expected text verbatim | {cb['expected_text_verbatim'] or '-'} | {cq['expected_text_verbatim'] or '-'} |",
                  "", "**BF16:**", "", "```text", b["text"], "```", "", "**Our Q4_0:**", "", "```text", q["text"], "```", ""]
    return "\n".join(lines), rows


def step_report(c: Ctx) -> None:
    s = {k: c.step(k) for k in STEPS}
    q = s["quantize"]
    manifest = {
        "_note": "Our Gemma 3 4B IT QAT Q4_0 GGUF, produced by tools/gemma/g0_pipeline.py. G1 bundles this file as "
                 "an asset and verifies the imported model's SHA-256 and byte size against it.",
        "file": c.our_q4.name, "sha256": q["sha256"], "bytes": q["bytes"], "quant_type": "Q4_0",
        "token_embd_type": q["summary"]["token_embd_type"], "tensor_types": q["summary"]["tensor_types"],
        "quantize_command": q["command"],
        "llama_cpp": {"tag": c.pin["tag"], "commit": c.pin["commit"]},
        "source": {"repo": s["download"]["repo"], "revision": s["download"]["revision"]},
        "bf16_intermediate": {"sha256": s["convert"]["sha256"], "bytes": s["convert"]["bytes"],
                              "command": s["convert"]["command"]},
        "gemma_config_version": c.config["config_version"],
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    write_json(c.out / "gemma_manifest.json", manifest)

    prompts = load_json(c.golden_out / "prompts.json")["prompts"]
    bf16, q4 = load_json(c.golden_out / "outputs_bf16.json"), load_json(c.golden_out / "outputs_q4_0.json")
    md_sbs, rows = _side_by_side(prompts, bf16, q4)
    (c.golden_out / "side_by_side.md").write_text(md_sbs + "\n", encoding="utf-8")
    ins = load_json(Path(s["inspect"]["inspect_file"]))
    ppl = s["perplexity"]

    identical = sum(1 for r in rows if r["first_diff_token"] is None)
    fds = sorted(r["first_diff_token"] for r in rows if r["first_diff_token"] is not None)

    def stat(model: str, key: str) -> str:
        vals = [r[f"{model}_checks"][key] for r in rows if r[f"{model}_checks"][key] is not None]
        return f"{sum(1 for v in vals if v)}/{len(vals)}"

    def verbatim(model: str) -> str:
        vals = [ok for r in rows for ok in r[f"{model}_checks"]["expected_text_verbatim"].values()]
        return f"{sum(vals)}/{len(vals)}"

    def invalid(model: str) -> int:
        return sum(1 for r in rows if r[f"{model}_checks"]["invalid_citations"])

    size_rows = [("HF safetensors (all files)", sum(f["bytes"] for f in s["download"]["files"] if f["file"].endswith(".safetensors"))),
                 ("BF16 GGUF (text only)", s["convert"]["bytes"]), ("our Q4_0", q["bytes"]),
                 ("official Q4_0", s["official"]["bytes"])]
    if s["variant"].get("bytes"):
        size_rows.append((f"variant (our Q4_0, token_embd {s['variant']['token_embd_type']})", s["variant"]["bytes"]))
    tdiff = ins["ours_vs_official_types"]
    ge = ins["grid_exactness_vs_bf16"]
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    md = [
        "# Phase G0 report: Gemma 3 4B IT QAT -> Q4_0", "", f"Generated {now} by tools/gemma/g0_pipeline.py.", "",
        "## Sources", "",
        f"* Source checkpoint: `{s['download']['repo']}` @ `{s['download']['revision']}`",
        f"* Official cross-check: `{s['official']['repo']}` / `{s['official'].get('file', '')}` @ `{s['official']['revision']}`",
        f"* llama.cpp: `{c.pin['tag']}` (`{c.pin['commit']}`), built with {s['llama']['cmake_generator']}, "
        f"GGML_NATIVE={s['llama']['ggml_native']}",
        f"* Machine: {s['check']['platform']}, {s['check']['physical_cores']} physical cores, "
        f"RAM {human_bytes(s['check'].get('ram_total_bytes'))}, Python {s['check']['python']}", "",
        "## Files", "", "| file | size | sha256 |", "|---|---|---|",
    ]
    shas = {"BF16 GGUF (text only)": s["convert"]["sha256"], "our Q4_0": q["sha256"], "official Q4_0": s["official"]["sha256"]}
    for label, n in size_rows:
        md.append(f"| {label} | {human_bytes(n)} ({n:,} B) | {shas.get(label, s['variant'].get('sha256', '-') if 'variant' in label else '-')} |")
    md += ["", "## Tensor types", "", "| file | token_embd | types (tensor count) |", "|---|---|---|"]
    for k, sm in ins["summaries"].items():
        md.append(f"| {k} | {sm['token_embd_type']} | " +
                  ", ".join(f"{t}: {v['tensors']}" for t, v in sm["tensor_types"].items()) + " |")
    md += ["", f"Ours vs official: {len(tdiff['type_differences'])} tensors differ in type "
               f"({tdiff['type_difference_counts'] or 'none'}); only in ours: {tdiff['only_in_a'] or 'none'}; "
               f"only in official: {tdiff['only_in_b'] or 'none'}; shape differences: {len(tdiff['shape_differences'])}.",
           f"Metadata values that differ (ours vs official): {list(ins['ours_vs_official_metadata']['differing_values']) or 'none'}.",
           f"Tokenizer identical: BF16 = ours: {s['inspect']['tokenizer_same_bf16_ours']}; ours = official: "
           f"{s['inspect']['tokenizer_same_ours_official']}.", "",
           "## Grid exactness vs our BF16", "",
           "Share of weights that dequantize to exactly the BF16 value. QAT weights can in principle be reproduced "
           "exactly; a low value means llama-quantize chose different Q4_0 scales than the QAT grid. The official "
           "row is only meaningful if the official file was made from the same checkpoint as our BF16.", "",
           "| file | exact | rel. RMSE | token_embd | attn | ffn |", "|---|---|---|---|---|---|"]
    for k, r in ge.items():
        bg = r["by_group"]
        attn = [bg[g]["exact_fraction"] for g in ("attn_q", "attn_k", "attn_v", "attn_output") if g in bg]
        ffn = [bg[g]["exact_fraction"] for g in ("ffn_gate", "ffn_up", "ffn_down") if g in bg]
        te = bg.get("token_embd", {}).get("exact_fraction")
        md.append(f"| {k} | {r['overall_exact_fraction']:.2%} | {r['overall_rel_rmse']:.3e} | "
                  f"{'-' if te is None else f'{te:.2%}'} | {f'{min(attn):.2%}-{max(attn):.2%}' if attn else '-'} | "
                  f"{f'{min(ffn):.2%}-{max(ffn):.2%}' if ffn else '-'} |")
    md += ["", "## Perplexity (greedy-independent; same text for every model)", "",
           f"Text: `{ppl['text'].get('member', ppl['text']['path'])}` sha256 `{ppl['text']['sha256']}`; "
           f"ctx {ppl['params']['ctx']}, {ppl['params']['chunks']} chunks, {ppl['params']['threads']} threads, CPU.", "",
           "| model | PPL | +/- | vs BF16 | peak RAM | time |", "|---|---|---|---|---|---|"]
    for k, r in ppl["results"].items():
        md.append(f"| {k} | {r['ppl']:.4f} | {r['ppl_err']:.4f} | {r['vs_bf16_percent']:+.2f}% | "
                  f"{human_bytes(r['peak_rss_bytes'])} | {r['seconds']:.0f} s |")
    md += ["", ("**FLAG: our Q4_0 is more than 5% worse than BF16.**" if ppl["flag_ours_more_than_5_percent_worse"]
                else "Our Q4_0 is within 5% of BF16."), "",
           "## Golden prompts (20, greedy)", "",
           f"* Identical BF16 vs Q4_0 outputs: {identical}/{len(rows)}; first differing token index for the others: {fds or '-'}",
           f"* Answers citing at least one expected line: BF16 {stat('bf16', 'cites_expected_line')}, "
           f"Q4_0 {stat('q4_0', 'cites_expected_line')}",
           f"* Expected text copied verbatim: BF16 {verbatim('bf16')}, Q4_0 {verbatim('q4_0')}",
           f"* Answers with an invalid citation: BF16 {invalid('bf16')}, Q4_0 {invalid('q4_0')}",
           f"* llama.cpp template vs GGUF Jinja template: {'identical tokens for all prompts' if not s['golden']['jinja_mismatch'] else 'DIFFERENT for ' + str(s['golden']['jinja_mismatch'])}",
           f"* Peak RAM: BF16 {human_bytes(bf16['peak_rss_bytes'])}, Q4_0 {human_bytes(q4['peak_rss_bytes'])} "
           f"(runtime {json.dumps(q4['runtime'])})", "",
           "Side-by-side answers: `golden_llm/side_by_side.md`.", "",
           "| id | type | prompt tok | BF16 tok | Q4_0 tok | first diff | Q4_0 prompt t/s | Q4_0 decode t/s |",
           "|---|---|---|---|---|---|---|---|"]
    pq = {r["id"]: r for r in q4["outputs"]}
    for p, r in zip(prompts, rows):
        tq = pq[p["id"]]["timings"]
        md.append(f"| {p['id']} | {p['type']} | {p['n_prompt_tokens']} | {r['bf16_tokens']} | {r['q4_0_tokens']} | "
                  f"{'-' if r['first_diff_token'] is None else r['first_diff_token']} | "
                  f"{tq['prompt_tokens_per_s']:.1f} | {tq['decode_tokens_per_s']:.2f} |")
    md += ["", "## Peak RAM on this desktop", "", "| run | peak RAM |", "|---|---|",
           f"| convert_hf_to_gguf | {human_bytes(s['convert']['peak_rss_bytes'])} |",
           f"| llama-quantize | {human_bytes(q['peak_rss_bytes'])} |"]
    md += [f"| perplexity {k} | {human_bytes(r['peak_rss_bytes'])} |" for k, r in ppl["results"].items()]
    md += [f"| golden generation BF16 | {human_bytes(bf16['peak_rss_bytes'])} |",
           f"| golden generation Q4_0 | {human_bytes(q4['peak_rss_bytes'])} |", ""]
    (c.out / "g0_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    write_json(c.out / "g0_report.json", {"generated": now, "steps": s, "golden_rows": rows})
    c.save("report", {"report": str(c.out / "g0_report.md")})
    log(f"report written: {c.out / 'g0_report.md'}")


# ---------------------------------------------------------------- main

STEP_FUNCS = {"check": step_check, "llama": step_llama, "download": step_download, "convert": step_convert,
              "quantize": step_quantize, "official": step_official, "variant": step_variant,
              "inspect": step_inspect, "perplexity": step_perplexity, "golden": step_golden, "report": step_report}


def main(argv: list[str] | None = None) -> None:
    sources = load_json(SOURCES_FILE)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", default=["all"], help=f"'all' (default) or any of: {' '.join(STEPS)}")
    ap.add_argument("--work-dir", type=Path, default=GEMMA_TOOLS / "work",
                    help="big intermediate files (~30 GB): HF download, BF16 GGUF, official GGUF, llama.cpp build")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where our Q4_0, the manifest, golden_llm outputs and the report go "
                         "(default: tools/reference/gemma)")
    ap.add_argument("--source-repo", default=None, help=f"default: {sources['source_repo']} (gemma_sources.json)")
    ap.add_argument("--source-revision", default="main", help="HF revision to download (a commit sha pins it)")
    ap.add_argument("--official-revision", default="main")
    ap.add_argument("--threads", type=int, default=None, help="default: physical core count")
    ap.add_argument("--jobs", type=int, default=None, help="parallel compile jobs (default: logical cores)")
    ap.add_argument("--ppl-chunks", type=int, default=150, help="perplexity chunks (default 150 x 512 tokens)")
    ap.add_argument("--ppl-ctx", type=int, default=512)
    ap.add_argument("--skip-variant", action="store_true", help="do not build the token_embd comparison variant")
    ap.add_argument("--skip-disk-check", action="store_true")
    ap.add_argument("--force", action="store_true", help="redo the named steps even if state.json says done")
    ap.add_argument("--llama-cpp-dir", type=Path, default=None, help="existing llama.cpp checkout at the pinned commit")
    ap.add_argument("--local-source-dir", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--local-official-gguf", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--perplexity-text", type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    steps = STEPS if args.steps == ["all"] else args.steps
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        ap.error(f"unknown step(s) {unknown}; choose from {STEPS}")
    c = Ctx(args)
    c.work.mkdir(parents=True, exist_ok=True)
    for name in steps:
        if c.done(name) and name != "check":
            log(f"{name}: already done (use --force to redo)")
            continue
        t0 = time.monotonic()
        log(f"== {name}")
        STEP_FUNCS[name](c)
        log(f"== {name} done in {time.monotonic() - t0:.0f} s")


if __name__ == "__main__":
    main()
