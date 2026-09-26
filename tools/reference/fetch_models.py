#!/usr/bin/env python3
"""Phase 0, step 4: download the official PP-OCRv6 small ONNX models, extract
the character dictionary and configs, and print each model's I/O.

Outputs (under --models-dir, default tools/reference/models/):
  det/inference.onnx, det/inference.yml
  rec/inference.onnx, rec/inference.yml
  rec/ppocr_keys.txt        character_dict from rec/inference.yml, one per line
  manifest.json             source, revision, sha256 and size of every file
  model_io.json             ONNX input/output names, shapes, dtypes, opsets
  preprocess_config.json    every pre/post-processing value + where it came from

Nothing is guessed: if a required file or config key is missing the script stops.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

from refcommon import (
    DEFAULT_MODELS_DIR,
    OFFICIAL_ONNX_MODELS,
    ConfigError,
    read_det_config,
    read_rec_config,
    sha256_file,
    write_json,
)

REQUIRED_FILES = ("inference.onnx", "inference.yml")


def download_hf(repo_id: str, dest: Path, revision: str | None) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(repo_id, revision=revision)
    snapshot_download(repo_id=repo_id, revision=info.sha, local_dir=str(dest))
    return {"type": "huggingface", "repo_id": repo_id, "revision": info.sha}


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if dest not in target.parents and target != dest:
            raise RuntimeError(f"Refusing to extract {member.name!r} outside {dest}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"Refusing to extract link {member.name!r}")
    tar.extractall(dest)


def download_bos(url: str, dest: Path) -> dict:
    import requests

    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / Path(url).name
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tar_path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        tar_sha = sha256_file(tar_path)
        extract_dir = Path(td) / "x"
        extract_dir.mkdir()
        with tarfile.open(tar_path) as tar:
            _safe_extract(tar, extract_dir)
        # The tarball wraps files in one directory; locate the ONNX model rather
        # than assuming the directory name.
        onnx_files = list(extract_dir.rglob("inference.onnx"))
        if len(onnx_files) != 1:
            raise RuntimeError(
                f"{url}: expected exactly one inference.onnx, found {onnx_files}"
            )
        src_dir = onnx_files[0].parent
        dest.mkdir(parents=True, exist_ok=True)
        for p in src_dir.iterdir():
            if p.is_file():
                shutil.copy2(p, dest / p.name)
    return {"type": "bos", "url": url, "archive_sha256": tar_sha}


def fetch_one(role: str, dest: Path, source: str, revision: str | None) -> dict:
    spec = OFFICIAL_ONNX_MODELS[role]
    if dest.exists():
        shutil.rmtree(dest)
    errors = []
    order = {"auto": ["hf", "bos"], "hf": ["hf"], "bos": ["bos"]}[source]
    for src in order:
        try:
            print(f"[{role}] downloading from {src} ...", flush=True)
            if src == "hf":
                origin = download_hf(spec["hf_repo"], dest, revision)
            else:
                origin = download_bos(spec["bos_url"], dest)
            break
        except Exception as e:  # report every failure; don't silently fall through
            errors.append(f"{src}: {type(e).__name__}: {e}")
            print(f"[{role}]   failed: {errors[-1]}", flush=True)
            if dest.exists():
                shutil.rmtree(dest)
    else:
        raise SystemExit(
            f"Could not download the {role} model from any source:\n  "
            + "\n  ".join(errors)
        )

    missing = [f for f in REQUIRED_FILES if not (dest / f).is_file()]
    if missing:
        found = sorted(p.name for p in dest.iterdir())
        raise SystemExit(
            f"[{role}] downloaded package is missing {missing}. Files present: {found}. "
            "Stopping instead of guessing."
        )
    files = {}
    for p in sorted(dest.rglob("*")):
        if p.is_file() and ".cache" not in p.parts:
            files[str(p.relative_to(dest))] = {
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            }
    return {"origin": origin, "files": files}


def describe_onnx(path: Path) -> dict:
    import onnx
    import onnxruntime as ort

    model = onnx.load(str(path), load_external_data=False)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def meta(args):
        return [
            {"name": a.name, "shape": [d if d is not None else "?" for d in a.shape], "type": a.type}
            for a in args
        ]

    return {
        "file": str(path.name),
        "ir_version": model.ir_version,
        "opset_import": {(o.domain or "ai.onnx"): o.version for o in model.opset_import},
        "producer": f"{model.producer_name} {model.producer_version}".strip(),
        "inputs": meta(sess.get_inputs()),
        "outputs": meta(sess.get_outputs()),
    }


def write_dictionary(chars: list[str], path: Path) -> None:
    for i, c in enumerate(chars):
        if "\n" in c or "\r" in c:
            raise ConfigError(f"character_dict[{i}] contains a newline; cannot write one-per-line")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(chars))
        f.write("\n")
    # Round-trip check: the Android side will read this file line by line.
    with open(path, "r", encoding="utf-8", newline="\n") as f:
        back = f.read().split("\n")[:-1]
    if back != chars:
        raise ConfigError(f"{path} does not round-trip to the same character list")


def print_sourced(title: str, cfg: dict) -> None:
    print(f"\n=== {title} ===")
    for key, s in cfg.items():
        val = s.value
        if key == "character_dict":
            val = f"<{len(val)} entries; first 10: {val[:10]!r}; last 3: {val[-3:]!r}>"
        print(f"  {key:26s} = {val!r}")
        print(f"  {'':26s}   source: {s.source}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    ap.add_argument("--source", choices=["auto", "hf", "bos"], default="auto",
                    help="auto = Hugging Face first, then Baidu BOS")
    ap.add_argument("--hf-revision", default=None,
                    help="Pin a Hugging Face revision (commit sha). Default: latest; the sha used is recorded.")
    ap.add_argument("--skip-download", action="store_true",
                    help="Only inspect models already in --models-dir")
    args = ap.parse_args()

    models_dir: Path = args.models_dir.resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = models_dir / "manifest.json"

    if args.skip_download:
        if not manifest_path.exists():
            raise SystemExit(f"--skip-download given but {manifest_path} does not exist")
        import json
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}
        for role in ("det", "rec"):
            manifest[role] = fetch_one(role, models_dir / role, args.source, args.hf_revision)
            manifest[role]["official_source"] = OFFICIAL_ONNX_MODELS[role]

    det_dir, rec_dir = models_dir / "det", models_dir / "rec"
    det_cfg = read_det_config(det_dir)
    rec_cfg = read_rec_config(rec_dir)

    for role, cfg in (("det", det_cfg), ("rec", rec_cfg)):
        expected = OFFICIAL_ONNX_MODELS[role]["expected_model_name"]
        if cfg["model_name"].value != expected:
            raise SystemExit(
                f"{role}/inference.yml says model_name={cfg['model_name'].value!r}, "
                f"expected {expected!r}. Stopping."
            )

    chars = rec_cfg["character_dict"].value
    dict_path = rec_dir / "ppocr_keys.txt"
    write_dictionary(chars, dict_path)
    manifest["rec"]["files"]["ppocr_keys.txt"] = {
        "sha256": sha256_file(dict_path),
        "bytes": dict_path.stat().st_size,
        "derived_from": "rec/inference.yml PostProcess.character_dict",
    }

    io = {"det": describe_onnx(det_dir / "inference.onnx"),
          "rec": describe_onnx(rec_dir / "inference.onnx")}

    # Consistency check: rec output class dimension must equal blank + dict (+ space).
    rec_out_shape = io["rec"]["outputs"][0]["shape"]
    n_classes = rec_cfg["ctc_classes"].value
    last = rec_out_shape[-1]
    if isinstance(last, int):
        ok = last == n_classes
        io["rec"]["class_dim_check"] = {
            "onnx_output_last_dim": last,
            "expected_from_dict": n_classes,
            "dict_entries": len(chars),
            "passed": ok,
        }
        if not ok:
            print(f"\n!! rec output last dim {last} != blank+dict(+space) {n_classes}. "
                  "Dictionary alignment is wrong; do not proceed.", file=sys.stderr)
    else:
        io["rec"]["class_dim_check"] = {
            "onnx_output_last_dim": last,
            "expected_from_dict": n_classes,
            "passed": None,
            "note": "class dim is symbolic in the graph; checked at runtime by run_reference.py",
        }

    write_json(manifest_path, manifest)
    write_json(models_dir / "model_io.json", io)
    write_json(
        models_dir / "preprocess_config.json",
        {
            "det": {k: v.to_json() for k, v in det_cfg.items()},
            "rec": {k: v.to_json() for k, v in rec_cfg.items() if k != "character_dict"}
            | {"character_dict": {"value": f"{len(chars)} entries -> rec/ppocr_keys.txt",
                                  "source": rec_cfg["character_dict"].source}},
        },
    )

    print("\n=== Model files ===")
    for role in ("det", "rec"):
        print(f"  [{role}] origin: {manifest[role]['origin']}")
        for name, meta in manifest[role]["files"].items():
            print(f"      {name:22s} {meta['bytes']:>12,d} B  sha256={meta['sha256']}")

    print("\n=== ONNX model I/O ===")
    for role in ("det", "rec"):
        m = io[role]
        print(f"  [{role}] ir_version={m['ir_version']} opset={m['opset_import']} producer={m['producer']!r}")
        for kind in ("inputs", "outputs"):
            for a in m[kind]:
                print(f"      {kind[:-1]:6s} name={a['name']!r:20s} shape={a['shape']} type={a['type']}")
    print(f"  [rec] class-dim check: {io['rec']['class_dim_check']}")

    print_sourced("Detection config (det/inference.yml)", det_cfg)
    print_sourced("Recognition config (rec/inference.yml + PaddleX)", rec_cfg)

    print(f"\nWrote {manifest_path}, {models_dir/'model_io.json'}, "
          f"{models_dir/'preprocess_config.json'}, {dict_path}")
    check = io["rec"]["class_dim_check"].get("passed")
    return 1 if check is False else 0


if __name__ == "__main__":
    sys.exit(main())
