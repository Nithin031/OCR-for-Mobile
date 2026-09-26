"""GGUF checks and comparisons for G0, using gguf-py from the pinned llama.cpp checkout."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

K_QUANT_PREFIXES = ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "IQ", "TQ")
VISION_MARKERS = ("v.", "mm.", "vision", "mmproj")
_gguf = None


def import_gguf(llama_cpp_dir: Path):
    """Import gguf-py from the pinned checkout (not from PyPI), same code the converter uses."""
    global _gguf
    if _gguf is None:
        sys.path.insert(0, str(Path(llama_cpp_dir) / "gguf-py"))
        import gguf  # noqa: PLC0415
        _gguf = gguf
    return _gguf


class GgufFile:
    def __init__(self, path: Path):
        if _gguf is None:
            raise RuntimeError("call import_gguf() first")
        self.path = Path(path)
        self.reader = _gguf.GGUFReader(str(path))
        self.tensors = {t.name: t for t in self.reader.tensors}

    def field(self, key: str, default: Any = None) -> Any:
        f = self.reader.fields.get(key)
        return f.contents() if f is not None else default

    def type_name(self, name: str) -> str:
        return self.tensors[name].tensor_type.name

    def shape(self, name: str) -> list[int]:
        return [int(x) for x in self.tensors[name].shape]

    def summary(self) -> dict:
        by_type: dict[str, dict[str, int]] = {}
        n_params = 0
        for t in self.tensors.values():
            e = by_type.setdefault(t.tensor_type.name, {"tensors": 0, "elements": 0, "bytes": 0})
            e["tensors"] += 1
            e["elements"] += int(t.n_elements)
            e["bytes"] += int(t.n_bytes)
            n_params += int(t.n_elements)
        return {
            "architecture": self.field("general.architecture"),
            "file_type": self.field("general.file_type"),
            "n_tensors": len(self.tensors),
            "n_params": n_params,
            "tensor_types": dict(sorted(by_type.items())),
            "token_embd_type": self.type_name("token_embd.weight") if "token_embd.weight" in self.tensors else None,
            "has_output_weight": "output.weight" in self.tensors,
        }


def _quantizable(name: str, n_dims: int) -> bool:
    # mirrors llama.cpp b11201 tensor_allows_quantization() for a dense Gemma 3 model
    return n_dims >= 2 and name.endswith("weight") and "_norm.weight" not in name


def vision_tensors(g: GgufFile) -> list[str]:
    return [n for n in g.tensors if n.startswith(VISION_MARKERS[:2]) or any(m in n for m in VISION_MARKERS[2:])]


def check_bf16(g: GgufFile) -> list[str]:
    problems = []
    for name, t in g.tensors.items():
        want = "BF16" if len(t.shape) >= 2 else "F32"
        if t.tensor_type.name != want:
            problems.append(f"{name}: {t.tensor_type.name}, expected {want}")
    return problems


def check_q4_0(g: GgufFile, token_embd_type: str = "Q4_0") -> list[str]:
    problems = []
    for name, t in g.tensors.items():
        tn = t.tensor_type.name
        if tn.startswith(K_QUANT_PREFIXES):
            problems.append(f"{name}: {tn} is a K/IQ/TQ quant")
            continue
        if _quantizable(name, len(t.shape)):
            want = token_embd_type if name == "token_embd.weight" else "Q4_0"
        else:
            want = "F32"
        if tn != want:
            problems.append(f"{name}: {tn}, expected {want}")
    imatrix_keys = [k for k in g.reader.fields if k.startswith("quantize.imatrix")]
    if imatrix_keys:
        problems.append(f"importance-matrix metadata present: {imatrix_keys}")
    return problems


def compare_types(a: GgufFile, b: GgufFile) -> dict:
    names = sorted(set(a.tensors) | set(b.tensors))
    only_a = [n for n in names if n not in b.tensors]
    only_b = [n for n in names if n not in a.tensors]
    type_diff, shape_diff = [], []
    for n in names:
        if n in a.tensors and n in b.tensors:
            if a.type_name(n) != b.type_name(n):
                type_diff.append({"tensor": n, "a": a.type_name(n), "b": b.type_name(n)})
            if a.shape(n) != b.shape(n):
                shape_diff.append({"tensor": n, "a": a.shape(n), "b": b.shape(n)})
    pairs: dict[str, int] = {}
    for d in type_diff:
        key = f"{d['a']} -> {d['b']}"
        pairs[key] = pairs.get(key, 0) + 1
    return {"only_in_a": only_a, "only_in_b": only_b, "type_differences": type_diff,
            "type_difference_counts": pairs, "shape_differences": shape_diff}


_META_KEYS = [
    "general.architecture", "general.file_type", "general.quantization_version",
    "gemma3.context_length", "gemma3.embedding_length", "gemma3.block_count", "gemma3.feed_forward_length",
    "gemma3.attention.head_count", "gemma3.attention.head_count_kv", "gemma3.attention.key_length",
    "gemma3.attention.value_length", "gemma3.attention.sliding_window", "gemma3.rope.freq_base",
    "gemma3.attention.layer_norm_rms_eps", "tokenizer.ggml.model", "tokenizer.ggml.pre",
    "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
    "tokenizer.ggml.padding_token_id", "tokenizer.ggml.unknown_token_id",
    "tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token", "tokenizer.ggml.add_space_prefix",
]


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    return v


def metadata(g: GgufFile) -> dict:
    out = {k: _jsonable(g.field(k)) for k in _META_KEYS}
    tmpl = g.field("tokenizer.chat_template")
    out["tokenizer.chat_template.sha256"] = hashlib.sha256(tmpl.encode("utf-8")).hexdigest() if tmpl else None
    return out


def compare_metadata(a: GgufFile, b: GgufFile) -> dict:
    ma, mb = metadata(a), metadata(b)
    differing = {k: {"a": ma[k], "b": mb[k]} for k in ma if ma[k] != mb[k]}
    ka, kb = set(a.reader.fields), set(b.reader.fields)
    ignore_prefix = ("GGUF.",)
    return {
        "differing_values": differing,
        "keys_only_in_a": sorted(k for k in ka - kb if not k.startswith(ignore_prefix)),
        "keys_only_in_b": sorted(k for k in kb - ka if not k.startswith(ignore_prefix)),
    }


def tokenizer_fingerprint(g: GgufFile) -> str:
    h = hashlib.sha256()
    for key in ("tokenizer.ggml.model", "tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.scores",
                "tokenizer.ggml.merges", "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
                "tokenizer.ggml.eot_token_id", "tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token",
                "tokenizer.ggml.add_space_prefix"):
        v = g.field(key)
        if isinstance(v, list) and v and isinstance(v[0], float):
            v = np.asarray(v, dtype=np.float32).tobytes().hex()
        h.update(key.encode())
        h.update(json.dumps(_jsonable(v), ensure_ascii=True).encode())
    return h.hexdigest()


def grid_exactness(q: GgufFile, ref: GgufFile, max_elems_per_chunk: int = 1 << 24) -> dict:
    """For every tensor whose type differs from the reference file: dequantize both and compare.

    exact_fraction: share of weights reproduced bit-exactly (as float32) from the BF16 reference.
    A QAT checkpoint whose weights sit on the Q4_0 grid can in principle be reproduced exactly;
    a low value means the quantizer picked different scales than the QAT grid.
    """
    per_tensor = []
    tot_n = tot_exact = 0
    tot_se = tot_ss = 0.0
    for name, t in q.tensors.items():
        r = ref.tensors.get(name)
        if r is None or len(t.shape) < 2 or t.tensor_type == r.tensor_type:
            continue
        if [int(x) for x in t.shape] != [int(x) for x in r.shape]:
            per_tensor.append({"tensor": name, "error": "shape differs from reference"})
            continue
        qd = t.data.reshape(-1, t.data.shape[-1])
        rd = r.data.reshape(-1, r.data.shape[-1])
        ncols = int(t.shape[0])
        rows = qd.shape[0]
        step = max(1, max_elems_per_chunk // ncols)
        n = exact = 0
        se = ss = 0.0
        for i in range(0, rows, step):
            a = _gguf.quants.dequantize(np.asarray(qd[i:i + step]), t.tensor_type).astype(np.float32, copy=False)
            b = _gguf.quants.dequantize(np.asarray(rd[i:i + step]), r.tensor_type).astype(np.float32, copy=False)
            a = a.reshape(-1)
            b = b.reshape(-1)
            n += a.size
            exact += int(np.count_nonzero(a == b))
            d = (a - b).astype(np.float64)
            se += float(np.dot(d, d))
            bb = b.astype(np.float64)
            ss += float(np.dot(bb, bb))
        per_tensor.append({"tensor": name, "type": t.tensor_type.name, "ref_type": r.tensor_type.name,
                           "elements": n, "exact_fraction": exact / n if n else None,
                           "rel_rmse": (se / ss) ** 0.5 if ss > 0 else None})
        tot_n += n
        tot_exact += exact
        tot_se += se
        tot_ss += ss

    def group(name: str) -> str:
        if name == "token_embd.weight":
            return "token_embd"
        for g in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"):
            if f".{g}." in name:
                return g
        return "other"

    groups: dict[str, dict[str, float]] = {}
    for e in per_tensor:
        if "error" in e:
            continue
        g = groups.setdefault(group(e["tensor"]), {"elements": 0, "exact": 0.0})
        g["elements"] += e["elements"]
        g["exact"] += e["exact_fraction"] * e["elements"]
    return {
        "overall_exact_fraction": tot_exact / tot_n if tot_n else None,
        "overall_rel_rmse": (tot_se / tot_ss) ** 0.5 if tot_ss > 0 else None,
        "elements_compared": tot_n,
        "by_group": {k: {"elements": int(v["elements"]), "exact_fraction": v["exact"] / v["elements"]}
                     for k, v in sorted(groups.items())},
        "per_tensor": per_tensor,
    }
