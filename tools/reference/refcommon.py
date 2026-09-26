"""Shared helpers for the Phase 0 reference tools.

Everything here reads values from files on disk (the model's own inference.yml,
or the installed PaddleX source) and returns them together with a human-readable
"source" string, so every number that the Android port will later reuse can be
traced back to where it came from.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = HERE / "models"
DEFAULT_SAMPLES_DIR = HERE / "samples"
DEFAULT_GOLDEN_DIR = HERE / "golden"

EXPECTED_DET_MODEL_NAME = "PP-OCRv6_small_det"
EXPECTED_REC_MODEL_NAME = "PP-OCRv6_small_rec"

# Official ONNX exports of PP-OCRv6 small. Source for these locations:
# PaddleOCR repo, docs/version3.x/inference_deployment/cross_platform/
# android_deployment.en.md ("Prepare Models" table), which is the model list
# used by the official deploy/ppocr-android demo.
OFFICIAL_ONNX_MODELS = {
    "det": {
        "hf_repo": "PaddlePaddle/PP-OCRv6_small_det_onnx",
        "bos_url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/"
        "official_inference_model/paddle3.0.0/PP-OCRv6_small_det_onnx_infer.tar",
        "expected_model_name": EXPECTED_DET_MODEL_NAME,
    },
    "rec": {
        "hf_repo": "PaddlePaddle/PP-OCRv6_small_rec_onnx",
        "bos_url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/"
        "official_inference_model/paddle3.0.0/PP-OCRv6_small_rec_onnx_infer.tar",
        "expected_model_name": EXPECTED_REC_MODEL_NAME,
    },
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class ConfigError(RuntimeError):
    """A value the port depends on is missing or inconsistent. Never guessed."""


@dataclass
class Sourced:
    value: Any
    source: str

    def to_json(self) -> dict:
        return {"value": self.value, "source": self.source}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_images(samples_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in samples_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _find_op(transform_ops: list, op_name: str) -> tuple[int, dict] | None:
    for i, op in enumerate(transform_ops or []):
        if isinstance(op, dict) and op_name in op:
            return i, (op[op_name] or {})
    return None


def _pkg_relative(file: str) -> str:
    """'paddlex/…/x.py' instead of the absolute site-packages path."""
    marker = "site-packages/"
    return file.split(marker, 1)[1] if marker in file else file


def _find_line(obj, pattern: str) -> str:
    """'<file>:<line>' of the first source line of `obj` matching `pattern`."""
    obj = inspect.unwrap(obj)
    file = inspect.getsourcefile(obj)
    lines, start = inspect.getsourcelines(obj)
    rx = re.compile(pattern)
    for offset, text in enumerate(lines):
        if rx.search(text):
            return f"{_pkg_relative(file)}:{start + offset}  `{text.strip()}`"
    raise ConfigError(f"Pattern {pattern!r} not found in {_pkg_relative(file)}:{start}")


def read_det_config(det_dir: Path) -> dict[str, Sourced]:
    """Detection pre/post-processing values from the det model's inference.yml."""
    yml_path = det_dir / "inference.yml"
    cfg = load_yaml(yml_path)
    rel = f"{yml_path.parent.name}/{yml_path.name}"
    out: dict[str, Sourced] = {}

    out["model_name"] = Sourced(
        cfg.get("Global", {}).get("model_name"), f"{rel}: Global.model_name"
    )

    ops = cfg.get("PreProcess", {}).get("transform_ops", [])
    out["preprocess_ops"] = Sourced(
        [list(op.keys())[0] for op in ops], f"{rel}: PreProcess.transform_ops"
    )

    found = _find_op(ops, "DecodeImage")
    if found is None:
        raise ConfigError(f"{rel}: no DecodeImage op in PreProcess")
    i, args = found
    out["img_mode"] = Sourced(
        args.get("img_mode"), f"{rel}: PreProcess.transform_ops[{i}].DecodeImage.img_mode"
    )

    found = _find_op(ops, "DetResizeForTest")
    if found is not None:
        i, args = found
        # Whatever the model config says about resizing. The reference run
        # overrides limit_type/limit_side_len explicitly (see run_reference.py),
        # but we still report the model's own values.
        out["det_resize_config"] = Sourced(
            args, f"{rel}: PreProcess.transform_ops[{i}].DetResizeForTest"
        )

    found = _find_op(ops, "NormalizeImage")
    if found is None:
        raise ConfigError(f"{rel}: no NormalizeImage op in PreProcess")
    i, args = found
    for key in ("mean", "std", "scale", "order"):
        if key not in args:
            raise ConfigError(
                f"{rel}: PreProcess.transform_ops[{i}].NormalizeImage has no '{key}'"
            )
        val = args[key]
        src = f"{rel}: PreProcess.transform_ops[{i}].NormalizeImage.{key}"
        if key == "scale" and isinstance(val, str):
            # PaddleX evaluates string scales (e.g. "1./255.") with eval();
            # see NormalizeImage.__init__ in the installed paddlex.
            src += f" (string {val!r}, evaluated like PaddleX does)"
            val = float(eval(val, {"__builtins__": {}}, {}))  # noqa: S307
        out[f"normalize_{key}"] = Sourced(val, src)

    post = cfg.get("PostProcess", {})
    if post.get("name") != "DBPostProcess":
        raise ConfigError(f"{rel}: PostProcess.name is {post.get('name')!r}, expected DBPostProcess")
    for key in ("thresh", "box_thresh", "unclip_ratio", "max_candidates"):
        if key not in post:
            raise ConfigError(f"{rel}: PostProcess has no '{key}'")
        out[f"db_{key}"] = Sourced(post[key], f"{rel}: PostProcess.{key}")
    for key in ("use_dilation", "score_mode", "box_type"):
        if key in post:
            out[f"db_{key}"] = Sourced(post[key], f"{rel}: PostProcess.{key}")
        else:
            out[f"db_{key}"] = Sourced(
                None, f"{rel}: PostProcess.{key} absent -> PaddleX default applies"
            )
    return out


def read_rec_config(rec_dir: Path) -> dict[str, Sourced]:
    """Recognition values from inference.yml, plus the ones PaddleX hardcodes."""
    yml_path = rec_dir / "inference.yml"
    cfg = load_yaml(yml_path)
    rel = f"{yml_path.parent.name}/{yml_path.name}"
    out: dict[str, Sourced] = {}

    out["model_name"] = Sourced(
        cfg.get("Global", {}).get("model_name"), f"{rel}: Global.model_name"
    )
    ops = cfg.get("PreProcess", {}).get("transform_ops", [])
    out["preprocess_ops"] = Sourced(
        [list(op.keys())[0] for op in ops], f"{rel}: PreProcess.transform_ops"
    )

    found = _find_op(ops, "DecodeImage")
    if found is None:
        raise ConfigError(f"{rel}: no DecodeImage op in PreProcess")
    i, args = found
    out["img_mode"] = Sourced(
        args.get("img_mode"), f"{rel}: PreProcess.transform_ops[{i}].DecodeImage.img_mode"
    )

    found = _find_op(ops, "RecResizeImg")
    if found is None or "image_shape" not in found[1]:
        raise ConfigError(f"{rel}: no RecResizeImg.image_shape in PreProcess")
    i, args = found
    out["rec_image_shape"] = Sourced(
        list(args["image_shape"]),
        f"{rel}: PreProcess.transform_ops[{i}].RecResizeImg.image_shape  (C, H, W_min)",
    )

    found = _find_op(ops, "NormalizeImage")
    if found is not None:
        i, args = found
        out["normalize"] = Sourced(
            args, f"{rel}: PreProcess.transform_ops[{i}].NormalizeImage"
        )
    else:
        # The rec yml has no NormalizeImage op: PaddleX's OCRReisizeNormImg
        # normalizes inline. Point at the exact lines in the installed package.
        from paddlex.inference.models.text_recognition.processors import (
            OCRReisizeNormImg,
        )

        out["normalize"] = Sourced(
            "x = (x / 255 - 0.5) / 0.5  (per channel, CHW)",
            "not in inference.yml; hardcoded in PaddleX: "
            + _find_line(OCRReisizeNormImg.resize_norm_img, r"/ 255")
            + " ; "
            + _find_line(OCRReisizeNormImg.resize_norm_img, r"-= 0\.5")
            + " ; "
            + _find_line(OCRReisizeNormImg.resize_norm_img, r"/= 0\.5"),
        )
        out["rec_max_width"] = Sourced(
            OCRReisizeNormImg([3, 48, 320]).max_imgW,
            "not in inference.yml; PaddleX "
            + _find_line(OCRReisizeNormImg, r"self\.max_imgW\s*="),
        )

    post = cfg.get("PostProcess", {})
    if post.get("name") != "CTCLabelDecode":
        raise ConfigError(f"{rel}: PostProcess.name is {post.get('name')!r}, expected CTCLabelDecode")
    chars = post.get("character_dict")
    if not chars:
        raise ConfigError(f"{rel}: PostProcess.character_dict missing or empty")
    out["character_dict"] = Sourced(chars, f"{rel}: PostProcess.character_dict")

    # How PaddleX turns character_dict into the class list. Taken from the
    # installed code, not from memory: blank is prepended by
    # CTCLabelDecode.add_special_char and a space is appended when
    # use_space_char is True (the default the rec predictor leaves in place).
    from paddlex.inference.models.text_recognition.predictor import (
        TextRecRunnerPredictor,
    )
    from paddlex.inference.models.text_recognition.processors import (
        BaseRecLabelDecode,
        CTCLabelDecode,
    )

    use_space_default = inspect.signature(CTCLabelDecode.__init__).parameters[
        "use_space_char"
    ].default
    if "use_space_char" in post:
        out["use_space_char"] = Sourced(post["use_space_char"], f"{rel}: PostProcess.use_space_char")
    else:
        out["use_space_char"] = Sourced(
            use_space_default,
            "not in inference.yml; CTCLabelDecode default "
            + _find_line(CTCLabelDecode.__init__, r"def __init__")
            + " ; predictor passes only character_list: "
            + _find_line(TextRecRunnerPredictor.build_postprocess, r"character_list="),
        )
    decoder = CTCLabelDecode(character_list=chars, use_space_char=out["use_space_char"].value)
    out["ctc_classes"] = Sourced(
        len(decoder.character),
        "len(CTCLabelDecode(character_dict).character) from installed PaddleX",
    )
    out["ctc_blank_index"] = Sourced(
        decoder.get_ignored_tokens(),
        _find_line(BaseRecLabelDecode.get_ignored_tokens, r"return")
        + " ; blank inserted at index 0 by "
        + _find_line(CTCLabelDecode.add_special_char, r"blank"),
    )
    out["space_appended_at_index"] = Sourced(
        (len(decoder.character) - 1) if decoder.character[-1] == " " else None,
        _find_line(BaseRecLabelDecode.__init__, r'append\(" "\)'),
    )
    out["ctc_confidence"] = Sourced(
        "mean of max-prob over kept timesteps (after collapse + blank removal); 0 if none kept",
        _find_line(BaseRecLabelDecode.decode, r"np\.mean\(conf_list\)"),
    )
    return out


def _parse_poly(poly) -> list[list[float]]:
    return [[float(p[0]), float(p[1])] for p in poly]


# ---------------------------------------------------------------------------
# Metrics. Same definitions will be ported to the Android golden test (4f).
# ---------------------------------------------------------------------------


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)


def quad_iou(a: list[list[float]], b: list[list[float]]) -> float:
    """IoU of two convex quadrilaterals (DB boxes are min-area rectangles)."""
    import cv2
    import numpy as np

    pa = np.asarray(a, dtype=np.float32)
    pb = np.asarray(b, dtype=np.float32)
    area_a = abs(cv2.contourArea(pa))
    area_b = abs(cv2.contourArea(pb))
    if area_a <= 0 or area_b <= 0:
        return 0.0
    # convexHull guarantees consistent orientation for intersectConvexConvex.
    inter, _ = cv2.intersectConvexConvex(cv2.convexHull(pa), cv2.convexHull(pb))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def match_lines(ref: list[dict], hyp: list[dict], iou_thresh: float = 0.5) -> dict:
    """Greedy one-to-one matching by IoU (highest IoU first)."""
    pairs = []
    for i, r in enumerate(ref):
        for j, h in enumerate(hyp):
            iou = quad_iou(r["box"], h["box"])
            if iou >= iou_thresh:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_r, used_h, matches = set(), set(), []
    for iou, i, j in pairs:
        if i in used_r or j in used_h:
            continue
        used_r.add(i)
        used_h.add(j)
        matches.append(
            {
                "ref_index": i,
                "hyp_index": j,
                "iou": iou,
                "ref_text": ref[i]["text"],
                "hyp_text": hyp[j]["text"],
                "cer": cer(ref[i]["text"], hyp[j]["text"]),
            }
        )
    total_chars = sum(len(ref[m["ref_index"]]["text"]) for m in matches)
    total_edits = sum(levenshtein(m["ref_text"], m["hyp_text"]) for m in matches)
    return {
        "ref_count": len(ref),
        "hyp_count": len(hyp),
        "matched": len(matches),
        "unmatched_ref": sorted(set(range(len(ref))) - used_r),
        "unmatched_hyp": sorted(set(range(len(hyp))) - used_h),
        "matched_cer": (total_edits / total_chars) if total_chars else 0.0,
        "matches": matches,
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
