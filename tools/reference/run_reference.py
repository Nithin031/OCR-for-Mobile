#!/usr/bin/env python3
"""Phase 0, steps 1-3: run the official PaddleOCR pipeline with PP-OCRv6 small
on every image in samples/ and save golden outputs for the Android port.

Run fetch_models.py first.

For each detection-resize setting (default: max:960 and max:1280) this writes:

  golden/<setting>/onnx/<image>.json     PaddleOCR pipeline, onnxruntime engine,
                                         using exactly the ONNX files in models/
                                         (the same files the APK will ship).
                                         -> this is what the Android golden test
                                            (Phase 4f) compares against.
  golden/<setting>/paddle/<image>.json   Same pipeline on the native Paddle
                                         engine (models downloaded by PaddleOCR).
  golden/<setting>/crosscheck.json       ONNX vs Paddle: box count, IoU>=0.5
                                         matches, CER. Shows whether the ONNX
                                         export itself loses accuracy.
  golden/<setting>/pipeline_config.yaml  Full effective PaddleX pipeline config.

and, for ONE image (--intermediates-image, default: first sample):

  golden/<first setting>/intermediates/<image>.json
      det input tensor shape + per-channel stats, det probability-map shape +
      stats, and the first 5 recognition inputs' shapes (see README for why
      this pass uses rec batch size 1). --save-tensors also writes the raw
      tensors to a .npz next to it.

Recognized text is written only to the JSON files, never printed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

# PaddlePaddle 3.3.1 on Windows: the new PIR executor combined with oneDNN/MKLDNN
# triggers "ConvertPirAttribute2RuntimeAttribute not support
# [pir::ArrayAttribute<pir::DoubleAttribute>]" at runtime.
# PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=0 keeps the default run_mode as "paddle"
# (which calls config.disable_mkldnn()) instead of "mkldnn" (which enables it).
# setdefault() means a user-set "1" still takes precedence.
if sys.platform == "win32":
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")

import numpy as np

from refcommon import (
    DEFAULT_GOLDEN_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_SAMPLES_DIR,
    EXPECTED_DET_MODEL_NAME,
    EXPECTED_REC_MODEL_NAME,
    ConfigError,
    _pkg_relative,
    levenshtein,
    list_images,
    load_yaml,
    match_lines,
    read_det_config,
    read_rec_config,
    sha256_file,
    write_json,
)


# ---------------------------------------------------------------------------
# Capture of ONNX Runtime calls (for the intermediates pass).
# Patching InferenceSession is independent of PaddleX internals: whatever the
# pipeline feeds the models is exactly what we record.
# ---------------------------------------------------------------------------


class OrtCapture:
    def __init__(self, det_model: Path, rec_model: Path):
        self.det_model = det_model.resolve()
        self.rec_model = rec_model.resolve()
        self.active = False
        self.calls: list[dict] = []
        self._installed = False

    def install(self):
        if self._installed:
            return
        import onnxruntime as ort

        cap = self
        orig_init = ort.InferenceSession.__init__
        orig_run = ort.InferenceSession.run

        def init(self, path_or_bytes, *a, **k):
            orig_init(self, path_or_bytes, *a, **k)
            self._ref_model_path = (
                Path(path_or_bytes).resolve()
                if isinstance(path_or_bytes, (str, os.PathLike))
                else None
            )

        def run(self, output_names, input_feed, *a, **k):
            outputs = orig_run(self, output_names, input_feed, *a, **k)
            if cap.active:
                cap._record(getattr(self, "_ref_model_path", None), input_feed, outputs)
            return outputs

        ort.InferenceSession.__init__ = init
        ort.InferenceSession.run = run
        self._installed = True

    def _record(self, model_path, input_feed, outputs):
        role = {self.det_model: "det", self.rec_model: "rec"}.get(model_path, "other")
        self.calls.append(
            {
                "role": role,
                "inputs": {k: np.asarray(v) for k, v in input_feed.items()},
                "outputs": [np.asarray(o) for o in outputs],
            }
        )


def _stats(a: np.ndarray) -> dict:
    a = a.astype(np.float64, copy=False)
    return {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()), "std": float(a.std())}


# ---------------------------------------------------------------------------


def versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ("paddleocr", "paddlex", "paddlepaddle", "onnxruntime", "onnx",
                "numpy", "opencv-contrib-python", "opencv-python", "opencv-python-headless"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def parse_resize(spec: str) -> tuple[str, int]:
    try:
        kind, side = spec.split(":")
        side = int(side)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected <max|min>:<int>, got {spec!r}")
    if kind not in ("max", "min"):
        raise argparse.ArgumentTypeError(f"limit type must be max or min, got {kind!r}")
    return kind, side


def ocr_yaml_defaults() -> dict:
    """Pipeline-level defaults from PaddleX's own OCR.yaml (not from memory)."""
    import paddlex

    path = Path(paddlex.__file__).parent / "configs" / "pipelines" / "OCR.yaml"
    cfg = load_yaml(path)
    rec = cfg["SubModules"]["TextRecognition"]
    det = cfg["SubModules"]["TextDetection"]
    return {
        "path": _pkg_relative(str(path)),
        "rec_batch_size": rec["batch_size"],
        "rec_score_thresh": rec["score_thresh"],
        "det_limit_type": det["limit_type"],
        "det_limit_side_len": det["limit_side_len"],
        "det_max_side_limit": det["max_side_limit"],
    }


def build_pipeline(*, engine: str, models_dir: Path | None, limit_type: str, limit_side: int,
                   det_cfg: dict, score_thresh: float, rec_batch_size: int):
    from paddleocr import PaddleOCR

    kwargs = dict(
        text_detection_model_name=EXPECTED_DET_MODEL_NAME,
        text_recognition_model_name=EXPECTED_REC_MODEL_NAME,
        # The Android pipeline has no doc-orientation, unwarping (the ML Kit
        # scanner does that) or text-line orientation model, so the reference
        # must not use them either.
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_type=limit_type,
        text_det_limit_side_len=limit_side,
        text_det_thresh=det_cfg["db_thresh"].value,
        text_det_box_thresh=det_cfg["db_box_thresh"].value,
        text_det_unclip_ratio=det_cfg["db_unclip_ratio"].value,
        text_rec_score_thresh=score_thresh,
        text_recognition_batch_size=rec_batch_size,
        device="cpu",
    )
    if engine == "onnxruntime":
        kwargs.update(
            text_detection_model_dir=str(models_dir / "det"),
            text_recognition_model_dir=str(models_dir / "rec"),
            engine="onnxruntime",
        )
    # engine == "paddle": leave engine unset -> PaddleOCR's default Paddle
    # engine; it downloads the native PP-OCRv6 small models by name.
    return PaddleOCR(**kwargs)


def read_image(path: Path) -> tuple[np.ndarray, int | None]:
    """BGR uint8, EXIF orientation applied (cv2.IMREAD_COLOR does this)."""
    import cv2
    from PIL import Image

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cv2 could not read {path}")
    try:
        with Image.open(path) as im:
            orientation = im.getexif().get(0x0112)
    except Exception:
        orientation = None
    return img, orientation


def _poly(p) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in np.asarray(p).reshape(-1, 2)]


def _builtin(o):
    if isinstance(o, dict):
        return {k: _builtin(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_builtin(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def result_to_json(res, image_path: Path, img: np.ndarray, orientation, meta: dict) -> dict:
    texts, scores, polys = res["rec_texts"], res["rec_scores"], res["rec_polys"]
    if not (len(texts) == len(scores) == len(polys)):
        raise RuntimeError(f"{image_path.name}: rec_texts/scores/polys lengths differ")
    return {
        "image": image_path.name,
        "image_sha256": sha256_file(image_path),
        "image_size": {"width": int(img.shape[1]), "height": int(img.shape[0])},
        "exif_orientation": orientation,
        **meta,
        "text_det_params": _builtin(res["text_det_params"]),
        "text_rec_score_thresh": float(res["text_rec_score_thresh"]),
        "det_box_count": len(res["dt_polys"]),
        "det_boxes": [_poly(p) for p in res["dt_polys"]],
        "lines": [
            {"text": t, "confidence": float(s), "box": _poly(p)}
            for t, s, p in zip(texts, scores, polys)
        ],
    }


def run_setting(ocr, images, out_dir: Path, meta: dict) -> dict[str, dict]:
    results = {}
    for path in images:
        img, orientation = read_image(path)
        (res,) = ocr.predict(img)
        j = result_to_json(res, path, img, orientation, meta)
        write_json(out_dir / f"{path.stem}.json", j)
        results[path.name] = j
        print(f"    {path.name}: {j['det_box_count']} boxes, {len(j['lines'])} lines", flush=True)
    return results


def run_intermediates(args, capture: OrtCapture, image: Path, limit_type, limit_side,
                      det_cfg, score_thresh, out_dir: Path) -> None:
    ocr = build_pipeline(engine="onnxruntime", models_dir=args.models_dir,
                         limit_type=limit_type, limit_side=limit_side, det_cfg=det_cfg,
                         score_thresh=score_thresh, rec_batch_size=1)
    img, _ = read_image(image)
    capture.calls.clear()
    capture.active = True
    try:
        (res,) = ocr.predict(img)
    finally:
        capture.active = False
        ocr.close()

    det_calls = [c for c in capture.calls if c["role"] == "det"]
    rec_calls = [c for c in capture.calls if c["role"] == "rec"]
    if len(det_calls) != 1:
        raise RuntimeError(f"expected 1 detection call, captured {len(det_calls)}")
    det = det_calls[0]
    (det_in_name, det_in), = det["inputs"].items()
    prob = det["outputs"][0]
    h, w = img.shape[:2]
    thresh = det_cfg["db_thresh"].value
    out = {
        "image": image.name,
        "original_size_hw": [h, w],
        "det_limit_type": limit_type,
        "det_limit_side_len": limit_side,
        "detection": {
            "input_name": det_in_name,
            "input_shape": list(det_in.shape),
            "input_dtype": str(det_in.dtype),
            "resize_ratio_hw": [det_in.shape[2] / h, det_in.shape[3] / w],
            "input_stats_per_channel": [_stats(det_in[0, c]) for c in range(det_in.shape[1])],
            "prob_map_shape": list(prob.shape),
            "prob_map_stats": _stats(prob),
            "prob_map_frac_above_thresh": float((prob > thresh).mean()),
            "thresh_used": thresh,
            "boxes_after_postprocess": len(res["dt_polys"]),
        },
        "recognition": {
            "note": "rec batch size forced to 1 for this pass so each call is one "
                    "text line; calls are in PaddleX order (sorted by crop aspect ratio)",
            "total_calls": len(rec_calls),
            "first_5": [
                {
                    "input_name": next(iter(c["inputs"])),
                    "input_shape": list(next(iter(c["inputs"].values())).shape),
                    "input_stats": _stats(next(iter(c["inputs"].values()))),
                    "output_shape": list(c["outputs"][0].shape),
                }
                for c in rec_calls[:5]
            ],
        },
    }
    if rec_calls:
        classes = rec_calls[0]["outputs"][0].shape[-1]
        expected = read_rec_config(args.models_dir / "rec")["ctc_classes"].value
        out["recognition"]["class_dim_check"] = {
            "runtime_output_last_dim": int(classes),
            "expected_from_dict": expected,
            "passed": int(classes) == expected,
        }
    write_json(out_dir / f"{image.stem}.json", out)
    if args.save_tensors:
        arrays = {"det_input": det_in, "det_prob_map": prob}
        for i, c in enumerate(rec_calls[:5]):
            arrays[f"rec_input_{i}"] = next(iter(c["inputs"].values()))
            arrays[f"rec_output_{i}"] = c["outputs"][0]
        np.savez_compressed(out_dir / f"{image.stem}.npz", **arrays)

    print("\n=== Intermediates for", image.name, "===")
    d = out["detection"]
    print(f"  det input  {d['input_name']!r} shape={d['input_shape']} (original HxW={h}x{w})")
    print(f"  det prob map shape={d['prob_map_shape']}")
    for i, r in enumerate(out["recognition"]["first_5"]):
        print(f"  rec input #{i} {r['input_name']!r} shape={r['input_shape']} -> output {r['output_shape']}")
    if "class_dim_check" in out["recognition"]:
        print(f"  rec class-dim check: {out['recognition']['class_dim_check']}")


def crosscheck(onnx_res: dict, paddle_res: dict) -> dict:
    per_image, edits, chars = {}, 0, 0
    for name, ref in paddle_res.items():
        m = match_lines(ref["lines"], onnx_res[name]["lines"])
        per_image[name] = m
        for mm in m["matches"]:
            chars += len(mm["ref_text"])
            edits += levenshtein(mm["ref_text"], mm["hyp_text"])
    return {
        "reference": "paddle",
        "hypothesis": "onnx",
        "iou_threshold": 0.5,
        "overall_matched_cer": edits / chars if chars else 0.0,
        "per_image": per_image,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES_DIR)
    ap.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_GOLDEN_DIR)
    ap.add_argument("--det-resize", type=parse_resize, action="append",
                    help="Detection resize as <limit_type>:<side>, repeatable. "
                         "Default: max:960 and max:1280 (the Android plan). "
                         "PaddleOCR's own pipeline default is min:64 (see README).")
    ap.add_argument("--rec-batch-size", type=int, default=None,
                    help="Default: TextRecognition.batch_size from PaddleX OCR.yaml")
    ap.add_argument("--skip-paddle", action="store_true",
                    help="Skip the native-Paddle cross-check run")
    ap.add_argument("--intermediates-image", default=None,
                    help="File name in samples/ to record intermediates for (default: first)")
    ap.add_argument("--save-tensors", action="store_true",
                    help="Also save raw det/rec tensors for the intermediates image (.npz)")
    args = ap.parse_args()
    args.models_dir = args.models_dir.resolve()

    settings = args.det_resize or [("max", 960), ("max", 1280)]
    images = list_images(args.samples_dir)
    if not images:
        raise SystemExit(f"No images in {args.samples_dir}. Add 10-20 form images first.")

    for f in ("det/inference.onnx", "det/inference.yml", "rec/inference.onnx",
              "rec/inference.yml", "manifest.json"):
        if not (args.models_dir / f).is_file():
            raise SystemExit(f"Missing {args.models_dir / f}. Run fetch_models.py first.")

    det_cfg = read_det_config(args.models_dir / "det")
    read_rec_config(args.models_dir / "rec")  # validates the rec config early
    defaults = ocr_yaml_defaults()
    rec_bs = args.rec_batch_size or defaults["rec_batch_size"]
    score_thresh = defaults["rec_score_thresh"]
    manifest = json.loads((args.models_dir / "manifest.json").read_text(encoding="utf-8"))

    capture = OrtCapture(args.models_dir / "det" / "inference.onnx",
                         args.models_dir / "rec" / "inference.onnx")
    capture.install()

    common_meta = {
        "versions": versions(),
        "models": {
            role: {
                "origin": manifest[role]["origin"],
                "inference.onnx_sha256": manifest[role]["files"]["inference.onnx"]["sha256"],
            }
            for role in ("det", "rec")
        },
        "pipeline": {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "rec_batch_size": rec_bs,
            "rec_batch_size_source": "--rec-batch-size" if args.rec_batch_size
                                     else f"{defaults['path']}: SubModules.TextRecognition.batch_size",
            "rec_score_thresh_source": f"{defaults['path']}: SubModules.TextRecognition.score_thresh",
            "det_thresholds_source": {k: det_cfg[k].source
                                      for k in ("db_thresh", "db_box_thresh", "db_unclip_ratio")},
            "image_decode": "cv2.imread(IMREAD_COLOR) -> BGR, EXIF orientation applied",
        },
    }

    index = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "images": [p.name for p in images],
        "settings": {},
        **common_meta,
    }

    for limit_type, side in settings:
        tag = f"{limit_type}{side}"
        sdir = args.out_dir / tag
        print(f"\n=== Setting {tag}: limit_type={limit_type} limit_side_len={side} ===", flush=True)

        print("  [onnx] PaddleOCR pipeline, engine=onnxruntime, models from", args.models_dir)
        ocr = build_pipeline(engine="onnxruntime", models_dir=args.models_dir,
                             limit_type=limit_type, limit_side=side, det_cfg=det_cfg,
                             score_thresh=score_thresh, rec_batch_size=rec_bs)
        sdir.mkdir(parents=True, exist_ok=True)
        ocr.export_paddlex_config_to_yaml(str(sdir / "pipeline_config.yaml"))
        onnx_res = run_setting(ocr, images, sdir / "onnx", {**common_meta, "engine": "onnxruntime"})
        ocr.close()
        index["settings"][tag] = {"limit_type": limit_type, "limit_side_len": side,
                                  "onnx": f"{tag}/onnx/"}

        if not args.skip_paddle:
            print("  [paddle] PaddleOCR pipeline, default Paddle engine (native models)")
            ocr = build_pipeline(engine="paddle", models_dir=None,
                                 limit_type=limit_type, limit_side=side, det_cfg=det_cfg,
                                 score_thresh=score_thresh, rec_batch_size=rec_bs)
            paddle_res = run_setting(ocr, images, sdir / "paddle",
                                     {**common_meta, "engine": "paddle (PaddleOCR default)"})
            ocr.close()
            cc = crosscheck(onnx_res, paddle_res)
            write_json(sdir / "crosscheck.json", cc)
            index["settings"][tag]["paddle"] = f"{tag}/paddle/"
            index["settings"][tag]["crosscheck_overall_matched_cer"] = cc["overall_matched_cer"]
            print(f"  ONNX vs Paddle: overall matched CER = {cc['overall_matched_cer']:.4%}")
            for name, m in cc["per_image"].items():
                print(f"    {name}: paddle={m['ref_count']} onnx={m['hyp_count']} "
                      f"matched={m['matched']} cer={m['matched_cer']:.4%}")

    inter_img = images[0]
    if args.intermediates_image:
        matches = [p for p in images if p.name == args.intermediates_image]
        if not matches:
            raise SystemExit(f"--intermediates-image {args.intermediates_image!r} not in samples/")
        inter_img = matches[0]
    limit_type, side = settings[0]
    inter_dir = args.out_dir / f"{limit_type}{side}" / "intermediates"
    run_intermediates(args, capture, inter_img, limit_type, side, det_cfg, score_thresh, inter_dir)
    index["intermediates"] = f"{limit_type}{side}/intermediates/{inter_img.stem}.json"

    write_json(args.out_dir / "index.json", index)
    print(f"\nWrote golden outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConfigError as e:
        raise SystemExit(f"Config error: {e}")
