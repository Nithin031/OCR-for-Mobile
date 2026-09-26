# Phase 0: Desktop reference ("golden outputs")

These scripts run the **official** PaddleOCR 3.x pipeline with **PP-OCRv6 small**
(det + rec) on the sample forms. They save the results as JSON so the Android
port (Phase 4f golden test) can be checked against them. They also fetch the
official ONNX models, dictionary and configs that the app will ship, and print
each model's I/O.

| File | Purpose |
|---|---|
| `fetch_models.py` | Downloads the official ONNX det/rec models, writes the dictionary, prints model I/O and every preprocessing value **with its source** |
| `run_reference.py` | Produces the golden JSONs, the ONNX-vs-Paddle cross-check, and the one-image intermediates |
| `refcommon.py` | Config readers (with provenance) and the metrics (IoU matching, CER) that Phase 4f will port |
| `requirements.txt` / `requirements.lock.txt` | Pinned deps / exact `pip freeze` the scripts were checked with |

## Where the models come from (verified 2026-09-26)

* `paddleocr==3.7.0` / `paddlex==3.7.2` register `PP-OCRv6_small_det` and
  `PP-OCRv6_small_rec` (`paddlex/inference/utils/official_models.py`).
* The **official ONNX exports** are listed in the PaddleOCR repo at
  `docs/version3.x/inference_deployment/cross_platform/android_deployment.en.md`
  (commit `dab3fe3`, 2026-09-16). That page documents the official
  `deploy/ppocr-android` demo:
  * Hugging Face: `PaddlePaddle/PP-OCRv6_small_det_onnx`, `PaddlePaddle/PP-OCRv6_small_rec_onnx`
  * Baidu BOS: `https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_small_{det,rec}_onnx_infer.tar`
* The character dictionary is **not a separate file** in PaddleOCR 3.x. It is
  `PostProcess.character_dict` inside `rec/inference.yml`. `fetch_models.py`
  writes it out to `models/rec/ppocr_keys.txt` (one entry per line, same order)
  and checks that it round-trips.
* `fetch_models.py` records the Hugging Face commit sha (or BOS archive sha256)
  and the sha256 of every file in `models/manifest.json`.

If both hosts are unreachable, the script stops and prints the errors for each
source. It never substitutes another model. The documented fallback is PaddleX's
own converter (`paddlex --install paddle2onnx`, then
`paddlex --paddle2onnx --paddle_model_dir … --onnx_model_dir …`). It is **not
wired into these scripts**. We would only use it if the official ONNX repos
disappear, and then we would have to re-verify it.

## Commands

Python 3.11 on Linux or macOS, CPU only. Run from the repo root:

```bash
cd tools/reference
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt      # or: pip install -r requirements.txt

# 1) Models, dictionary, configs, model I/O (needs internet: huggingface.co or bcebos.com)
python fetch_models.py                     # --source hf|bos|auto, --hf-revision <sha> to pin

# 2) Put 10-20 form images in samples/  (FAKE data only - see Privacy below)

# 3) Golden outputs (needs internet the first time: the native-Paddle
#    cross-check downloads the non-ONNX PP-OCRv6 small models by name)
python run_reference.py --save-tensors

# Useful variants
python run_reference.py --skip-paddle                          # ONNX goldens only
python run_reference.py --det-resize max:960 --det-resize min:64
python run_reference.py --intermediates-image my_form.jpg
python fetch_models.py --skip-download                         # re-inspect existing models
```

## What gets written

```
models/                          (*.onnx git-ignored; the rest is committed)
  det/inference.onnx, det/inference.yml
  rec/inference.onnx, rec/inference.yml, rec/ppocr_keys.txt
  manifest.json                  origin + revision + sha256/size of every file
  model_io.json                  ONNX input/output names, shapes, dtypes, opset
  preprocess_config.json         every value below + where it came from
golden/
  index.json                     images, settings, package versions, model shas
  max960/  max1280/              one folder per --det-resize setting
    onnx/<image>.json            <- the ground truth for the Android golden test
    paddle/<image>.json          same pipeline on the native Paddle engine
    crosscheck.json              ONNX vs Paddle: counts, IoU>=0.5 matches, CER
    pipeline_config.yaml         full effective PaddleX pipeline config
  max960/intermediates/<image>.json (+ .npz with --save-tensors)
```

Per-image golden JSON (`onnx/<image>.json`):

```jsonc
{
  "image": "form_01.jpg", "image_sha256": "…",
  "image_size": {"width": 2480, "height": 3508},   // after EXIF rotation
  "exif_orientation": 1,
  "engine": "onnxruntime",
  "text_det_params": {"limit_type": "max", "limit_side_len": 960, "max_side_limit": 4000,
                      "thresh": …, "box_thresh": …, "unclip_ratio": …},
  "text_rec_score_thresh": 0.0,
  "det_box_count": 57,
  "det_boxes": [[[x,y],[x,y],[x,y],[x,y]], …],     // every detected box, original image coords
  "lines": [{"text": "…", "confidence": 0.98, "box": [[x,y]×4]}, …],
  "versions": {…}, "models": {…}, "pipeline": {…}  // provenance
}
```

`lines` is in PaddleOCR's own sort order. The golden test matches lines by IoU,
so the order doesn't matter there.

`intermediates/<image>.json` holds:

* the detection input tensor shape
* per-channel min/max/mean/std, which is how Phase 4b checks normalization
* the resize ratios
* the probability-map shape and stats
* the first 5 recognition input shapes, with output shapes and a class-dim check

**This pass forces rec batch size 1**, so each recognition input is exactly one
text line (unpadded apart from the 320-px minimum width). With batching, "the
first 5 inputs" would be padded multi-line batches, which are much less useful
for checking the per-line resize.

## Settings used, and why

* **Detection resize.** The default is `max:960` and `max:1280`, per the project
  plan. Note that **PaddleOCR's pipeline default is different**: `OCR.yaml`
  sets `limit_type: min`, `limit_side_len: 64`, `max_side_limit: 4000`. In
  practice that means "native resolution, capped at 4000 px". The official
  Android demo's `PaddleOCRConfig` uses the same default. PaddleX's
  model-level default for PP-OCRv6 is `max`/960
  (`_TEXT_DET_MAX_LIMIT_MODELS` in `text_detection/predictor.py`). Add
  `--det-resize min:64` to also produce goldens for the official pipeline
  default. That is worth comparing on dense forms, since 960 px on an A4 scan
  shrinks small print a lot.
* **`thresh`, `box_thresh`, `unclip_ratio`** are read from `det/inference.yml`
  `PostProcess` and passed explicitly, so the pipeline's `OCR.yaml` can't
  silently override them.
* **Rec batch size** and **rec score threshold** come from PaddleX `OCR.yaml`
  (`SubModules.TextRecognition.batch_size` / `score_thresh`). The path is
  recorded in each JSON.
* Doc-orientation, doc-unwarping and text-line-orientation models are **off**.
  The Android pipeline has none of them; the ML Kit scanner does the unwarping.
* Images are decoded with `cv2.imread(IMREAD_COLOR)` (BGR, EXIF orientation
  applied), which matches the pipeline's own reader.

## Preprocessing facts for Phase 4, read from the installed PaddleX 3.7.2

These are confirmed from source, not memory. `preprocess_config.json` prints
file:line for the values that are not in `inference.yml`.

* **Det resize** (`text_detection/processors.py`, `DetResizeForTest.resize_image_type0`):
  1. `ratio` from `limit_type`/`limit_side_len`.
  2. `h' = int(h*ratio)` (truncation).
  3. Apply `max_side_limit`.
  4. Then **`max(int(round(h'/32)*32), 32)`**. Python `round` is round-half-to-even.
  5. `cv2.resize` uses the default bilinear interpolation.
  6. The box scale-back ratio is `resized/original` **after** rounding.
* **Det normalize**: mean/std/scale/order come from `det/inference.yml`
  `NormalizeImage`. `scale` may be the string `1./255.`, which PaddleX
  `eval`s. Input channel order comes from `DecodeImage.img_mode`.
* **Rec resize/normalize** (`text_recognition/processors.py`, `OCRReisizeNormImg`):
  * H comes from `RecResizeImg.image_shape` (expected `[3, 48, 320]`).
  * Each crop is resized to `ceil(48*w/h)` width, but the target width is at
    least `320` (`image_shape[2]`) and at most `3200`.
  * The result is right-padded with zeros to that width. Then the batch is
    zero-padded to its max width.
  * Normalization is **hardcoded** `(x/255 - 0.5)/0.5`. There is no
    `NormalizeImage` op in the rec yml.
* **Batching**: crops are sorted by aspect ratio before batching (`ocr/pipeline.py`).
* **CTC**: blank = index 0 (prepended). **A space is appended at the end**:
  `use_space_char=True` is the default, and the predictor never overrides it.
  So classes = `len(dict) + 2`, and the scripts check this against the ONNX
  output dim. Confidence = mean of the max-prob over the kept timesteps.
* **Crop** (`components/common/crop_image_regions.py`, `get_rotate_crop_image`):
  * Crop width = `int(max(|p0-p1|, |p2-p3|))`, height = `int(max(|p0-p3|, |p1-p2|))`.
  * `cv2.warpPerspective` with `INTER_CUBIC` and `BORDER_REPLICATE`.
  * Then `np.rot90` (90° counter-clockwise) when `h/w >= 1.5`. The plan says
    "> 1.5", so match Python's `>=`.
* The official `deploy/ppocr-android` demo hardcodes the det mean/std and ships
  only `rec/inference.yml`. We won't copy that; our port reads both ymls.

## Privacy

The golden JSONs contain every recognized string. Use **blank public forms
filled with made-up details** only. Nothing here uploads images or text
anywhere. The only network calls are the model downloads.

## Status of this phase

The scripts were checked for import, argument and plumbing correctness against
the pinned packages. The real models and a real golden run have **not** been
produced yet (see the Phase 0 summary in the PR/commit message for why).
