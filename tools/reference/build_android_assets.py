#!/usr/bin/env python3
"""
Build android_assets/models/ from models/.
Copies the five files the Android engine needs, verifies SHA-256 against
the known values in models/manifest.json, and writes a fresh manifest for
the android_assets folder.
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(r"D:\tools\reference")
SRC_MODELS = ROOT / "models"
DST = ROOT / "android_assets" / "models"

COPY_MAP = {
    DST / "det" / "inference.onnx":   SRC_MODELS / "det" / "inference.onnx",
    DST / "rec" / "inference.onnx":   SRC_MODELS / "rec" / "inference.onnx",
    DST / "rec" / "ppocr_keys.txt":   SRC_MODELS / "rec" / "ppocr_keys.txt",
    DST / "preprocess_config.json":   SRC_MODELS / "preprocess_config.json",
    DST / "model_io.json":            SRC_MODELS / "model_io.json",
}

# SHA-256 from models/manifest.json (the three HF-tracked files)
KNOWN_SHA256 = {
    "det/inference.onnx":  "d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e",
    "rec/inference.onnx":  "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634",
    "rec/ppocr_keys.txt":  "b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d",
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# Create directories
for dst_path in COPY_MAP:
    dst_path.parent.mkdir(parents=True, exist_ok=True)

# Copy and verify
manifest_entries = {}
all_ok = True

for dst_path, src_path in COPY_MAP.items():
    if not src_path.exists():
        print(f"  [MISSING SOURCE] {src_path}")
        all_ok = False
        continue

    shutil.copy2(src_path, dst_path)
    digest = sha256(dst_path)
    size = dst_path.stat().st_size

    # Relative key for manifest and verification
    rel = dst_path.relative_to(DST).as_posix()
    manifest_entries[rel] = {"sha256": digest, "bytes": size}

    if rel in KNOWN_SHA256:
        expected = KNOWN_SHA256[rel]
        if digest == expected:
            print(f"  [SHA OK ] {rel}  ({size:,} bytes)")
        else:
            print(f"  [SHA FAIL] {rel}")
            print(f"             expected: {expected}")
            print(f"             got:      {digest}")
            all_ok = False
    else:
        print(f"  [COPIED ] {rel}  sha256={digest[:16]}…  ({size:,} bytes)")

# ppocr_keys.txt line count
keys_path = DST / "rec" / "ppocr_keys.txt"
if keys_path.exists():
    line_count = sum(1 for _ in open(keys_path, encoding="utf-8"))
    if line_count == 18708:
        print(f"\n  [LINE COUNT OK] ppocr_keys.txt has exactly 18708 lines")
    else:
        print(f"\n  [LINE COUNT FAIL] ppocr_keys.txt has {line_count} lines (expected 18708)")
        all_ok = False

# Write android_assets/models/manifest.json
out_manifest = {
    "_note": "SHA-256 of each file deployed to Android assets. "
             "Regenerate with: python build_android_assets.py (from tools/reference/).",
    "source_models_manifest": str(SRC_MODELS / "manifest.json"),
    "files": manifest_entries,
}
manifest_out = DST / "manifest.json"
with open(manifest_out, "w", encoding="utf-8") as f:
    json.dump(out_manifest, f, indent=2, ensure_ascii=False)
    f.write("\n")
print(f"\n  Wrote {manifest_out}")

print(f"\n{'ALL CHECKS PASSED' if all_ok else 'ONE OR MORE CHECKS FAILED'}")
