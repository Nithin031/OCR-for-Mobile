# OCR for Mobile

Fully offline OCR Android app for government forms and official documents.
Built with PP-OCRv6 Small (ONNX Runtime) and ML Kit Text Recognition.

## Module structure

| Module | Purpose |
|--------|---------|
| `app` | UI (Jetpack Compose), camera/picker integration |
| `ocr-core` | OCR engine interface, PP-OCRv6 engine, ML Kit engine, utilities |

## Build prerequisites

- Android Studio Meerkat (2025.1.x) or later, with Android SDK 35
- JDK 11+ (bundled with Android Studio)
- Run `tools/reference/fetch_models.py` and `tools/reference/build_android_assets.py`
  to populate the ONNX model files before Phase 4 (not required for Phases 1–3)

## Getting started

```bash
# 1. Clone the repo
git clone https://github.com/nithin031/ocr-for-mobile.git
cd ocr-for-mobile

# 2. (For Phase 4+) Download and prepare model assets
cd tools/reference
pip install -r requirements.txt
python fetch_models.py
python build_android_assets.py
cd ../..

# 3. Build and install
./gradlew :app:installDebug
```

Android Studio: **File → Open**, select the root folder, then **Run**.

## Asset pipeline

`tools/reference/android_assets/models/` → Gradle `copyModels` task →
`ocr-core/src/main/assets/models/` (gitignored; rebuilt on every `preBuild`).

The copy task runs automatically before every build, so assets always match
Phase 0 outputs. ONNX files are large (~30 MB) and gitignored in the source;
run `build_android_assets.py` once after `fetch_models.py`.

## Running the golden test (Phase 4f)

```bash
./gradlew :ocr-core:connectedAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.ocr.core.GoldenTest
adb pull /sdcard/Android/data/com.ocr.mobile.test/files/golden_report.txt
```

## Running the benchmark (Phase 6)

Long-press the "OCR for Mobile" title on the home screen to open the
developer benchmark screen. Select a folder of images and tap **Run**.

## Known limitations

- Latin script only (Phase 1–6). Indian scripts planned for a later phase.
- ONNX model files must be built from source (not committed to git due to size).
- ML Kit Document Scanner requires Google Play services (app falls back to
  "Pick image" if the scanner module is unavailable).

## Demo: offline OCR + on-device LLM

Scan a document (or pick a photo), read its text on the phone with ML Kit Text
Recognition, then ask an on-device LLM (Gemma 3 1B IT, int4, MediaPipe LLM
Inference on the CPU) about it: **Summarize**, **What does it ask for?**, or any
question you type. Nothing leaves the phone: the app makes no network calls of
its own, and recognized text, prompts and answers are never logged.

### Build

Gradle 8.14.3 needs JDK 17–24 (Android Studio's bundled Java 25 JBR does not
work). With a JDK 21 installed:

```powershell
$env:JAVA_HOME = "C:\Users\<you>\.jdks\jbr-21.0.11"
.\gradlew.bat :app:assembleDebug
# -> app\build\outputs\apk\debug\app-debug.apk
```

### Get the model (once)

The app never downloads the model; you import it yourself.

1. Accept the Gemma license at
   <https://huggingface.co/litert-community/Gemma3-1B-IT>.
2. Download `gemma3-1b-it-int4.task` (about 550 MB).
3. Copy it to the phone's **Download** folder.
4. In the app, open any scanned document, tap **Import model file (.task)** and
   pick the file. It is copied into app-private storage; you can then delete
   the copy in Download.

### Use

1. Tap **Scan document** (or pick an image from the gallery).
2. The recognized text appears under **Recognized text** (lines + time taken).
3. Under **Ask the on-device AI (Gemma 3 1B)**, tap **Summarize** or **What does
   it ask for?**, or type a question and tap **Ask**. The first answer also
   loads the model, so it takes longer; answers can take up to a minute.
4. Answers are AI-generated: always check names, ID numbers, dates and amounts
   against the document. Only the first ~2000 characters of text are sent to
   the model; the answer card shows how many lines were used.
## Engines and golden check (Phases 3–4)

- **Engines:** pick one on the home screen. *ML Kit* (default, bundled Latin model) or
  *PP-OCRv6 Small* (ONNX Runtime CPU + OpenCV, det max side 960). Both run fully offline.
  If PP-OCR fails to load or run, the result screen offers **Run with ML Kit**.
- **Assets:** `tools/reference/fetch_models.py` → `build_android_assets.py` →
  `tools/reference/android_assets/models/` → Gradle `copyModels` → APK assets.
  `pipeline_max960.json` holds the max960 pipeline values and PaddleX defaults, each with its source.
- **Golden check (debug builds):** home screen → select engine → *Run golden check*. It runs 3 sample
  forms (EPFO, Parivahan, Aadhaar Update) against `golden/max960/paddle`, matches boxes by IoU ≥ 0.5 and
  reports box counts, unmatched boxes and per-line/overall CER:
  `adb pull /sdcard/Android/data/com.ocr.mobile/files/golden_report.txt`
- **Known limitations:** no benchmark mode; 960 only; Latin/English dictionary use only, no Indian
  scripts; not speed-optimized (FP32, CPU); no image quality gate; arm64-v8a devices only.
