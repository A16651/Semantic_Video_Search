# Manual Steps & Platform Compilation Guide

This document defines the instructions to compile native C++ assets, execute database bindings, and acquire on-device model files for the Semantic Video Search project.

> **Note:** The main Flutter app resides inside the `media_core_ffi/` directory.

---

## 1. Platform Folder Setup & Native C++ Library Compilation

### A. Generating Platform Runners (Android & Windows)
If the `android/` or `windows/` directories do not exist inside `media_core_ffi/`, generate them by running:

```bash
cd media_core_ffi
flutter create --platforms=windows,android .
```

### B. Windows Desktop Compilation (MSVC x64)

Ensure Visual Studio with "Desktop development with C++" is installed. Launch a **Developer PowerShell for VS 2022** and execute:

```powershell
cd media_core_ffi/native
cl.exe /LD /O2 /EHsc media_core.cpp /Fe:media_core.dll
```

Move the generated `media_core.dll` library to the `media_core_ffi/` root folder or place it in the same directory as your compiled Flutter executable.

### C. Android Native Development Kit (NDK) Compilation

Ensure your `ANDROID_NDK_HOME` environment path is set. Run:

```bash
cd media_core_ffi/native

# Set target compiler path depending on host OS
$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ \
  -target aarch64-linux-android31 \
  -shared -O3 -fPIC \
  media_core.cpp \
  -o libmedia_core.so
```

Copy the generated `libmedia_core.so` directly into `media_core_ffi/android/app/src/main/jniLibs/arm64-v8a/libmedia_core.so`.

---

## 2. ObjectBox Code Generation & Dependency Resolution

Before running or deploying the Flutter application, compile the ObjectBox bindings:

```bash
cd media_core_ffi
flutter pub get
dart run build_runner build --delete-conflicting-outputs
```

*Note: If you encounter an analyzer incompatibility error on newer Dart SDK versions, ensure `pubspec.yaml` includes `dependency_overrides` for `source_gen: ^4.0.0`.*

---

## 3. On-Device AI Models Setup (SigLIP, Whisper, PP-OCR)

The offline engine requires ONNX INT8 quantized model files for visual embeddings, audio transcription, and optical character recognition.

### Model Files to Place in `local_models/`:
Create a `local_models/` folder inside `media_core_ffi/` and place the following ONNX files:
* **SigLIP-SO400M / SigLIP Base INT8:** `local_models/siglip.onnx`
  - Download from Hugging Face repositories hosting ONNX quantized vision models (e.g., [`onnx-community/siglip-base-patch16-224`](https://huggingface.co/onnx-community/siglip-base-patch16-224)) or export via `optimum-cli`.
* **Whisper Tiny INT8:** `local_models/whisper.onnx`
  - Download from Hugging Face [`onnx-community/whisper-tiny`](https://huggingface.co/onnx-community/whisper-tiny).
* **PP-OCR INT8:** `local_models/pp_ocr.onnx`
  - Download PaddleOCR ONNX INT8 model from PaddleOCR ONNX community exports.

---

## 3.5. First-Launch Model Delivery System (via .env)

The application implements a robust first-launch model delivery system that downloads heavy ONNX models on demand, streaming them directly to disk via `IOSink` and verifying them with chunked SHA-256 signatures to comply with a strict 500MB RAM budget.

### Setup `.env` Configuration:
Create a `.env` file in `media_core_ffi/` (already registered in `pubspec.yaml` assets) with the following variable:
```env
MODEL_BASE_URL=https://models.example.com/v1
```

### Models Downloaded:
- `siglip.onnx`
- `whisper_tiny.onnx`
- `whisper.encoder`
- `whisper.decoder`
- `pp_ocr.onnx`
- `tokenizer.json`

### Handling Network Drops & Offline Testing:
The model delivery screen handles network drops gracefully by displaying retry options. To test fully offline, click **LOCAL BYPASS**, which emulates the stream download and SHA-256 verification of all 6 files and allows immediate access to the Cyber Neural Cinematic Gallery.

---

## 4. How to Run the Application

Navigate to the `media_core_ffi` directory and launch the app:

### Run on Windows Desktop:
```bash
cd media_core_ffi
flutter run -d windows
```

### Run on Android (Device or Emulator):
```bash
cd media_core_ffi
flutter run -d android
```

