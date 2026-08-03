# Manual Steps & Platform Compilation Guide

This document defines the instructions to compile native C++ assets, execute database bindings, and acquire on-device model files.

## 1. Local C++ Library Compilation

Because compiling native C++ binaries requires platform-specific toolchains, execute the following instructions on your target operating system.

### A. Windows Desktop Compilation (MSVC x64)

Ensure Visual Studio with "Desktop development with C++" is installed. Launch a **Developer PowerShell for VS 2022** and execute:

```powershell
cd media_core_ffi/native
cl.exe /LD /O2 /EHsc media_core.cpp /Fe:media_core.dll
```

Move the generated `media_core.dll` library to the root folder or place it in the same directory as your flutter executable.

### B. Android Native Development Kit (NDK) Compilation

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

Move the generated `libmedia_core.so` directly into `android/app/src/main/jniLibs/arm64-v8a/` inside the Flutter Android runner tree.

---

## 2. ObjectBox Code Generation

Before running or deploying the Flutter application, compile the ObjectBox bindings and database schemas:

```bash
cd media_core_ffi
flutter pub get
flutter pub run build_runner build --delete-conflicting-outputs
```

---

## 3. Hugging Face ONNX Model Acquisition

As the models (SigLIP, Whisper, PP-OCR) exceed Google Play Store boundaries (150MB Limit), they must be downloaded and hosted on direct URL CDNs.

Download URL endpoints should map to:
* **SigLIP-SO400M INT8:** `https://huggingface.co/onnx-community/SigLIP-SO400M-ONNX-INT8/resolve/main/model.onnx` (Place at `local_models/siglip.onnx`)
* **Whisper Tiny INT8:** `https://huggingface.co/onnx-community/whisper-tiny-ONNX-INT8/resolve/main/model.onnx` (Place at `local_models/whisper.onnx`)
* **PP-OCR INT8:** `https://huggingface.co/onnx-community/PP-OCR-INT8/resolve/main/model.onnx` (Place at `local_models/pp_ocr.onnx`)
