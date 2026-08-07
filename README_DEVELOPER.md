# Cyber-Neural Cinematic AI Gallery: Developer Guide

Welcome to the development repository of the high-performance offline AI Gallery App. This guide is curated for engineers cloning and building the repository for the first time.

---

## 🗺️ Architectural Overview

Our application is designed for extreme local efficiency, offline functionality, and ultra-high-responsiveness (60FPS locked) on Windows and Android devices. It operates within a strict **500MB active RAM budget** on low-resource hardware.

```
       +--------------------------------------------------------+
       |                  Flutter UI (60 FPS)                   |
       |  - Cyber-Neural Cinematic UI / Dashboards / Player     |
       |  - Standardized 512-dim Vector and OCR-Text Querying   |
       +----------------------------+---------------------------+
                                    |
                    Isolate Spawning & IPC (SendPort)
                                    |
                                    v
       +--------------------------------------------------------+
       |             Dart background Isolate Worker             |
       |  - Standardized ObjectBox Relational / KNN database   |
       |  - Native heap buffer allocations (calloc/malloc)      |
       +----------------------------+---------------------------+
                                    |
                        Dart FFI Native Bindings
                                    |
                                    v
       +--------------------------------------------------------+
       |                Native C++ Engine Core                  |
       |  - SAD (Sum of Absolute Differences) Pixel-Diff SADs   |
       |  - Whisper decoders and Mel-spectrograph generators    |
       |  - Image normalization & projection (SigLIP / PP-OCR)  |
       +--------------------------------------------------------+
```

### Key Pillars:
1. **Dart FFI Layer (`media_core_ffi.dart`):** Communicates with local native libraries (`media_core.dll` on Windows, `libmedia_core.so` on Android/Linux) to handle memory-heavy operations on physical pixel and audio arrays.
2. **Auxiliary Worker Isolate (`worker_entry.dart`):** Offloads heavy native computations (e.g. SAD calculations, Whisper mel spec, and RGB-to-CHW normalization) to a separate OS thread to avoid locking the UI thread.
3. **Relational KNN DB (`database_manager.dart`):** Manages offline indexes and performs standardized 512-dimension vector similarity matching (Cosine / KNN) entirely offline.
4. **Model Loader Delivery:** Automatically streams required INT8 ONNX models from custom endpoints (`MODEL_BASE_URL` in `.env`) directly to disk via `IOSink` to strictly obey RAM budgets.

---

## 🛠️ Environment Prerequisites

Make sure the following SDKs are installed on your machine before compiling:
1. **Flutter SDK:** Version `>=3.22.0` (Dart SDK `>=3.4.0 <4.0.0`).
2. **Windows compilation:**
   - **Visual Studio 2022** with the **Desktop development with C++** workload.
   - **CMake** `>=3.20`.
   - **vcpkg** (C++ package manager) for dependency locks.
3. **Android compilation:**
   - **Android Studio** with **NDK (Side-by-side)** version `25.x` or `26.x`.
   - **CMake** (installed via Android SDK Manager).
   - JDK 17.

---

## 🚀 Basic Setup to Get Started

1. **Clone the repository:**
   ```bash
   git clone <repo_url>
   cd <repo_url>
   ```

2. **Configure Environment variables:**
   Create a `.env` file in the root of `media_core_ffi/`:
   ```env
   MODEL_BASE_URL=https://huggingface.co/onnx-community/whisper-tiny/resolve/main/
   ```

3. **Install Flutter Packages:**
   ```bash
   cd media_core_ffi
   flutter pub get
   ```

---

## 🪟 How to Build & Run on Windows (VS Code)

1. **Launch VS Code:**
   Open the `media_core_ffi` directory in VS Code.

2. **Install Extensions:**
   Ensure the following extensions are installed:
   - *Dart* & *Flutter*
   - *C/C++* & *CMake Tools*

3. **Native C++ Library compilation (vcpkg / MSVC):**
   If you need to recompile the `media_core.dll` library:
   ```bash
   cd native
   mkdir build && cd build
   cmake -DCMAKE_TOOLCHAIN_FILE=[path_to_vcpkg]/scripts/buildsystems/vcpkg.cmake ..
   cmake --build . --config Release
   ```
   *Note: Place the compiled `media_core.dll` inside `media_core_ffi/` root or system PATH.*

4. **Run the Flutter application:**
   Select **Windows (desktop)** as the target device in VS Code’s status bar, then press `F5` or run:
   ```bash
   flutter run -d windows
   ```

---

## 🤖 How to Build & Run on Android

Assuming you have Android Studio configured with NDK and a physical Android test device connected via ADB.

1. **Check NDK Setup:**
   Ensure `local.properties` in `android/` contains the path to your Android SDK and NDK:
   ```properties
   sdk.dir=C:\\Users\\User\\AppData\\Local\\Android\\Sdk
   ndk.dir=C:\\Users\\User\\AppData\\Local\\Android\\Sdk\\ndk\\25.1.8937393
   ```

2. **Ensure correct C++ Android Compilation:**
   The Flutter build tool invokes the CMake/NDK toolchain automatically to build `libmedia_core.so` for Android.

3. **Deploy on Connected Device:**
   Run:
   ```bash
   flutter run -d <your_device_id>
   ```

4. **Natively Shared Media Testing:**
   Test native Android share panel receiver by clicking "Share" on any image/video from your host Gallery app and choosing **media_core_ffi** as the destination.

---

## 🧬 Memory & FFI Heap Best Practices

* Always pair FFI allocations (`calloc` / `malloc`) with corresponding `calloc.free` or `MediaCoreBridge.freeFloat`/`freeByte` in `finally` blocks.
* Stream files to disk chunk-by-chunk using `IOSink` to prevent entire model binary files from being read into memory.
