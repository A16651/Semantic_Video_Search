# Developer Documentation & Architecture Guide

Welcome to the **Neural Cinema Engine** developer guide. This document details the high-performance offline on-device AI system architecture, memory preservation strategies, and background processing pipelines implemented in this project.

---

## 1. System Architecture Overview

The application is structured into three primary architectural tiers:
1. **The Native C++ Engine (media_core):** Implements low-level high-performance operations, including Pixel-Difference Sum of Absolute Differences (SAD) for scene change detection, ONNX Runtime environment loaders, Mel-spectrogram computation via PocketFFT, and token decoding.
2. **The Dart FFI Bridge (media_core_ffi.dart):** Exposes C-compatible pointers, structures, and native function signatures to Dart. It ensures direct memory access without subprocess or shell serialization overhead.
3. **The Flutter Isolate & UI Layer:** Spawns a background Isolate Worker (`worker_entry.dart`) to execute heavy AI pipeline telemetry. The UI thread renders at a fluid 60 FPS under a strict cinematic cyber-neural design system.

---

## 2. In-Memory Streaming & Model Loader (Workflow 1)

Heavy AI models are critical assets required for offline text encoding, image projection, audio transcription, and optical character recognition (OCR):
* **Files Tracked:**
  - `decoder_model.onnx`
  - `decoder_with_past_model.onnx`
  - `encoder_model.onnx`
  - `ppocr_det_fp32.onnx`
  - `ppocr_rec_fp32.onnx`
  - `siglip.onnx`
  - `tokenizer.json`
* **On-Disk Target:** Files are streamed directly to `getApplicationDocumentsDirectory()` on first launch.
* **Network Strategy:** Uses chunked response streaming with an active `IOSink` to stream directly to disk, avoiding whole-file buffering in RAM to honor our strict **500MB RAM budget**.

---

## 3. High-Priority Processing & Android Foreground Services (Workflow 4)

To prevent the operating system from terminating the heavy background worker during complex video analysis:
* **Background Worker Isolate:** All processing (SAD frame calculations, RGB normalization, Whisper Mel-spectrogram conversion) is offloaded to an auxiliary Dart Isolate to maintain 60 FPS responsiveness on the main thread.
* **Android Foreground Service Elevation:** When immediate priority processing or re-processing is requested (via the media detail page's persistent "Re-Process" button), the background task is elevated to a native Android Foreground Service with a sticky notification overlay via the `flutter_foreground_task` package.
* **Re-Processing Support:** Allows re-running video files dynamically through any selected pipeline ("Frames Only", "Audio Only", or "Full Summary"), updating ObjectBox metadata upon completion.

---

## 4. Windows Watched Directories & Deduplication (Workflow 5)

For Windows desktop environments, auto-import watched directory monitoring is fully integrated:
* **Directory Selection:** Uses `file_picker`'s `getDirectoryPath()` static method to resolve absolute folder paths.
* **Path Serialization:** Natively serializes Windows directory paths using backslashes (`\`) rather than Unix forward slashes.
* **Startup Auto-Scan & Deduplication:** On app startup, watched folders are scanned recursively (`listSync(recursive: true)`). To avoid re-processing existing files, the scanner executes an exact deduplication check against the ObjectBox asset records. Only unindexed items are routed to the Dart Worker Isolate.

---

## 5. FFI Memory Management Guardrails

Because the bridge relies on direct `malloc` / native pointer return values, explicit memory deallocation is mandatory to prevent RAM accumulation:
* **Memory Budget:** Statically restricted to a **500MB RAM ceiling** on low-tier mobile devices.
* **Native Deallocators:** Every pointer returned from C++ (such as the Float array for RGB normalization or the char* decoded string) is released by calling `free_float_buffer` and `free_byte_buffer` respectively inside `finally` blocks.
* **Calloc Freeing:** Dart side allocations created via `package:ffi` are guaranteed to call `calloc.free(ptr)` immediately after FFI function execution.
