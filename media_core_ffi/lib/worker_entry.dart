import 'dart:ffi' as ffi;
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'package:flutter/foundation.dart';
import 'package:ffmpeg_kit_flutter_min_gpl/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_flutter_min_gpl/return_code.dart';
import 'media_core_ffi.dart';
import 'background_worker.dart';
import 'database_manager.dart';

// Robust cross-platform FFmpeg command runner prioritizing native Process execution and timeout safety to prevent Dart Isolate platform channel deadlocks
Future<bool> runFFmpegCommand(List<String> args) async {
  workerDebugPrint("FFmpeg command started with args: ${args.join(' ')}");

  // 1. Try direct system process invocation first (bypasses Dart isolate platform channel deadlock)
  try {
    workerDebugPrint("Executing Process.run('ffmpeg', ...) via native OS process...");
    final res = await Process.run('ffmpeg', ['-y', ...args]).timeout(const Duration(minutes: 10));
    workerDebugPrint("Process.run finished with exitCode: ${res.exitCode}");
    if (res.exitCode == 0) {
      return true;
    } else {
      workerDebugPrint("Process.run stderr: ${res.stderr}");
    }
  } catch (e) {
    workerDebugPrint("Process.run failed or timed out: $e");
  }

  // 2. Fall back to FFmpegKit with strict 10-minute timeout guard
  try {
    workerDebugPrint("Falling back to FFmpegKit.executeWithArguments (10-minute timeout guard)...");
    final session = await FFmpegKit.executeWithArguments(['-y', ...args]).timeout(const Duration(minutes: 10));
    final returnCode = await session.getReturnCode();
    workerDebugPrint("FFmpegKit session completed with returnCode: $returnCode");
    if (ReturnCode.isSuccess(returnCode)) {
      return true;
    }
  } catch (e) {
    workerDebugPrint("FFmpegKit fallback failed or timed out: $e");
  }

  workerDebugPrint("FFmpeg command execution failed on all paths.");
  return false;
}

// Heavy, blocking native FFI processing pipeline routed through a background Dart Worker Isolate.
// REAL NATIVE INFERENCE WITH TOP-DOWN RAW RGB & HEADERLESS PCM PRE-PROCESSING:
// 1. Audio Pre-processing: Extracts raw headerless 16kHz PCM s16le audio via FFmpeg (-f s16le -acodec pcm_s16le).
//    Zero 44-byte WAV header misalignment.
// 2. Vision Pre-processing: Extracts 1-FPS keyframes as raw top-down RGB24 files (-f image2 -vcodec rawvideo -pix_fmt rgb24).
//    - Zero upside-down BMP pixel flip.
//    - Full-resolution uncompressed RGB24 frame (640x480) is passed directly to PP-OCR for crisp text recognition.
//    - 224x224 downsampled RGB24 frame is passed to normalizeRgb24() and projectEmbedding() for SigLIP.
void workerEntryPoint(BackgroundIngestionTask task) async {
  final List<VideoFrameIndex> localFrames = [];
  final List<AudioTranscriptIndex> localTranscripts = [];

  workerDebugPrint("=== WORKER ISOLATE ENTRYPOINT STARTED ===");
  workerDebugPrint("Task Target -> videoId: ${task.videoId}, path: ${task.videoPath}");
  workerDebugPrint("Task Options -> processFrames: ${task.processFrames}, processAudio: ${task.processAudio}, processOcr: ${task.processOcr}, modelDir: ${task.modelDir}");

  try {
    task.replyPort.send(IngestionProgress(
      videoPath: task.videoPath,
      progress: 0.03,
      currentAction: "Initializing native C++ ONNX & FFI engine...",
    ));

    // Set working directory so C++ relative paths (siglip.onnx, tokenizer.json) resolve correctly
    if (task.modelDir.isNotEmpty) {
      try {
        Directory.current = Directory(task.modelDir).parent;
        workerDebugPrint("Set working directory to: ${Directory.current.path}");
      } catch (e) {
        try {
          Directory.current = Directory(task.modelDir);
          workerDebugPrint("Fallback working directory set to: ${Directory.current.path}");
        } catch (e2) {
          workerDebugPrint("Warning: Could not set working directory: $e2");
        }
      }
    }

    // Initialize FFI loading context within this Worker Isolate thread
    workerDebugPrint("Initializing MediaCoreBridge FFI library...");
    MediaCoreBridge.init();
    workerDebugPrint("MediaCoreBridge FFI successfully initialized.");

    final int videoId = task.videoId;
    const int targetWidth = 224;
    const int targetHeight = 224;
    const int targetImageSize = targetWidth * targetHeight * 3; // 224x224 RGB24 = 150,528 bytes

    const int srcW = 640;
    const int srcH = 480;
    const int fullFrameSize = srcW * srcH * 3; // 640x480 RGB24 = 921,600 bytes

    final Directory tempDir = Directory.systemTemp;
    final String normVideoPath = task.videoPath.replaceAll('\\', '/');

    // ─────────────────────────────────────────────────────────────────────
    // 1. REAL VISION DECODING & MULTI-RESOLUTION ONNX INFERENCE PIPELINE
    // ─────────────────────────────────────────────────────────────────────
    if (task.processFrames) {
      workerDebugPrint("--- STARTING VISION PIPELINE ---");
      task.replyPort.send(IngestionProgress(
        videoPath: task.videoPath,
        progress: 0.08,
        currentAction: "Decoding 1-FPS video keyframes via FFmpegKit...",
      ));

      final String framePrefix = 'temp_frame_${videoId}_${DateTime.now().millisecondsSinceEpoch}_';
      final String framePattern = '${tempDir.path}/$framePrefix%04d.rgb'.replaceAll('\\', '/');
      workerDebugPrint("Targeting 1-FPS extraction to pattern: $framePattern");

      final bool ffmpegSuccess = await runFFmpegCommand([
        '-i', task.videoPath,
        '-vf', 'fps=1,scale=640:480',
        '-f', 'image2',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24',
        framePattern,
      ]);

      workerDebugPrint("FFmpeg keyframe extraction status: $ffmpegSuccess");

      // Collect extracted raw RGB files
      final List<File> rgbFiles = [];
      if (ffmpegSuccess) {
        try {
          final List<FileSystemEntity> entities = tempDir.listSync();
          for (var entity in entities) {
            if (entity is File && entity.path.contains(framePrefix) && entity.path.endsWith('.rgb')) {
              rgbFiles.add(entity);
            }
          }
          rgbFiles.sort((a, b) => a.path.compareTo(b.path));
        } catch (e) {
          workerDebugPrint("Error scanning temp directory for RGB keyframes: $e");
        }
      }

      workerDebugPrint("Discovered ${rgbFiles.length} raw RGB keyframe files.");

      if (rgbFiles.isNotEmpty) {
        final int totalFrames = rgbFiles.length;
        workerDebugPrint("Allocating native 224x224 RGB frame buffers ($targetImageSize bytes each)...");
        final ffi.Pointer<ffi.Uint8> prevFrame = calloc<ffi.Uint8>(targetImageSize);
        final ffi.Pointer<ffi.Uint8> currFrame = calloc<ffi.Uint8>(targetImageSize);

        try {
          for (int idx = 0; idx < totalFrames; idx++) {
            final File rgbFile = rgbFiles[idx];
            final int ts = idx * 1000; // 1 FPS = 1000ms steps
            final double stepProgress = 0.12 + (idx / totalFrames) * 0.38;

            workerDebugPrint("Processing Keyframe ${idx + 1}/$totalFrames (Timestamp: ${ts}ms)");

            Uint8List rawRgbBytes;
            try {
              rawRgbBytes = rgbFile.readAsBytesSync();
            } catch (e) {
              workerDebugPrint("Failed to read raw RGB file ${rgbFile.path}: $e");
              continue;
            }

            if (rawRgbBytes.length < fullFrameSize) {
              workerDebugPrint("Skipping frame ${idx + 1}: file size ${rawRgbBytes.length} < expected $fullFrameSize");
              continue;
            }

            workerDebugPrint("Allocating top-down 640x480 full-res RGB24 buffer ($fullFrameSize bytes)...");
            final ffi.Pointer<ffi.Uint8> fullResFrame = calloc<ffi.Uint8>(fullFrameSize);

            try {
              // Copy top-down raw RGB24 bytes directly into fullResFrame
              for (int i = 0; i < fullFrameSize; i++) {
                fullResFrame[i] = rawRgbBytes[i];
              }

              // Downsample fullResFrame to 224x224 currFrame strictly for SigLIP vision projection
              for (int dy = 0; dy < targetHeight; dy++) {
                final double sy = (dy / (targetHeight - 1)) * (srcH - 1);
                final int y0 = sy.floor().clamp(0, srcH - 1);
                final int y1 = (y0 + 1).clamp(0, srcH - 1);
                final double fy = sy - y0;

                for (int dx = 0; dx < targetWidth; dx++) {
                  final double sx = (dx / (targetWidth - 1)) * (srcW - 1);
                  final int x0 = sx.floor().clamp(0, srcW - 1);
                  final int x1 = (x0 + 1).clamp(0, srcW - 1);
                  final double fx = sx - x0;

                  for (int c = 0; c < 3; c++) {
                    final int offset00 = (y0 * srcW + x0) * 3 + c;
                    final int offset01 = (y0 * srcW + x1) * 3 + c;
                    final int offset10 = (y1 * srcW + x0) * 3 + c;
                    final int offset11 = (y1 * srcW + x1) * 3 + c;

                    final double v00 = fullResFrame[offset00].toDouble();
                    final double v01 = fullResFrame[offset01].toDouble();
                    final double v10 = fullResFrame[offset10].toDouble();
                    final double v11 = fullResFrame[offset11].toDouble();

                    final double val = (1.0 - fx) * (1.0 - fy) * v00 +
                        fx * (1.0 - fy) * v01 +
                        (1.0 - fx) * fy * v10 +
                        fx * fy * v11;

                    currFrame[dy * targetWidth * 3 + dx * 3 + c] = val.round().clamp(0, 255);
                  }
                }
              }

              // Save keyframe 0 thumbnail to disk
              if (idx == 0) {
                try {
                  final String videoDir = File(normVideoPath).parent.path;
                  final String thumbPath = '$videoDir/thumb_$videoId.jpg';
                  workerDebugPrint("Exporting keyframe 0 thumbnail to: $thumbPath");
                  MediaCoreBridge.saveFrameAsJpeg(currFrame, targetWidth, targetHeight, thumbPath);
                } catch (e) {
                  workerDebugPrint("Failed to export keyframe thumbnail: $e");
                }
              }

              // 1. SAD pixel-diffing between consecutive temporal frames via C++ FFI
              workerDebugPrint("Computing C++ SAD pixel diffing for frame ${idx + 1}...");
              bool isSceneChange = idx == 0 ||
                  MediaCoreBridge.computeSad(prevFrame, currFrame, targetWidth, targetHeight, 8.0);
              workerDebugPrint("Frame ${idx + 1} SAD scene change detection -> isSceneChange: $isSceneChange");

              // 2. Normalize RGB24 HWC to CHW Float32 tensor via C++ FFI for SigLIP
              workerDebugPrint("Converting RGB24 HWC to CHW Float32 tensor for SigLIP...");
              final ffi.Pointer<ffi.Float> chwTensor =
                  MediaCoreBridge.normalizeRgb24(currFrame, targetWidth, targetHeight, targetWidth, targetHeight);

              // 3. Execute real SigLIP ONNX Vision Transformer (pixel_values [1, 3, 224, 224] -> 512-dim visual vector)
              List<double> projected512;
              try {
                workerDebugPrint("Calling MediaCoreBridge.encodeImageFrame (SigLIP ONNX)...");
                projected512 = MediaCoreBridge.encodeImageFrame(chwTensor);
                workerDebugPrint("SigLIP encoding successful (vector dim: ${projected512.length}).");
              } catch (e) {
                workerDebugPrint("SigLIP encoding error: $e. Falling back to zero vector.");
                projected512 = List<double>.generate(512, (_) => 0.0);
              }

              MediaCoreBridge.freeFloat(chwTensor);

              for (int b = 0; b < targetImageSize; b++) {
                prevFrame[b] = currFrame[b];
              }

              // Execute real C++ ONNX PP-OCR model pipeline on top-down uncompressed RGB24 keyframe pixels
              String frameOcrText = '';
              if (task.processOcr && (isSceneChange || idx % 5 == 0)) {
                try {
                  workerDebugPrint("Passing frame ${idx + 1} to C++ PP-OCR engine...");
                  final ocrHits = MediaCoreBridge.runPpOcr(
                    fullResFrame,
                    srcW,
                    srcH,
                    task.modelDir,
                  );
                  if (ocrHits.isNotEmpty) {
                    final texts = ocrHits.map((h) => h['text'] as String).toList();
                    frameOcrText = texts.join(' | ');
                    workerDebugPrint("PP-OCR extracted ${ocrHits.length} text hits: '$frameOcrText'");
                  } else {
                    workerDebugPrint("PP-OCR complete: no text detected in frame ${idx + 1}.");
                  }
                } catch (e) {
                  workerDebugPrint("PP-OCR execution error on frame ${idx + 1}: $e");
                }
              }

              final String detectedLabel = isSceneChange
                  ? '["Scene Keyframe", "Active Dynamic"]'
                  : '["Standard Frame"]';

              localFrames.add(VideoFrameIndex(
                videoId: videoId,
                timestampMs: ts,
                embedding512: projected512,
                detectedObjects: detectedLabel,
                detectedFaces: '[]',
                ocrText: frameOcrText,
              ));

              task.replyPort.send(IngestionProgress(
                videoPath: task.videoPath,
                progress: stepProgress,
                currentAction:
                    "Frame ${idx + 1}/$totalFrames (${srcW}x$srcH Top-Down RGB OCR / 224x224 SigLIP) projected at ${ts ~/ 1000}s",
              ));
            } finally {
              calloc.free(fullResFrame);
            }

            // Clean up temporary raw RGB file immediately after processing
            try {
              rgbFile.deleteSync();
            } catch (_) {}
          }
        } finally {
          calloc.free(prevFrame);
          calloc.free(currFrame);
        }
      } else {
        workerDebugPrint("Fallback: No keyframe RGBs decoded by FFmpeg for ${task.videoPath}");
      }
      workerDebugPrint("--- COMPLETED VISION PIPELINE (${localFrames.length} frames indexed) ---");
    }

    // ─────────────────────────────────────────────────────────────────────
    // 2. CONTINUOUS 30-SECOND WHISPER AUDIO PIPELINE & RAW HEADERLESS PCM
    // Extracts raw 16kHz mono PCM s16le audio via FFmpeg (-f s16le -acodec pcm_s16le).
    // ZERO 44-byte WAV header misalignment trap.
    // ─────────────────────────────────────────────────────────────────────
    if (task.processAudio) {
      workerDebugPrint("--- STARTING AUDIO PIPELINE ---");
      task.replyPort.send(IngestionProgress(
        videoPath: task.videoPath,
        progress: 0.52,
        currentAction: "Decoding audio track to raw headerless 16kHz mono PCM via FFmpegKit...",
      ));

      final String tempPcmPath = '${tempDir.path}/temp_audio_${videoId}_${DateTime.now().millisecondsSinceEpoch}.pcm'.replaceAll('\\', '/');
      workerDebugPrint("Target PCM audio output file: $tempPcmPath");

      final bool pcmSuccess = await runFFmpegCommand([
        '-i', task.videoPath,
        '-vn',
        '-f', 's16le',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        tempPcmPath,
      ]);

      workerDebugPrint("FFmpeg PCM extraction status: $pcmSuccess");

      List<int> fullPcm = [];
      final File tempPcmFile = File(tempPcmPath);

      if (pcmSuccess && tempPcmFile.existsSync()) {
        try {
          final Uint8List pcmBytes = tempPcmFile.readAsBytesSync();
          tempPcmFile.deleteSync(); // Delete temporary PCM file immediately after reading

          final int totalSamples = pcmBytes.length ~/ 2;
          workerDebugPrint("Loaded raw PCM buffer: ${pcmBytes.length} bytes ($totalSamples samples at 16kHz).");

          final ByteData bd = ByteData.view(pcmBytes.buffer, pcmBytes.offsetInBytes, pcmBytes.length);
          fullPcm = List<int>.generate(totalSamples, (i) => bd.getInt16(i * 2, Endian.little));
        } catch (e) {
          workerDebugPrint("Error reading raw PCM audio bytes: $e");
        }
      }

      if (fullPcm.isNotEmpty) {
        task.replyPort.send(IngestionProgress(
          videoPath: task.videoPath,
          progress: 0.65,
          currentAction:
              "Executing native C++ 30-second chunking Whisper speech recognition on ${fullPcm.length} PCM samples...",
        ));

        // Pre-initialize & cache ONNX encoder and decoder sessions exactly ONCE
        workerDebugPrint("Initializing C++ Whisper ONNX model sessions from modelDir: ${task.modelDir}...");
        MediaCoreBridge.initWhisperModels(task.modelDir);

        // Pass uncompressed 16kHz PCM buffer to native C++ whisperTranscribeFullPcm
        workerDebugPrint("Invoking native MediaCoreBridge.whisperTranscribeFullPcm...");
        final String rawTranscripts = MediaCoreBridge.whisperTranscribeFullPcm(
            fullPcm, task.modelDir);
        workerDebugPrint("Whisper transcription returned raw string of length: ${rawTranscripts.length}");

        if (rawTranscripts.isNotEmpty) {
          // Parse delimited format: [startMs-endMs] sentence | [startMs-endMs] sentence
          final List<String> chunks = rawTranscripts.split(' | ');
          workerDebugPrint("Parsed ${chunks.length} transcript segments from native result.");

          for (int i = 0; i < chunks.length; i++) {
            final String chunkStr = chunks[i].trim();
            if (chunkStr.isEmpty) continue;

            int startMs = i * 30000;
            int endMs = (i + 1) * 30000;
            String sentence = chunkStr;

            final match = RegExp(r'^\[(\d+)-(\d+)\]\s*(.*)$').firstMatch(chunkStr);
            if (match != null) {
              startMs = int.tryParse(match.group(1)!) ?? startMs;
              endMs = int.tryParse(match.group(2)!) ?? endMs;
              sentence = match.group(3) ?? sentence;
            }

            final String cleanedSentence = sentence
                .replaceAll(RegExp(r'[^\x20-\x7E]'), ' ')
                .replaceAll(RegExp(r'\s+'), ' ')
                .trim();

            if (cleanedSentence.isNotEmpty) {
              List<double> textEmbed512;
              try {
                workerDebugPrint("Encoding transcript sentence ${i + 1}/${chunks.length}: '$cleanedSentence'");
                textEmbed512 = MediaCoreBridge.encodeText(cleanedSentence);
              } catch (e) {
                workerDebugPrint("Warning: Text encoding error for sentence '$cleanedSentence': $e. Falling back to zero vector.");
                textEmbed512 = List<double>.filled(512, 0.0);
              }

              localTranscripts.add(AudioTranscriptIndex(
                videoId: videoId,
                timestampStartMs: startMs,
                timestampEndMs: endMs,
                sentence: cleanedSentence,
                textEmbedding512: textEmbed512,
              ));
            }
          }
        }
      } else {
        workerDebugPrint("Fallback: No PCM audio decoded by FFmpeg for ${task.videoPath}");
      }

      task.replyPort.send(IngestionProgress(
        videoPath: task.videoPath,
        progress: 0.90,
        currentAction:
            "Transcribed ${localTranscripts.length} continuous Whisper speech segments.",
      ));
      workerDebugPrint("--- COMPLETED AUDIO PIPELINE (${localTranscripts.length} segments indexed) ---");
    }

    // ─────────────────────────────────────────────────────────────────────
    // COMPLETION — send all real computed data back to main UI isolate
    // ─────────────────────────────────────────────────────────────────────
    workerDebugPrint("=== WORKER ISOLATE PIPELINE FINISHED SUCCESSFULLY ===");
    workerDebugPrint("Summary -> ${localFrames.length} frames, ${localTranscripts.length} transcripts computed.");

    task.replyPort.send(IngestionProgress(
      videoPath: task.videoPath,
      progress: 1.0,
      currentAction:
          "Native FFI pipeline complete. ${localFrames.length} 1-FPS visual vectors, ${localTranscripts.length} 30-second Whisper audio segments.",
      completed: true,
      computedFrames: localFrames,
      computedTranscripts: localTranscripts,
    ));
  } catch (e, st) {
    workerDebugPrint("FATAL WORKER ISOLATE ERROR: $e\n$st");
    task.replyPort.send(IngestionProgress(
      videoPath: task.videoPath,
      progress: 1.0,
      currentAction: "Ingestion error: $e",
      completed: true,
      error: e.toString(),
      computedFrames: localFrames,
      computedTranscripts: localTranscripts,
    ));
  }
}

void workerDebugPrint(String message) {
  if (kDebugMode) {
    print("[NeuralIsolateWorker] $message");
  }
}
