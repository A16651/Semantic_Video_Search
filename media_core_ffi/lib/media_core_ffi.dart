import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import 'dart:io';

// Declare standard Dart FFI C Types representation
final class TrackingBoxStruct extends ffi.Struct {
  @ffi.Int32()
  external int id;

  @ffi.Array(64)
  external ffi.Array<ffi.Uint8> label;

  @ffi.Float()
  external double confidence;

  @ffi.Float()
  external double xMin;
  @ffi.Float()
  external double yMin;
  @ffi.Float()
  external double xMax;
  @ffi.Float()
  external double yMax;
}

final class OCRTextResultStruct extends ffi.Struct {
  @ffi.Array(256)
  external ffi.Array<ffi.Uint8> text;

  @ffi.Float()
  external double confidence;

  @ffi.Array(8)
  external ffi.Array<ffi.Float> boundingBox;
}

// FFI Signatures
typedef ComputeSadThresholdC = ffi.Bool Function(
  ffi.Pointer<ffi.Uint8> frameA,
  ffi.Pointer<ffi.Uint8> frameB,
  ffi.Uint32 width,
  ffi.Uint32 height,
  ffi.Float thresholdPercentage,
);
typedef ComputeSadThresholdDart = bool Function(
  ffi.Pointer<ffi.Uint8> frameA,
  ffi.Pointer<ffi.Uint8> frameB,
  int width,
  int height,
  double thresholdPercentage,
);

typedef EncodeTextQueryC = ffi.Pointer<ffi.Float> Function(
  ffi.Pointer<Utf8> query,
  ffi.Pointer<ffi.Int32> outDimension,
);
typedef EncodeTextQueryDart = ffi.Pointer<ffi.Float> Function(
  ffi.Pointer<Utf8> query,
  ffi.Pointer<ffi.Int32> outDimension,
);

typedef ProjectImageEmbeddingC = ffi.Pointer<ffi.Float> Function(
  ffi.Pointer<ffi.Float> raw1152Embedding,
  ffi.Pointer<ffi.Int32> outDimension,
);
typedef ProjectImageEmbeddingDart = ffi.Pointer<ffi.Float> Function(
  ffi.Pointer<ffi.Float> raw1152Embedding,
  ffi.Pointer<ffi.Int32> outDimension,
);

typedef WhisperComputeMelC = ffi.Pointer<ffi.Float> Function(
  ffi.Pointer<ffi.Int16> pcmData,
  ffi.Int32 sampleCount,
  ffi.Pointer<ffi.Int32> outBinCount,
);
typedef WhisperComputeMelDart = ffi.Pointer<ffi.Float> Function(
  ffi.Pointer<ffi.Int16> pcmData,
  int sampleCount,
  ffi.Pointer<ffi.Int32> outBinCount,
);

typedef WhisperDecodeTokensC = ffi.Pointer<Utf8> Function(
  ffi.Pointer<ffi.Int32> tokenIds,
  ffi.Int32 tokenCount,
  ffi.Pointer<Utf8> tokenizerJsonPath,
);
typedef WhisperDecodeTokensDart = ffi.Pointer<Utf8> Function(
  ffi.Pointer<ffi.Int32> tokenIds,
  int tokenCount,
  ffi.Pointer<Utf8> tokenizerJsonPath,
);

typedef WhisperTranscribeAudioC = ffi.Pointer<Utf8> Function(
  ffi.Pointer<ffi.Float> melData,
  ffi.Int32 melBins,
  ffi.Pointer<Utf8> modelDir,
);
typedef WhisperTranscribeAudioDart = ffi.Pointer<Utf8> Function(
  ffi.Pointer<ffi.Float> melData,
  int melBins,
  ffi.Pointer<Utf8> modelDir,
);

typedef WhisperTranscribeFullPcmC = ffi.Pointer<Utf8> Function(
  ffi.Pointer<ffi.Int16> pcmData,
  ffi.Int32 totalSamples,
  ffi.Pointer<Utf8> modelDir,
);
typedef WhisperTranscribeFullPcmDart = ffi.Pointer<Utf8> Function(
  ffi.Pointer<ffi.Int16> pcmData,
  int totalSamples,
  ffi.Pointer<Utf8> modelDir,
);

typedef RunPpOcrC = ffi.Pointer<OCRTextResultStruct> Function(
  ffi.Pointer<ffi.Uint8> rgbData,
  ffi.Uint32 width,
  ffi.Uint32 height,
  ffi.Pointer<Utf8> modelDir,
  ffi.Pointer<ffi.Int32> outCount,
);
typedef RunPpOcrDart = ffi.Pointer<OCRTextResultStruct> Function(
  ffi.Pointer<ffi.Uint8> rgbData,
  int width,
  int height,
  ffi.Pointer<Utf8> modelDir,
  ffi.Pointer<ffi.Int32> outCount,
);

typedef FreeOcrResultsC = ffi.Void Function(ffi.Pointer<OCRTextResultStruct> ptr);
typedef FreeOcrResultsDart = void Function(ffi.Pointer<OCRTextResultStruct> ptr);

typedef SaveFrameAsJpegC = ffi.Bool Function(
  ffi.Pointer<ffi.Uint8> rgbData,
  ffi.Uint32 width,
  ffi.Uint32 height,
  ffi.Pointer<Utf8> outputPath,
);
typedef SaveFrameAsJpegDart = bool Function(
  ffi.Pointer<ffi.Uint8> rgbData,
  int width,
  int height,
  ffi.Pointer<Utf8> outputPath,
);

typedef EncodeImageFrameC = ffi.Pointer<ffi.Float> Function(
  ffi.Pointer<ffi.Float> chwData,
  ffi.Pointer<ffi.Int32> outDimension,
);
typedef EncodeImageFrameDart = ffi.Pointer<ffi.Float> Function(
  ffi.Pointer<ffi.Float> chwData,
  ffi.Pointer<ffi.Int32> outDimension,
);

typedef NormalizeRgb24HwcToChwC = ffi.Pointer<ffi.Float> Function(
  ffi.Pointer<ffi.Uint8> rgbData,
  ffi.Uint32 width,
  ffi.Uint32 height,
  ffi.Uint32 targetW,
  ffi.Uint32 targetH,
);
typedef NormalizeRgb24HwcToChwDart = ffi.Pointer<ffi.Float> Function(
  ffi.Pointer<ffi.Uint8> rgbData,
  int width,
  int height,
  int targetW,
  int targetH,
);

typedef InitSiglipModelC = ffi.Bool Function(ffi.Pointer<Utf8> modelDir);
typedef InitSiglipModelDart = bool Function(ffi.Pointer<Utf8> modelDir);

typedef InitBpeTokenizerC = ffi.Bool Function(ffi.Pointer<Utf8> modelDir);
typedef InitBpeTokenizerDart = bool Function(ffi.Pointer<Utf8> modelDir);

typedef InitWhisperModelsC = ffi.Bool Function(ffi.Pointer<Utf8> modelDir);
typedef InitWhisperModelsDart = bool Function(ffi.Pointer<Utf8> modelDir);

typedef FreeStringBufferC = ffi.Void Function(ffi.Pointer<Utf8> ptr);
typedef FreeStringBufferDart = void Function(ffi.Pointer<Utf8> ptr);

typedef FreeFloatBufferC = ffi.Void Function(ffi.Pointer<ffi.Float> ptr);
typedef FreeFloatBufferDart = void Function(ffi.Pointer<ffi.Float> ptr);

typedef FreeByteBufferC = ffi.Void Function(ffi.Pointer<ffi.Uint8> ptr);
typedef FreeByteBufferDart = void Function(ffi.Pointer<ffi.Uint8> ptr);


class MediaCoreBridge {
  static ffi.DynamicLibrary? _lib;

  static void init() {
    if (_lib != null) return;
    if (Platform.isWindows) {
      try {
        _lib = ffi.DynamicLibrary.open('media_core.dll');
      } catch (_) {
        // Fallback paths if standard DLL load fails due to working directory changes
        final exeDir = File(Platform.resolvedExecutable).parent.path;
        final candidates = [
          '$exeDir/media_core.dll',
          '${Directory.current.path}/media_core.dll',
          '${Directory.current.path}/media_core_ffi/media_core.dll',
          '${Directory.current.path}/../media_core.dll',
        ];
        for (var path in candidates) {
          if (File(path).existsSync()) {
            try {
              _lib = ffi.DynamicLibrary.open(path);
              break;
            } catch (_) {}
          }
        }
        _lib ??= ffi.DynamicLibrary.open('media_core.dll');
      }
    } else if (Platform.isAndroid || Platform.isLinux) {
      _lib = ffi.DynamicLibrary.open('libmedia_core.so');
    } else {
      _lib = ffi.DynamicLibrary.process();
    }
  }

  static bool computeSad(
    ffi.Pointer<ffi.Uint8> frameA,
    ffi.Pointer<ffi.Uint8> frameB,
    int width,
    int height,
    double threshold,
  ) {
    init();
    final func = _lib!
        .lookup<ffi.NativeFunction<ComputeSadThresholdC>>('compute_sad_threshold')
        .asFunction<ComputeSadThresholdDart>();
    return func(frameA, frameB, width, height, threshold);
  }

  static List<double> encodeText(String query) {
    init();
    final func = _lib!
        .lookup<ffi.NativeFunction<EncodeTextQueryC>>('encode_text_query')
        .asFunction<EncodeTextQueryDart>();

    final queryPtr = query.toNativeUtf8();
    final outDimPtr = calloc<ffi.Int32>();

    try {
      final floatPtr = func(queryPtr, outDimPtr);
      if (floatPtr == ffi.nullptr) {
        throw StateError("SigLIP Inference Failed: Unable to generate text embedding for '$query'");
      }
      final size = outDimPtr.value;
      final List<double> result = [];
      for (int i = 0; i < size; i++) {
        result.add(floatPtr[i]);
      }
      freeFloat(floatPtr);
      return result;
    } finally {
      calloc.free(queryPtr);
      calloc.free(outDimPtr);
    }
  }

  static bool saveFrameAsJpeg(
    ffi.Pointer<ffi.Uint8> rgbData,
    int width,
    int height,
    String outputPath,
  ) {
    init();
    final func = _lib!
        .lookup<ffi.NativeFunction<SaveFrameAsJpegC>>('save_frame_as_jpeg')
        .asFunction<SaveFrameAsJpegDart>();
    final pathPtr = outputPath.toNativeUtf8();
    try {
      return func(rgbData, width, height, pathPtr);
    } finally {
      calloc.free(pathPtr);
    }
  }

  static List<double> encodeImageFrame(ffi.Pointer<ffi.Float> chwTensor) {
    init();
    final func = _lib!
        .lookup<ffi.NativeFunction<EncodeImageFrameC>>('encode_image_frame')
        .asFunction<EncodeImageFrameDart>();

    final outDimPtr = calloc<ffi.Int32>();
    try {
      final floatPtr = func(chwTensor, outDimPtr);
      if (floatPtr == ffi.nullptr) {
        throw StateError("SigLIP Vision Transformer Inference Failed");
      }
      final size = outDimPtr.value;
      final List<double> result = [];
      for (int i = 0; i < size; i++) {
        result.add(floatPtr[i]);
      }
      freeFloat(floatPtr);
      return result;
    } finally {
      calloc.free(outDimPtr);
    }
  }

  static List<double> projectEmbedding(List<double> raw1152) {
    init();
    final func = _lib!
        .lookup<ffi.NativeFunction<ProjectImageEmbeddingC>>('project_image_embedding')
        .asFunction<ProjectImageEmbeddingDart>();

    final rawPtr = calloc<ffi.Float>(raw1152.length);
    for (int i = 0; i < raw1152.length; i++) {
      rawPtr[i] = raw1152[i];
    }
    final outDimPtr = calloc<ffi.Int32>();

    try {
      final floatPtr = func(rawPtr, outDimPtr);
      final size = outDimPtr.value;
      final List<double> result = [];
      for (int i = 0; i < size; i++) {
        result.add(floatPtr[i]);
      }
      freeFloat(floatPtr);
      return result;
    } finally {
      calloc.free(rawPtr);
      calloc.free(outDimPtr);
    }
  }

  static List<double> whisperComputeMel(List<int> pcmData) {
    init();
    final func = _lib!
        .lookup<ffi.NativeFunction<WhisperComputeMelC>>('whisper_compute_mel')
        .asFunction<WhisperComputeMelDart>();

    final pcmPtr = calloc<ffi.Int16>(pcmData.length);
    for (int i = 0; i < pcmData.length; i++) {
      pcmPtr[i] = pcmData[i];
    }
    final outBinPtr = calloc<ffi.Int32>();

    try {
      final floatPtr = func(pcmPtr, pcmData.length, outBinPtr);
      final size = outBinPtr.value;
      final List<double> result = [];
      for (int i = 0; i < size; i++) {
        result.add(floatPtr[i]);
      }
      freeFloat(floatPtr);
      return result;
    } finally {
      calloc.free(pcmPtr);
      calloc.free(outBinPtr);
    }
  }

  static List<double> normalizeRgb24HwcToChw(
    List<int> rgbData,
    int width,
    int height,
    int targetW,
    int targetH,
  ) {
    init();
    final func = _lib!
        .lookup<ffi.NativeFunction<NormalizeRgb24HwcToChwC>>('normalize_rgb24_hwc_to_chw')
        .asFunction<NormalizeRgb24HwcToChwDart>();

    final rgbPtr = calloc<ffi.Uint8>(rgbData.length);
    for (int i = 0; i < rgbData.length; i++) {
      rgbPtr[i] = rgbData[i];
    }

    try {
      final floatPtr = func(rgbPtr, width, height, targetW, targetH);
      final size = targetW * targetH * 3;
      final List<double> result = [];
      for (int i = 0; i < size; i++) {
        result.add(floatPtr[i]);
      }
      freeFloat(floatPtr);
      return result;
    } finally {
      calloc.free(rgbPtr);
    }
  }

  static ffi.Pointer<ffi.Float> normalizeRgbDirect(
    ffi.Pointer<ffi.Uint8> rgbPtr,
    int width,
    int height,
    int targetW,
    int targetH,
  ) {
    init();
    final func = _lib!
        .lookup<ffi.NativeFunction<NormalizeRgb24HwcToChwC>>('normalize_rgb24_hwc_to_chw')
        .asFunction<NormalizeRgb24HwcToChwDart>();
    return func(rgbPtr, width, height, targetW, targetH);
  }

  static String whisperDecodeTokens(List<int> tokenIds, String tokenizerJsonPath) {
    init();
    final func = _lib!
        .lookup<ffi.NativeFunction<WhisperDecodeTokensC>>('whisper_decode_tokens')
        .asFunction<WhisperDecodeTokensDart>();

    final tokenPtr = calloc<ffi.Int32>(tokenIds.length);
    for (int i = 0; i < tokenIds.length; i++) {
      tokenPtr[i] = tokenIds[i];
    }
    final pathPtr = tokenizerJsonPath.toNativeUtf8();

    try {
      final strPtr = func(tokenPtr, tokenIds.length, pathPtr);
      final String result = strPtr.toDartString();
      freeByte(strPtr.cast<ffi.Uint8>());
      return result;
    } finally {
      calloc.free(tokenPtr);
      calloc.free(pathPtr);
    }
  }

  static String whisperTranscribeAudio(List<double> melData, String modelDir) {
    init();
    final func = _lib!
        .lookup<ffi.NativeFunction<WhisperTranscribeAudioC>>('whisper_transcribe_audio')
        .asFunction<WhisperTranscribeAudioDart>();

    final melPtr = calloc<ffi.Float>(melData.length);
    for (int i = 0; i < melData.length; i++) {
      melPtr[i] = melData[i];
    }
    final dirPtr = modelDir.toNativeUtf8();

    try {
      final strPtr = func(melPtr, melData.length, dirPtr);
      final String result = strPtr.toDartString();
      freeByte(strPtr.cast<ffi.Uint8>());
      return result;
    } finally {
      calloc.free(melPtr);
      calloc.free(dirPtr);
    }
  }

  static String whisperTranscribeFullPcm(List<int> pcmData, String modelDir) {
    init();
    final func = _lib!
        .lookup<ffi.NativeFunction<WhisperTranscribeFullPcmC>>('whisper_transcribe_full_pcm')
        .asFunction<WhisperTranscribeFullPcmDart>();

    final pcmPtr = calloc<ffi.Int16>(pcmData.length);
    for (int i = 0; i < pcmData.length; i++) {
      pcmPtr[i] = pcmData[i];
    }
    final dirPtr = modelDir.toNativeUtf8();

    try {
      final strPtr = func(pcmPtr, pcmData.length, dirPtr);
      final String result = strPtr.toDartString();
      freeString(strPtr);
      return result;
    } finally {
      calloc.free(pcmPtr);
      calloc.free(dirPtr);
    }
  }

  static List<Map<String, dynamic>> runPpOcr(
    ffi.Pointer<ffi.Uint8> rgbData,
    int width,
    int height,
    String modelDir,
  ) {
    init();
    final func = _lib!
        .lookup<ffi.NativeFunction<RunPpOcrC>>('run_pp_ocr')
        .asFunction<RunPpOcrDart>();

    final outCountPtr = calloc<ffi.Int32>();
    final dirPtr = modelDir.toNativeUtf8();

    try {
      final ffi.Pointer<OCRTextResultStruct> resultsPtr =
          func(rgbData, width, height, dirPtr, outCountPtr);
      final int count = outCountPtr.value;

      final List<Map<String, dynamic>> list = [];
      if (resultsPtr != ffi.nullptr && count > 0) {
        for (int i = 0; i < count; i++) {
          final structItem = (resultsPtr + i).ref;

          final List<int> charCodes = [];
          for (int c = 0; c < 256; c++) {
            final code = structItem.text[c];
            if (code == 0) break;
            charCodes.add(code);
          }
          final String text = String.fromCharCodes(charCodes).trim();

          final List<double> box = [];
          for (int b = 0; b < 8; b++) {
            box.add(structItem.boundingBox[b]);
          }

          if (text.isNotEmpty) {
            list.add({
              'text': text,
              'confidence': structItem.confidence,
              'boundingBox': box,
            });
          }
        }

        // CRITICAL MEMORY GUARD: Free native OCR struct array memory immediately after reading
        freeOcrResults(resultsPtr);
      }
      return list;
    } finally {
      calloc.free(outCountPtr);
      calloc.free(dirPtr);
    }
  }

  static void freeOcrResults(ffi.Pointer<OCRTextResultStruct> ptr) {
    init();
    final func = _lib!
        .lookup<ffi.NativeFunction<FreeOcrResultsC>>('free_ocr_results')
        .asFunction<FreeOcrResultsDart>();
    func(ptr);
  }

  static ffi.Pointer<ffi.Float> normalizeRgb24(
    ffi.Pointer<ffi.Uint8> rgbData,
    int width,
    int height,
    int targetW,
    int targetH,
  ) {
    init();
    final func = _lib!
        .lookup<ffi.NativeFunction<NormalizeRgb24HwcToChwC>>('normalize_rgb24_hwc_to_chw')
        .asFunction<NormalizeRgb24HwcToChwDart>();
    return func(rgbData, width, height, targetW, targetH);
  }

  static bool initSiglipModel(String modelDir) {
    init();
    final func = _lib!
        .lookup<ffi.NativeFunction<InitSiglipModelC>>('init_siglip_model')
        .asFunction<InitSiglipModelDart>();
    final pathPtr = modelDir.toNativeUtf8();
    try {
      return func(pathPtr);
    } finally {
      calloc.free(pathPtr);
    }
  }

  static bool initBpeTokenizer(String modelDir) {
    init();
    final func = _lib!
        .lookup<ffi.NativeFunction<InitBpeTokenizerC>>('init_bpe_tokenizer')
        .asFunction<InitBpeTokenizerDart>();
    final pathPtr = modelDir.toNativeUtf8();
    try {
      return func(pathPtr);
    } finally {
      calloc.free(pathPtr);
    }
  }

  static bool initWhisperModels(String modelDir) {
    init();
    final func = _lib!
        .lookup<ffi.NativeFunction<InitWhisperModelsC>>('init_whisper_models')
        .asFunction<InitWhisperModelsDart>();
    final pathPtr = modelDir.toNativeUtf8();
    try {
      return func(pathPtr);
    } finally {
      calloc.free(pathPtr);
    }
  }

  static void freeString(ffi.Pointer<Utf8> ptr) {
    init();
    final func = _lib!
        .lookup<ffi.NativeFunction<FreeStringBufferC>>('free_string_buffer')
        .asFunction<FreeStringBufferDart>();
    func(ptr);
  }

  static void freeFloat(ffi.Pointer<ffi.Float> ptr) {
    init();
    final func = _lib!
        .lookup<ffi.NativeFunction<FreeFloatBufferC>>('free_float_buffer')
        .asFunction<FreeFloatBufferDart>();
    func(ptr);
  }

  static void freeByte(ffi.Pointer<ffi.Uint8> ptr) {
    init();
    final func = _lib!
        .lookup<ffi.NativeFunction<FreeByteBufferC>>('free_byte_buffer')
        .asFunction<FreeByteBufferDart>();
    func(ptr);
  }
}
