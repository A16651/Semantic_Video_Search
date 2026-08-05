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

typedef FreeFloatBufferC = ffi.Void Function(ffi.Pointer<ffi.Float> ptr);
typedef FreeFloatBufferDart = void Function(ffi.Pointer<ffi.Float> ptr);

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

typedef FreeByteBufferC = ffi.Void Function(ffi.Pointer<ffi.Uint8> ptr);
typedef FreeByteBufferDart = void Function(ffi.Pointer<ffi.Uint8> ptr);

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


class MediaCoreBridge {
  static ffi.DynamicLibrary? _lib;

  static void init() {
    if (_lib != null) return;
    if (Platform.isWindows) {
      _lib = ffi.DynamicLibrary.open('media_core.dll');
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

  static List<double> computeMel(List<int> pcmData) {
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
