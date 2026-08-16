import 'package:flutter_test/flutter_test.dart';
import 'package:media_core_ffi/media_core_ffi.dart';

void main() {
  test('MediaCoreBridge FFI initialization and function execution', () {
    print("Testing MediaCoreBridge.init()...");
    MediaCoreBridge.init();

    final textEmbed = MediaCoreBridge.encodeText("Test video scene");
    expect(textEmbed.length, equals(512));
    print("encodeText: 512-dim vector OK");

    final dummy1152 = List<double>.filled(1152, 0.5);
    final projected = MediaCoreBridge.projectEmbedding(dummy1152);
    expect(projected.length, equals(512));
    print("projectEmbedding: 512-dim vector OK");

    final dummyPcm = List<int>.generate(16000, (i) => (i % 100) * 10);
    final mel = MediaCoreBridge.whisperComputeMel(dummyPcm);
    expect(mel.length, greaterThan(0));
    print("whisperComputeMel: ${mel.length} bins OK");
  });
}
