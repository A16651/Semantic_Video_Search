import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:media_core_ffi/gallery_screen.dart';

void main() {
  testWidgets('CyberNeuralApp Smoke test - Verify Model Loader', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const CyberNeuralApp());

    // Verify that the Neural Model Loader title starts up correctly
    expect(find.text('NEURAL MODEL LOADER'), findsOneWidget);
    expect(find.text('FIRST LAUNCH CONFIGURATION'), findsOneWidget);

    // Verify presence of download option buttons
    expect(find.text('DOWNLOAD'), findsOneWidget);
    expect(find.text('LOCAL BYPASS'), findsOneWidget);
  });
}
