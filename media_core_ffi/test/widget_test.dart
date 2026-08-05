import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:media_core_ffi/gallery_screen.dart';

void main() {
  testWidgets('Neural Model Loader Screen Smoke Test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const CyberNeuralApp());

    // Verify that Neural Model Loader screen renders successfully.
    expect(find.text('NEURAL MODEL LOADER'), findsOneWidget);
    expect(find.text('FIRST LAUNCH CONFIGURATION'), findsOneWidget);
    expect(find.byType(LinearProgressIndicator), findsOneWidget);
  });
}
