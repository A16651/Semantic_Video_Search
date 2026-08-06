// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:media_core_ffi/main.dart';
import 'package:media_core_ffi/gallery_screen.dart';

void main() {
  testWidgets('Smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const CyberNeuralApp());

    // Verify that our ModelLoaderScreen loads with correct titles.
    expect(find.text('NEURAL MODEL LOADER'), findsOneWidget);
    expect(find.text('FIRST LAUNCH CONFIGURATION'), findsOneWidget);
  });
}
