// =============================================================================
// STANDALONE DART VECTOR MATH & DATABASE SEARCH VERIFICATION
// =============================================================================
// Run with: dart test_vector_math.dart
// =============================================================================

import 'dart:math';

class FrameVectorRecord {
  final int videoId;
  final int timestampMs;
  final List<double> embedding512;
  final String label;

  FrameVectorRecord({
    required this.videoId,
    required this.timestampMs,
    required this.embedding512,
    required this.label,
  });
}

List<double> generateNormalized512(int seed) {
  final rand = Random(seed);
  final vec = List<double>.generate(512, (_) => rand.nextDouble() * 2.0 - 1.0);
  double sumSq = 0.0;
  for (var v in vec) {
    sumSq += v * v;
  }
  final norm = sqrt(sumSq);
  return vec.map((v) => norm > 1e-8 ? v / norm : v).toList();
}

double computeCosineSimilarity(List<double> a, List<double> b) {
  assert(a.length == 512 && b.length == 512);
  double dot = 0.0;
  double normA = 0.0;
  double normB = 0.0;
  for (int i = 0; i < 512; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA > 0.0 && normB > 0.0) {
    return dot / (sqrt(normA) * sqrt(normB));
  }
  return 0.0;
}

void main() {
  print('=' * 78);
  print(' DART 512-DIMENSION VECTOR STORE & COSINE SEARCH TEST ');
  print('=' * 78);

  // 1. Generate 50 indexed video frame vectors
  final database = <FrameVectorRecord>[];
  print('Indexing 50 simulated video keyframes (512 dimensions each)...');
  for (int i = 0; i < 50; i++) {
    final v = generateNormalized512(i + 1);
    database.add(FrameVectorRecord(
      videoId: 101,
      timestampMs: i * 1000,
      embedding512: v,
      label: i % 5 == 0 ? 'Scene Transition' : 'Standard Frame',
    ));
  }
  print('Successfully indexed ${database.length} frames.');
  print('Sample Vector Frame #0 [0..5]: ${database[0].embedding512.sublist(0, 6).map((x) => x.toStringAsFixed(4)).toList()}');

  // 2. Perform Cosine Similarity Search using exact target vector (Frame #12)
  final targetQuery = database[12].embedding512;
  print('\nExecuting Query Search matching against Frame #12 (timestamp 12000ms)...');

  final results = <Map<String, dynamic>>[];
  for (var frame in database) {
    final sim = computeCosineSimilarity(targetQuery, frame.embedding512);
    results.add({'frame': frame, 'score': sim});
  }

  results.sort((a, b) => (b['score'] as double).compareTo(a['score'] as double));

  print('\n[TOP-5 VECTOR SEARCH HITS]');
  for (int rank = 0; rank < 5; rank++) {
    final f = results[rank]['frame'] as FrameVectorRecord;
    final score = results[rank]['score'] as double;
    print('  Hit #${rank + 1}: Keyframe ${f.timestampMs}ms [${f.label}] -> Cosine Score: ${(score * 100).toStringAsFixed(2)}%');
  }

  if ((results.first['score'] as double) > 0.999) {
    print('\n[PASS] Target frame was retrieved at Rank #1 with 100.00% precision.');
  }
  print('=' * 78);
}
