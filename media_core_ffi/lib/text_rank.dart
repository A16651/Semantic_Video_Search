import 'dart:math';

// Implements extractive TextRank text/transcript summarization based on local vector similarity.
// Computes cosine-similarity on 512-dimension sentence embeddings.
class ExtractiveTextRank {

  // Computes the Cosine Similarity metric between two 512-dimension vectors
  static double cosineSimilarity(List<double> vecA, List<double> vecB) {
    if (vecA.length != 512 || vecB.length != 512) {
      throw ArgumentError("TextRank calculation expects strictly 512-dimension neural representations.");
    }
    double dotProduct = 0.0;
    double normA = 0.0;
    double normB = 0.0;
    for (int i = 0; i < 512; i++) {
      dotProduct += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }
    if (normA == 0.0 || normB == 0.0) return 0.0;
    return dotProduct / (sqrt(normA) * sqrt(normB));
  }

  // Extracts key sentences from video transcripts without utilizing external generative LLMs
  static List<String> extractSummary(List<String> sentences, List<List<double>> embeddings, {int numSentences = 3}) {
    if (sentences.isEmpty || embeddings.isEmpty) return [];
    if (sentences.length != embeddings.length) {
      throw ArgumentError("The sentence count must match the total neural embeddings generated.");
    }

    final int n = sentences.length;
    final List<double> scores = List<double>.filled(n, 1.0);
    final List<List<double>> similarityMatrix = List.generate(n, (_) => List<double>.filled(n, 0.0));

    // 1. Build Similarity Matrix via exact pairwise Cosine Similarity calculations
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (i == j) {
          similarityMatrix[i][j] = 0.0;
        } else {
          similarityMatrix[i][j] = cosineSimilarity(embeddings[i], embeddings[j]);
        }
      }
    }

    // 2. Perform iterative PageRank / TextRank convergence (10 iterations is usually sufficient)
    const double d = 0.85; // Damping factor
    for (int iter = 0; iter < 10; iter++) {
      final List<double> nextScores = List<double>.filled(n, 0.0);
      for (int i = 0; i < n; i++) {
        double sumIn = 0.0;
        for (int j = 0; j < n; j++) {
          if (similarityMatrix[j][i] > 0) {
            // Out-degree calculation (sum of weights for node j)
            double outSum = 0.0;
            for (int k = 0; k < n; k++) {
              outSum += similarityMatrix[j][k];
            }
            if (outSum > 0) {
              sumIn += (similarityMatrix[j][i] / outSum) * scores[j];
            }
          }
        }
        nextScores[i] = (1.0 - d) + d * sumIn;
      }
      for (int i = 0; i < n; i++) {
        scores[i] = nextScores[i];
      }
    }

    // 3. Rank sentences based on computed convergence scores
    final List<MapEntry<int, double>> ranked = [];
    for (int i = 0; i < n; i++) {
      ranked.add(MapEntry(i, scores[i]));
    }
    ranked.sort((a, b) => b.value.compareTo(a.value));

    // 4. Retrieve top sentences preserving their chronological order in the video
    final List<int> topIndices = ranked.take(numSentences).map((e) => e.key).toList();
    topIndices.sort();

    return topIndices.map((idx) => sentences[idx]).toList();
  }
}
