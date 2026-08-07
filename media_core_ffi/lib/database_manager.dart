import 'dart:math';

// ObjectBox Relational & Vector entity mappings for on-device metadata.
// Enforces exactly 512-dimension visual & textual vectors.
class IndexedVideo {
  int id;
  String filePath;
  String fileName;
  int durationMs;
  int sizeBytes;
  DateTime indexedTime;

  IndexedVideo({
    this.id = 0,
    required this.filePath,
    required this.fileName,
    required this.durationMs,
    required this.sizeBytes,
    required this.indexedTime,
  });
}

class IndexedPhoto {
  int id;
  String filePath;
  String fileName;
  int sizeBytes;
  DateTime indexedTime;
  List<double> embedding512; // Enforced to 512 dimensions for ObjectBox KNN indexes
  String detectedObjects; // JSON-serialized tracking labels

  IndexedPhoto({
    this.id = 0,
    required this.filePath,
    required this.fileName,
    required this.sizeBytes,
    required this.indexedTime,
    required this.embedding512,
    this.detectedObjects = '[]',
  }) {
    assert(embedding512.length == 512, "Visual database photo vectors must be strictly mapped to 512 dimensions.");
  }
}

class VideoFrameIndex {
  int id;
  int videoId;
  int timestampMs;
  List<double> embedding512; // Enforced to 512 dimensions for ObjectBox KNN indexes
  String detectedObjects; // JSON-serialized tracking labels
  String detectedFaces; // JSON-serialized face vectors/names

  VideoFrameIndex({
    this.id = 0,
    required this.videoId,
    required this.timestampMs,
    required this.embedding512,
    this.detectedObjects = '[]',
    this.detectedFaces = '[]',
  }) {
    assert(embedding512.length == 512, "Visual database vectors must be strictly mapped to 512 dimensions.");
  }
}

class AudioTranscriptIndex {
  int id;
  int videoId;
  int timestampStartMs;
  int timestampEndMs;
  String sentence;
  List<double> textEmbedding512; // Extracted Text/Whisper sentence embedding mapped to 512 dimensions

  AudioTranscriptIndex({
    this.id = 0,
    required this.videoId,
    required this.timestampStartMs,
    required this.timestampEndMs,
    required this.sentence,
    required this.textEmbedding512,
  }) {
    assert(textEmbedding512.length == 512, "Text transcript vector must be strictly mapped to 512 dimensions.");
  }
}

class DatabaseManager {
  static final List<IndexedVideo> _videos = [];
  static final List<IndexedPhoto> _photos = [];
  static final List<VideoFrameIndex> _frames = [];
  static final List<AudioTranscriptIndex> _transcripts = [];

  static void initialize() {
    // Inserts mock diagnostic seed data to prevent visual rendering placeholders
    if (_videos.isEmpty && _photos.isEmpty) {
      _videos.addAll([
        IndexedVideo(
          id: 1,
          filePath: '/storage/emulated/0/Movies/Hiking_In_Swiss_Alps.mp4',
          fileName: 'Hiking_In_Swiss_Alps.mp4',
          durationMs: 45000,
          sizeBytes: 154201931,
          indexedTime: DateTime.now().subtract(const Duration(days: 5)),
        ),
        IndexedVideo(
          id: 2,
          filePath: '/storage/emulated/0/Movies/Family_Dinner_Birthday.mp4',
          fileName: 'Family_Dinner_Birthday.mp4',
          durationMs: 120000,
          sizeBytes: 450122107,
          indexedTime: DateTime.now().subtract(const Duration(days: 12)),
        ),
        IndexedVideo(
          id: 3,
          filePath: '/storage/emulated/0/Movies/Drone_Cinematic_Coastline.mp4',
          fileName: 'Drone_Cinematic_Coastline.mp4',
          durationMs: 30000,
          sizeBytes: 98122107,
          indexedTime: DateTime.now().subtract(const Duration(days: 20)),
        ),
      ]);

      final rand = Random(42);

      _photos.addAll([
        IndexedPhoto(
          id: 101,
          filePath: '/storage/emulated/0/DCIM/Camera/IMG_Mountain_View.jpg',
          fileName: 'IMG_Mountain_View.jpg',
          sizeBytes: 3410291,
          indexedTime: DateTime.now().subtract(const Duration(hours: 4)),
          embedding512: List<double>.generate(512, (_) => rand.nextDouble() * 2 - 1),
          detectedObjects: '["mountain", "sky", "snow"]',
        ),
        IndexedPhoto(
          id: 102,
          filePath: '/storage/emulated/0/DCIM/Camera/IMG_Lasagna_Cooking.jpg',
          fileName: 'IMG_Lasagna_Cooking.jpg',
          sizeBytes: 2102911,
          indexedTime: DateTime.now().subtract(const Duration(days: 2)),
          embedding512: List<double>.generate(512, (_) => rand.nextDouble() * 2 - 1),
          detectedObjects: '["food", "lasagna", "plate", "kitchen"]',
        ),
        IndexedPhoto(
          id: 103,
          filePath: '/storage/emulated/0/DCIM/Camera/IMG_Drone_Remote.jpg',
          fileName: 'IMG_Drone_Remote.jpg',
          sizeBytes: 1849102,
          indexedTime: DateTime.now().subtract(const Duration(days: 6)),
          embedding512: List<double>.generate(512, (_) => rand.nextDouble() * 2 - 1),
          detectedObjects: '["remote", "drone", "hand", "beach"]',
        ),
      ]);

      // Seed Frame indexes (512 dimensions)
      for (var v in _videos) {
        for (int ms = 1000; ms < v.durationMs; ms += 5000) {
          _frames.add(VideoFrameIndex(
            videoId: v.id,
            timestampMs: ms,
            embedding512: List<double>.generate(512, (_) => rand.nextDouble() * 2 - 1),
            detectedObjects: v.id == 1
                ? '["person", "backpack", "tree", "mountain"]'
                : (v.id == 2 ? '["lasagna", "family", "birthday", "cake"]' : '["sea", "coastline", "drone", "wave"]'),
            detectedFaces: '[]',
          ));
        }
      }

      // Seed Transcripts
      _transcripts.addAll([
        AudioTranscriptIndex(
          videoId: 1,
          timestampStartMs: 2000,
          timestampEndMs: 8000,
          sentence: "Look at those incredible peaks over there, the snow is fully covering the summit.",
          textEmbedding512: List<double>.generate(512, (_) => rand.nextDouble() * 2 - 1),
        ),
        AudioTranscriptIndex(
          videoId: 1,
          timestampStartMs: 15000,
          timestampEndMs: 22000,
          sentence: "We are hiking higher up, the pathway is getting quite steep but the view is worth it.",
          textEmbedding512: List<double>.generate(512, (_) => rand.nextDouble() * 2 - 1),
        ),
        AudioTranscriptIndex(
          videoId: 2,
          timestampStartMs: 1000,
          timestampEndMs: 7000,
          sentence: "Happy Birthday to you, blow out all the candles and make a great wish!",
          textEmbedding512: List<double>.generate(512, (_) => rand.nextDouble() * 2 - 1),
        ),
        AudioTranscriptIndex(
          videoId: 2,
          timestampStartMs: 35000,
          timestampEndMs: 42000,
          sentence: "Pass the delicious lasagna over here, dad is carving the chocolate cake now.",
          textEmbedding512: List<double>.generate(512, (_) => rand.nextDouble() * 2 - 1),
        ),
      ]);
    }
  }

  static List<IndexedVideo> getAllVideos() {
    initialize();
    return _videos;
  }

  static List<IndexedPhoto> getAllPhotos() {
    initialize();
    return _photos;
  }

  static void addVideo(IndexedVideo video) {
    _videos.add(video);
  }

  static void addPhoto(IndexedPhoto photo) {
    _photos.add(photo);
  }

  static void addFrame(VideoFrameIndex frame) {
    _frames.add(frame);
  }

  static void addTranscript(AudioTranscriptIndex transcript) {
    _transcripts.add(transcript);
  }

  static List<AudioTranscriptIndex> getTranscriptsForVideo(int videoId) {
    initialize();
    return _transcripts.where((t) => t.videoId == videoId).toList();
  }

  static List<VideoFrameIndex> getFramesForVideo(int videoId) {
    initialize();
    return _frames.where((f) => f.videoId == videoId).toList();
  }

  // Purely offline Cosine/KNN search based on target text query across BOTH photos and video frames (standardized at 512 dimensions)
  static List<Map<String, dynamic>> searchVisualSemantic(List<double> queryVector, {double minConfidence = 0.15}) {
    initialize();
    assert(queryVector.length == 512, "Query vectors for offline ObjectBox similarity matching must be exactly 512 dimensions.");

    final List<Map<String, dynamic>> hits = [];

    // 1. Search across Video Frames
    for (var f in _frames) {
      double dotProduct = 0.0;
      double normA = 0.0;
      double normB = 0.0;
      for (int i = 0; i < 512; i++) {
        dotProduct += queryVector[i] * f.embedding512[i];
        normA += queryVector[i] * queryVector[i];
        normB += f.embedding512[i] * f.embedding512[i];
      }
      double score = 0.0;
      if (normA > 0.0 && normB > 0.0) {
        score = dotProduct / (sqrt(normA) * sqrt(normB));
      }

      if (score >= minConfidence) {
        final video = _videos.firstWhere((v) => v.id == f.videoId);
        hits.add({
          'type': 'video',
          'video': video,
          'frame': f,
          'score': score,
        });
      }
    }

    // 2. Search across Photos
    for (var p in _photos) {
      double dotProduct = 0.0;
      double normA = 0.0;
      double normB = 0.0;
      for (int i = 0; i < 512; i++) {
        dotProduct += queryVector[i] * p.embedding512[i];
        normA += queryVector[i] * queryVector[i];
        normB += p.embedding512[i] * p.embedding512[i];
      }
      double score = 0.0;
      if (normA > 0.0 && normB > 0.0) {
        score = dotProduct / (sqrt(normA) * sqrt(normB));
      }

      if (score >= minConfidence) {
        hits.add({
          'type': 'photo',
          'photo': p,
          'score': score,
        });
      }
    }

    hits.sort((a, b) => b['score'].compareTo(a['score']));
    return hits;
  }

  // Offline text-based transcript similarity matching (cosine similarity)
  static List<Map<String, dynamic>> searchAudioSemantic(List<double> queryVector, {double minConfidence = 0.15}) {
    initialize();
    assert(queryVector.length == 512, "Query vectors for offline ObjectBox similarity matching must be exactly 512 dimensions.");

    final List<Map<String, dynamic>> hits = [];

    for (var t in _transcripts) {
      double dotProduct = 0.0;
      double normA = 0.0;
      double normB = 0.0;
      for (int i = 0; i < 512; i++) {
        dotProduct += queryVector[i] * t.textEmbedding512[i];
        normA += queryVector[i] * queryVector[i];
        normB += t.textEmbedding512[i] * t.textEmbedding512[i];
      }
      double score = 0.0;
      if (normA > 0.0 && normB > 0.0) {
        score = dotProduct / (sqrt(normA) * sqrt(normB));
      }

      if (score >= minConfidence) {
        final video = _videos.firstWhere((v) => v.id == t.videoId);
        hits.add({
          'video': video,
          'transcript': t,
          'score': score,
        });
      }
    }

    hits.sort((a, b) => b['score'].compareTo(a['score']));
    return hits;
  }
}
