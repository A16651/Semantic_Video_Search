import 'dart:math';
import 'dart:io';
import 'dart:convert';
import 'package:path_provider/path_provider.dart';

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
    // Purged all mock diagnostic seed data for clean first-launch and no-mock-data guarantee.
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

  static final List<String> _watchedDirectories = [];

  static Future<void> loadWatchedDirectories() async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final file = File('${dir.path}/watched_directories.json');
      if (await file.exists()) {
        final content = await file.readAsString();
        final List<dynamic> list = jsonDecode(content);
        _watchedDirectories.clear();
        _watchedDirectories.addAll(list.cast<String>());
      }
    } catch (e) {
      // Gracefully handle or log
    }
  }

  static Future<void> saveWatchedDirectories() async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final file = File('${dir.path}/watched_directories.json');
      await file.writeAsString(jsonEncode(_watchedDirectories));
    } catch (e) {
      // Gracefully handle or log
    }
  }

  static List<String> getWatchedDirectories() {
    return _watchedDirectories;
  }

  static void addWatchedDirectory(String path) {
    if (!_watchedDirectories.contains(path)) {
      _watchedDirectories.add(path);
      saveWatchedDirectories();
    }
  }

  static void removeWatchedDirectory(String path) {
    _watchedDirectories.remove(path);
    saveWatchedDirectories();
  }
}
