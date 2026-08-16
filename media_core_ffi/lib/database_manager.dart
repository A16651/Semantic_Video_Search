import 'dart:math';
import 'dart:io';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';

// Relational & Vector entity mappings for on-device metadata.
// Enforces exactly 512-dimension visual & textual vectors with disk serialization.

class IndexedVideo {
  int id;
  String filePath;
  String fileName;
  int durationMs;
  int sizeBytes;
  DateTime indexedTime;
  String thumbnailPath;

  IndexedVideo({
    this.id = 0,
    required this.filePath,
    required this.fileName,
    required this.durationMs,
    required this.sizeBytes,
    required this.indexedTime,
    this.thumbnailPath = '',
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'filePath': filePath,
        'fileName': fileName,
        'durationMs': durationMs,
        'sizeBytes': sizeBytes,
        'indexedTime': indexedTime.toIso8601String(),
        'thumbnailPath': thumbnailPath,
      };

  factory IndexedVideo.fromJson(Map<String, dynamic> json) => IndexedVideo(
        id: json['id'] ?? 0,
        filePath: json['filePath'] ?? '',
        fileName: json['fileName'] ?? '',
        durationMs: json['durationMs'] ?? 0,
        sizeBytes: json['sizeBytes'] ?? 0,
        indexedTime: DateTime.tryParse(json['indexedTime'] ?? '') ?? DateTime.now(),
        thumbnailPath: json['thumbnailPath'] ?? '',
      );
}

class IndexedPhoto {
  int id;
  String filePath;
  String fileName;
  int sizeBytes;
  DateTime indexedTime;
  List<double> embedding512; // Enforced to 512 dimensions for KNN indexes
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

  Map<String, dynamic> toJson() => {
        'id': id,
        'filePath': filePath,
        'fileName': fileName,
        'sizeBytes': sizeBytes,
        'indexedTime': indexedTime.toIso8601String(),
        'embedding512': embedding512,
        'detectedObjects': detectedObjects,
      };

  factory IndexedPhoto.fromJson(Map<String, dynamic> json) => IndexedPhoto(
        id: json['id'] ?? 0,
        filePath: json['filePath'] ?? '',
        fileName: json['fileName'] ?? '',
        sizeBytes: json['sizeBytes'] ?? 0,
        indexedTime: DateTime.tryParse(json['indexedTime'] ?? '') ?? DateTime.now(),
        embedding512: (json['embedding512'] as List<dynamic>?)?.map((e) => (e as num).toDouble()).toList() ??
            List<double>.filled(512, 0.0),
        detectedObjects: json['detectedObjects'] ?? '[]',
      );
}

class VideoFrameIndex {
  int id;
  int videoId;
  int timestampMs;
  List<double> embedding512; // Enforced to 512 dimensions for KNN indexes
  String detectedObjects; // JSON-serialized tracking labels
  String detectedFaces; // JSON-serialized face vectors/names
  String ocrText; // Real PP-OCR extracted text strings

  VideoFrameIndex({
    this.id = 0,
    required this.videoId,
    required this.timestampMs,
    required this.embedding512,
    this.detectedObjects = '[]',
    this.detectedFaces = '[]',
    this.ocrText = '',
  }) {
    assert(embedding512.length == 512, "Visual database vectors must be strictly mapped to 512 dimensions.");
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'videoId': videoId,
        'timestampMs': timestampMs,
        'embedding512': embedding512,
        'detectedObjects': detectedObjects,
        'detectedFaces': detectedFaces,
        'ocrText': ocrText,
      };

  factory VideoFrameIndex.fromJson(Map<String, dynamic> json) => VideoFrameIndex(
        id: json['id'] ?? 0,
        videoId: json['videoId'] ?? 0,
        timestampMs: json['timestampMs'] ?? 0,
        embedding512: (json['embedding512'] as List<dynamic>?)?.map((e) => (e as num).toDouble()).toList() ??
            List<double>.filled(512, 0.0),
        detectedObjects: json['detectedObjects'] ?? '[]',
        detectedFaces: json['detectedFaces'] ?? '[]',
        ocrText: json['ocrText'] ?? '',
      );
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

  Map<String, dynamic> toJson() => {
        'id': id,
        'videoId': videoId,
        'timestampStartMs': timestampStartMs,
        'timestampEndMs': timestampEndMs,
        'sentence': sentence,
        'textEmbedding512': textEmbedding512,
      };

  factory AudioTranscriptIndex.fromJson(Map<String, dynamic> json) => AudioTranscriptIndex(
        id: json['id'] ?? 0,
        videoId: json['videoId'] ?? 0,
        timestampStartMs: json['timestampStartMs'] ?? 0,
        timestampEndMs: json['timestampEndMs'] ?? 0,
        sentence: json['sentence'] ?? '',
        textEmbedding512: (json['textEmbedding512'] as List<dynamic>?)?.map((e) => (e as num).toDouble()).toList() ??
            List<double>.filled(512, 0.0),
      );
}

class DatabaseManager {
  static final List<IndexedVideo> _videos = [];
  static final List<IndexedPhoto> _photos = [];
  static final List<VideoFrameIndex> _frames = [];
  static final List<AudioTranscriptIndex> _transcripts = [];
  static bool _initialized = false;

  static void dbLog(String message) {
    if (kDebugMode) {
      print("[DatabaseManager] $message");
    }
  }

  static Future<void> initialize() async {
    if (_initialized) return;
    _initialized = true;
    dbLog("Initializing DatabaseManager and loading persisted state from disk...");
    await loadDatabaseFromDisk();
  }

  static Future<void> loadDatabaseFromDisk() async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final dbFile = File('${dir.path}/app_database_v1.json');
      dbLog("Reading persistent database file: ${dbFile.path}");

      if (await dbFile.exists()) {
        final String content = await dbFile.readAsString();
        final Map<String, dynamic> json = jsonDecode(content);

        _videos.clear();
        _photos.clear();
        _frames.clear();
        _transcripts.clear();

        if (json.containsKey('videos')) {
          for (var item in json['videos']) {
            _videos.add(IndexedVideo.fromJson(item));
          }
        }
        if (json.containsKey('photos')) {
          for (var item in json['photos']) {
            _photos.add(IndexedPhoto.fromJson(item));
          }
        }
        if (json.containsKey('frames')) {
          for (var item in json['frames']) {
            _frames.add(VideoFrameIndex.fromJson(item));
          }
        }
        if (json.containsKey('transcripts')) {
          for (var item in json['transcripts']) {
            _transcripts.add(AudioTranscriptIndex.fromJson(item));
          }
        }

        dbLog("Successfully restored from disk: ${_videos.length} videos, ${_photos.length} photos, ${_frames.length} frames, ${_transcripts.length} transcripts.");
      } else {
        dbLog("No existing database file found on disk. Initialized empty database.");
      }
    } catch (e, st) {
      dbLog("Error loading database from disk: $e\n$st");
    }
  }

  static Future<void> saveDatabaseToDisk() async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final dbFile = File('${dir.path}/app_database_v1.json');
      dbLog("Saving database state to disk: ${_videos.length} videos, ${_photos.length} photos, ${_frames.length} frames, ${_transcripts.length} transcripts...");

      final Map<String, dynamic> data = {
        'videos': _videos.map((v) => v.toJson()).toList(),
        'photos': _photos.map((p) => p.toJson()).toList(),
        'frames': _frames.map((f) => f.toJson()).toList(),
        'transcripts': _transcripts.map((t) => t.toJson()).toList(),
      };

      await dbFile.writeAsString(jsonEncode(data));
      dbLog("Database saved to disk successfully at ${dbFile.path}");
    } catch (e, st) {
      dbLog("Error saving database to disk: $e\n$st");
    }
  }

  static List<IndexedVideo> getAllVideos() {
    return _videos;
  }

  static List<IndexedPhoto> getAllPhotos() {
    return _photos;
  }

  static void addVideo(IndexedVideo video) {
    if (!_videos.any((v) => v.id == video.id || v.filePath == video.filePath)) {
      _videos.add(video);
      dbLog("Added video: id=${video.id}, path=${video.filePath}. Total videos: ${_videos.length}");
      saveDatabaseToDisk();
    }
  }

  static void addPhoto(IndexedPhoto photo) {
    if (!_photos.any((p) => p.id == photo.id || p.filePath == photo.filePath)) {
      _photos.add(photo);
      dbLog("Added photo: id=${photo.id}, path=${photo.filePath}. Total photos: ${_photos.length}");
      saveDatabaseToDisk();
    }
  }

  static void clearVideoData(int videoId) {
    dbLog("Clearing existing frames and transcripts for videoId=$videoId");
    _frames.removeWhere((f) => f.videoId == videoId);
    _transcripts.removeWhere((t) => t.videoId == videoId);
    saveDatabaseToDisk();
  }

  static void addFrame(VideoFrameIndex frame) {
    _frames.removeWhere((f) => f.videoId == frame.videoId && f.timestampMs == frame.timestampMs);
    _frames.add(frame);
    dbLog("Added VideoFrameIndex for video ${frame.videoId} at ${frame.timestampMs}ms (ocr: '${frame.ocrText}'). Total frames: ${_frames.length}");
    saveDatabaseToDisk();
  }

  static void addTranscript(AudioTranscriptIndex transcript) {
    _transcripts.removeWhere((t) =>
        t.videoId == transcript.videoId &&
        t.timestampStartMs == transcript.timestampStartMs &&
        t.timestampEndMs == transcript.timestampEndMs);
    _transcripts.add(transcript);
    dbLog("Added AudioTranscriptIndex for video ${transcript.videoId} [${transcript.timestampStartMs}-${transcript.timestampEndMs}ms]: '${transcript.sentence}'. Total transcripts: ${_transcripts.length}");
    saveDatabaseToDisk();
  }

  static List<AudioTranscriptIndex> getTranscriptsForVideo(int videoId) {
    final list = _transcripts.where((t) => t.videoId == videoId).toList();
    list.sort((a, b) => a.timestampStartMs.compareTo(b.timestampStartMs));
    return list;
  }

  static List<VideoFrameIndex> getFramesForVideo(int videoId) {
    final list = _frames.where((f) => f.videoId == videoId).toList();
    list.sort((a, b) => a.timestampMs.compareTo(b.timestampMs));
    return list;
  }

  static List<Map<String, dynamic>> searchVisualSemantic(List<double> queryVector, {double minConfidence = 0.15}) {
    assert(queryVector.length == 512, "Query vectors for similarity matching must be exactly 512 dimensions.");
    dbLog("Performing visual semantic similarity search across ${_frames.length} frames and ${_photos.length} photos...");

    final List<Map<String, dynamic>> hits = [];

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

      if (score >= minConfidence && _videos.any((v) => v.id == f.videoId)) {
        final video = _videos.firstWhere((v) => v.id == f.videoId);
        hits.add({
          'type': 'video',
          'video': video,
          'frame': f,
          'score': score,
        });
      }
    }

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

    hits.sort((a, b) => (b['score'] as double).compareTo(a['score'] as double));
    dbLog("Visual semantic search returned ${hits.length} matching hits.");
    return hits;
  }

  static List<Map<String, dynamic>> searchAudioSemantic(List<double> queryVector, {double minConfidence = 0.15}) {
    assert(queryVector.length == 512, "Query vectors for similarity matching must be exactly 512 dimensions.");
    dbLog("Performing audio transcript similarity search across ${_transcripts.length} transcripts...");

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

      if (score >= minConfidence && _videos.any((v) => v.id == t.videoId)) {
        final video = _videos.firstWhere((v) => v.id == t.videoId);
        hits.add({
          'video': video,
          'transcript': t,
          'score': score,
        });
      }
    }

    hits.sort((a, b) => (b['score'] as double).compareTo(a['score'] as double));
    dbLog("Audio transcript search returned ${hits.length} matching hits.");
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
        dbLog("Loaded ${_watchedDirectories.length} watched directories from disk.");
      }
    } catch (e) {
      dbLog("Error loading watched directories: $e");
    }
  }

  static Future<void> saveWatchedDirectories() async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final file = File('${dir.path}/watched_directories.json');
      await file.writeAsString(jsonEncode(_watchedDirectories));
      dbLog("Saved ${_watchedDirectories.length} watched directories to disk.");
    } catch (e) {
      dbLog("Error saving watched directories: $e");
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
