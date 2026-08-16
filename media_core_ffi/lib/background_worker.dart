import 'dart:isolate';
import 'worker_entry.dart';
import 'database_manager.dart';

// Task description sent to background Isolate
class BackgroundIngestionTask {
  final int videoId;
  final String videoPath;
  final bool processFrames;
  final bool processAudio;
  final bool processOcr;
  final String modelDir; // Absolute path to local_models dir for C++ path resolution
  final SendPort replyPort;

  BackgroundIngestionTask({
    required this.videoId,
    required this.videoPath,
    required this.processFrames,
    required this.processAudio,
    required this.processOcr,
    required this.modelDir,
    required this.replyPort,
  });
}

// Ingestion feedback sent from isolate back to UI thread
class IngestionProgress {
  final String videoPath;
  final double progress; // 0.0 to 1.0
  final String currentAction;
  final bool completed;
  final String? error;

  // Computed results — populated only when completed == true
  // Carried back through reply channel to bypass isolate memory isolation
  final List<VideoFrameIndex>? computedFrames;
  final List<AudioTranscriptIndex>? computedTranscripts;

  IngestionProgress({
    required this.videoPath,
    required this.progress,
    required this.currentAction,
    this.completed = false,
    this.error,
    this.computedFrames,
    this.computedTranscripts,
  });
}

class BackgroundWorker {
  static Isolate? _workerIsolate;
  static ReceivePort? _receivePort;

  // Spawns and configures the long-running background ingestion Isolate
  static Future<void> startWorker(
    int videoId,
    String videoPath, {
    bool processFrames = true,
    bool processAudio = true,
    bool processOcr = true,
    required String modelDir,
    required Function(IngestionProgress) onProgress,
  }) async {
    // If a worker is already running, stop it first
    stopWorker();

    _receivePort = ReceivePort();

    _receivePort!.listen((message) {
      if (message is IngestionProgress) {
        onProgress(message);
      }
    });

    final task = BackgroundIngestionTask(
      videoId: videoId,
      videoPath: videoPath,
      processFrames: processFrames,
      processAudio: processAudio,
      processOcr: processOcr,
      modelDir: modelDir,
      replyPort: _receivePort!.sendPort,
    );

    _workerIsolate = await Isolate.spawn(workerEntryPoint, task);
  }

  static void stopWorker() {
    if (_workerIsolate != null) {
      _workerIsolate!.kill(priority: Isolate.immediate);
      _workerIsolate = null;
    }
    if (_receivePort != null) {
      _receivePort!.close();
      _receivePort = null;
    }
  }
}
