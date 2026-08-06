import 'dart:isolate';
import 'worker_entry.dart';

// Task description sent to background Isolate
class BackgroundIngestionTask {
  final String videoPath;
  final bool processFrames;
  final bool processAudio;
  final bool processOcr;
  final SendPort replyPort;

  BackgroundIngestionTask({
    required this.videoPath,
    required this.processFrames,
    required this.processAudio,
    required this.processOcr,
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

  IngestionProgress({
    required this.videoPath,
    required this.progress,
    required this.currentAction,
    this.completed = false,
    this.error,
  });
}

class BackgroundWorker {
  static Isolate? _workerIsolate;
  static ReceivePort? _receivePort;

  // Spawns and configures the long-running background ingestion Isolate
  static Future<void> startWorker(String videoPath, {
    bool processFrames = true,
    bool processAudio = true,
    bool processOcr = true,
    required Function(IngestionProgress) onProgress,
  }) async {
    // If there is an existing worker running, stop it first
    stopWorker();

    _receivePort = ReceivePort();

    _receivePort!.listen((message) {
      if (message is IngestionProgress) {
        onProgress(message);
      }
    });

    final task = BackgroundIngestionTask(
      videoPath: videoPath,
      processFrames: processFrames,
      processAudio: processAudio,
      processOcr: processOcr,
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
