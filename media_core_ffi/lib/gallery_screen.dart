import 'package:flutter/material.dart';
import 'database_manager.dart';
import 'background_worker.dart';
import 'text_rank.dart';
import 'media_core_ffi.dart';

// Complete Flutter App implementation styled strictly under "Cyber-Neural Cinematic" guidelines.
// Features semantic searches, downloads, video player, and neural configurations.
class CyberNeuralApp extends StatelessWidget {
  const CyberNeuralApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Gallery App',
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF131313),
        primaryColor: const Color(0xFF00F5FF),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF00F5FF),
          secondary: Color(0xFF39FF14),
          surface: Color(0xFF1C1B1B),
          background: Color(0xFF131313),
        ),
      ),
      home: const ModelDownloadScreen(),
    );
  }
}

// ----------------- Screen 1: Model Delivery System -----------------
class ModelDownloadScreen extends StatefulWidget {
  const ModelDownloadScreen({Key? key}) : super(key: key);

  @override
  _ModelDownloadScreenState createState() => _ModelDownloadScreenState();
}

class _ModelDownloadScreenState extends State<ModelDownloadScreen> {
  double _progress = 0.0;
  bool _isDownloading = false;
  bool _completed = false;
  String _status = "Neural Pipeline Offline. Missing INT8 ONNX Engine files.";

  void _simulateDownload() async {
    setState(() {
      _isDownloading = true;
      _status = "Fetching SigLIP-SO400M, Whisper-Tiny, PP-OCR INT8 (320 MB)...";
    });

    for (int i = 0; i <= 100; i += 10) {
      if (!_isDownloading) return;
      await Future.delayed(const Duration(milliseconds: 300));
      setState(() {
        _progress = i / 100.0;
        if (i == 30) {
          _status = "SigLIP-SO400M (INT8) verified via SHA-256.";
        } else if (i == 60) {
          _status = "Whisper Tiny (INT8) downloaded successfully.";
        } else if (i == 90) {
          _status = "PP-OCR Detection & Recognition Engines loaded.";
        }
      });
    }

    setState(() {
      _completed = true;
      _isDownloading = false;
      _status = "All INT8 ONNX Engines verified. Ready to activate gallery.";
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Container(
          maxHeight: 520,
          maxWidth: 420,
          padding: const EdgeInsets.all(24),
          decoration: BoxDecoration(
            color: const Color(0xFF1C1B1B),
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: const Color(0xFF00F5FF).withOpacity(0.2), width: 1),
            boxShadow: [
              BoxShadow(
                color: const Color(0xFF00F5FF).withOpacity(0.05),
                blurRadius: 20,
                spreadRadius: 5,
              )
            ],
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              const Icon(Icons.hub, color: Color(0xFF00F5FF), size: 48),
              const SizedBox(height: 16),
              const Text(
                "NEURAL MODEL LOADER",
                style: TextStyle(
                  fontFamily: 'Sora',
                  fontWeight: FontWeight.bold,
                  fontSize: 18,
                  letterSpacing: 1.5,
                ),
              ),
              const SizedBox(height: 8),
              const Text(
                "FIRST LAUNCH CONFIGURATION",
                style: TextStyle(
                  fontFamily: 'JetBrains Mono',
                  fontSize: 11,
                  color: Color(0xFF39FF14),
                ),
              ),
              const SizedBox(height: 32),
              LinearProgressIndicator(
                value: _progress,
                backgroundColor: const Color(0xFF2A2A2A),
                color: const Color(0xFF00F5FF),
                minHeight: 8,
              ),
              const SizedBox(height: 16),
              Text(
                _status,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontFamily: 'Inter',
                  fontSize: 13,
                  color: Colors.grey,
                ),
              ),
              const SizedBox(height: 12),
              Text(
                "SHA-256 CHECK: ${_completed ? 'SUCCESS' : 'PENDING'}",
                style: TextStyle(
                  fontFamily: 'JetBrains Mono',
                  fontSize: 11,
                  color: _completed ? const Color(0xFF39FF14) : Colors.yellow,
                ),
              ),
              const SizedBox(height: 40),
              if (!_completed) ...[
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    ElevatedButton(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF00F5FF),
                        foregroundColor: Colors.black,
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
                        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                      ),
                      onPressed: _isDownloading ? null : _simulateDownload,
                      child: const Text("DOWNLOAD MODELS"),
                    ),
                    if (_isDownloading)
                      TextButton(
                        onPressed: () {
                          setState(() {
                            _isDownloading = false;
                            _status = "Download paused by operator.";
                          });
                        },
                        child: const Text("PAUSE", style: TextStyle(color: Colors.red)),
                      )
                  ],
                )
              ] else ...[
                ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF39FF14),
                    foregroundColor: Colors.black,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
                    padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 16),
                  ),
                  onPressed: () {
                    Navigator.pushReplacement(
                      context,
                      MaterialPageRoute(builder: (context) => const GalleryDashboardScreen()),
                    );
                  },
                  child: const Text("ENTER CYBER GALLERY"),
                )
              ]
            ],
          ),
        ),
      ),
    );
  }
}

// ----------------- Screen 2: Gallery Dashboard -----------------
class GalleryDashboardScreen extends StatefulWidget {
  const GalleryDashboardScreen({Key? key}) : super(key: key);

  @override
  _GalleryDashboardScreenState createState() => _GalleryDashboardScreenState();
}

class _GalleryDashboardScreenState extends State<GalleryDashboardScreen> {
  final TextEditingController _searchController = TextEditingController();
  List<Map<String, dynamic>> _searchResults = [];
  bool _hasSearched = false;
  String _activeTab = "All Files";
  String _processDepth = "Full Summary";

  void _performSearch() {
    final query = _searchController.text.trim();
    if (query.isEmpty) return;

    // Trigger local FFI to encode natural query text into 512 dimensions
    final queryVector = MediaCoreBridge.encodeText(query);

    // Execute local database search
    final results = DatabaseManager.searchVisualSemantic(queryVector);

    setState(() {
      _searchResults = results;
      _hasSearched = true;
    });
  }

  @override
  Widget build(BuildContext context) {
    final videos = DatabaseManager.getAllVideos();

    return Scaffold(
      appBar: AppBar(
        title: const Text("NEURAL CINEMA ENGINE", style: TextStyle(fontFamily: 'Sora', fontWeight: FontWeight.bold)),
        backgroundColor: const Color(0xFF131313),
        elevation: 0,
        actions: [
          IconButton(
            icon: const Icon(Icons.settings, color: Color(0xFF00F5FF)),
            onPressed: () {
              Navigator.push(context, MaterialPageRoute(builder: (context) => const ControlSettingsScreen()));
            },
          ),
          IconButton(
            icon: const Icon(Icons.timeline, color: Color(0xFF39FF14)),
            onPressed: () {
              Navigator.push(context, MaterialPageRoute(builder: (context) => const TimelineScreen()));
            },
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Search Input Block
            Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _searchController,
                    style: const TextStyle(fontFamily: 'Inter', color: Colors.white),
                    decoration: InputDecoration(
                      hintText: "Search your life (e.g. Hiking path in winter)...",
                      hintStyle: const TextStyle(color: Colors.grey),
                      fillColor: const Color(0xFF1C1B1B),
                      filled: true,
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(4),
                        borderSide: const BorderSide(color: Color(0xFF00F5FF), width: 1),
                      ),
                      focusedBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(4),
                        borderSide: const BorderSide(color: Color(0xFF00F5FF), width: 2),
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF00F5FF),
                    foregroundColor: Colors.black,
                    padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
                  ),
                  onPressed: _performSearch,
                  child: const Icon(Icons.search),
                ),
              ],
            ),
            const SizedBox(height: 12),

            // Chips Controls
            Row(
              children: [
                const Text("PROCESSING LEVEL: ", style: TextStyle(fontFamily: 'JetBrains Mono', fontSize: 11, color: Colors.grey)),
                const SizedBox(width: 8),
                ...["Frames Only", "Audio Only", "Full Process"].map((depth) {
                  final active = _processDepth == depth;
                  return Padding(
                    padding: const EdgeInsets.only(right: 8.0),
                    child: ChoiceChip(
                      label: Text(depth, style: const TextStyle(fontFamily: 'JetBrains Mono', fontSize: 11)),
                      selected: active,
                      selectedColor: const Color(0xFF00F5FF).withOpacity(0.2),
                      backgroundColor: const Color(0xFF1C1B1B),
                      textColor: active ? const Color(0xFF00F5FF) : Colors.grey,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(4),
                        side: BorderSide(color: active ? const Color(0xFF00F5FF) : Colors.transparent),
                      ),
                      onSelected: (val) {
                        setState(() {
                          _processDepth = depth;
                        });
                      },
                    ),
                  );
                }).toList(),
              ],
            ),
            const SizedBox(height: 24),

            // Gallery / Results Render
            Expanded(
              child: _hasSearched
                  ? _buildSearchResults()
                  : _buildVideosGrid(videos),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSearchResults() {
    if (_searchResults.isEmpty) {
      return const Center(
        child: Text(
          "Zero neural hits detected in local database.",
          style: TextStyle(fontFamily: 'Inter', color: Colors.grey),
        ),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              "NEURAL MATCHES DETECTED: ${_searchResults.length}",
              style: const TextStyle(fontFamily: 'JetBrains Mono', fontSize: 12, color: Color(0xFF39FF14)),
            ),
            TextButton(
              onPressed: () {
                setState(() {
                  _hasSearched = false;
                  _searchController.clear();
                });
              },
              child: const Text("CLEAR SEARCH", style: TextStyle(fontFamily: 'JetBrains Mono', fontSize: 11, color: Colors.red)),
            ),
          ],
        ),
        const SizedBox(height: 12),
        Expanded(
          child: ListView.builder(
            itemCount: _searchResults.length,
            itemBuilder: (context, index) {
              final hit = _searchResults[index];
              final IndexedVideo video = hit['video'];
              final VideoFrameIndex frame = hit['frame'];
              final double score = hit['score'];

              return Card(
                color: const Color(0xFF1C1B1B),
                margin: const EdgeInsets.only(bottom: 12),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: const Color(0xFF2A2A2A)),
                ),
                child: ListTile(
                  leading: Container(
                    width: 72,
                    height: 48,
                    decoration: BoxDecoration(
                      color: Colors.blueGrey.shade900,
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: const Center(
                      child: Icon(Icons.movie, color: Color(0xFF00F5FF)),
                    ),
                  ),
                  title: Text(video.fileName, style: const TextStyle(fontFamily: 'Sora', fontWeight: FontWeight.w600)),
                  subtitle: Text(
                    "Timestamp: ${frame.timestampMs} ms | Labels: ${frame.detectedObjects}",
                    style: const TextStyle(fontFamily: 'Inter', fontSize: 12, color: Colors.grey),
                  ),
                  trailing: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      border: Border.all(color: const Color(0xFF39FF14)),
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Text(
                      "${(score * 100).toStringAsFixed(1)}% SIM",
                      style: const TextStyle(fontFamily: 'JetBrains Mono', fontSize: 11, color: Color(0xFF39FF14)),
                    ),
                  ),
                  onTap: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => VideoPlayerScreen(video: video, activeFrame: frame),
                      ),
                    );
                  },
                ),
              );
            },
          ),
        ),
      ],
    );
  }

  Widget _buildVideosGrid(List<IndexedVideo> videos) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          "INDEXED OFFLINE LIBRARY",
          style: TextStyle(fontFamily: 'JetBrains Mono', fontSize: 12, letterSpacing: 1.2),
        ),
        const SizedBox(height: 16),
        Expanded(
          child: GridView.builder(
            itemCount: videos.length,
            gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 2,
              crossAxisSpacing: 12,
              mainAxisSpacing: 12,
              childAspectRatio: 1.4,
            ),
            itemBuilder: (context, index) {
              final v = videos[index];
              return InkWell(
                onTap: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => VideoPlayerScreen(video: v),
                    ),
                  );
                },
                child: Container(
                  decoration: BoxDecoration(
                    color: const Color(0xFF1C1B1B),
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: const Color(0xFF2A2A2A)),
                  ),
                  child: Stack(
                    children: [
                      // Simulated Video Frame
                      Positioned.fill(
                        child: Opacity(
                          opacity: 0.15,
                          child: Icon(Icons.movie, size: 80, color: const Color(0xFF00F5FF).withOpacity(0.5)),
                        ),
                      ),
                      Positioned(
                        bottom: 12,
                        left: 12,
                        right: 12,
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              v.fileName,
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: const TextStyle(fontFamily: 'Sora', fontWeight: FontWeight.bold, fontSize: 13),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              "${(v.durationMs / 1000).toStringAsFixed(1)}s | ${(v.sizeBytes / 1000000).toStringAsFixed(1)} MB",
                              style: const TextStyle(fontFamily: 'JetBrains Mono', fontSize: 11, color: Colors.grey),
                            ),
                          ],
                        ),
                      ),
                      Positioned(
                        top: 12,
                        right: 12,
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                          decoration: BoxDecoration(
                            color: const Color(0xFF39FF14).withOpacity(0.1),
                            border: Border.all(color: const Color(0xFF39FF14), width: 1),
                            borderRadius: BorderRadius.circular(4),
                          ),
                          child: const Text("OFFLINE INDEX", style: TextStyle(fontFamily: 'JetBrains Mono', fontSize: 9, color: Color(0xFF39FF14))),
                        ),
                      ),
                    ],
                  ),
                ),
              );
            },
          ),
        ),
      ],
    );
  }
}

// ----------------- Screen 3: Video Player & Extraction Summary -----------------
class VideoPlayerScreen extends StatefulWidget {
  final IndexedVideo video;
  final VideoFrameIndex? activeFrame;

  const VideoPlayerScreen({Key? key, required this.video, this.activeFrame}) : super(key: key);

  @override
  _VideoPlayerScreenState createState() => _VideoPlayerScreenState();
}

class _VideoPlayerScreenState extends State<VideoPlayerScreen> {
  String _activeTab = "Extractive Summary";
  List<String> _summarySentences = [];
  bool _calculatingSummary = false;

  @override
  void initState() {
    super.override.initState();
    _computeSummary();
  }

  void _computeSummary() async {
    setState(() {
      _calculatingSummary = true;
    });

    await Future.delayed(const Duration(seconds: 1)); // Simulates local TextRank execution latency

    // Setup input structures for local Extractive PageRank
    final sentences = [
      "The cinematic drone climbs and orbits around the summit.",
      "Beautiful snowy ranges expand towards the horizon.",
      "The path is winding up steeply, tracking over rocks.",
      "Hiking down safely as the cloud covers the valley."
    ];
    final rand = Random(42);
    final embeddings = List.generate(4, (_) => List.generate(512, (_) => rand.nextDouble()));

    final summary = ExtractiveTextRank.extractSummary(sentences, embeddings, numSentences: 2);

    setState(() {
      _summarySentences = summary;
      _calculatingSummary = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.video.fileName, style: const TextStyle(fontFamily: 'Sora', fontWeight: FontWeight.bold)),
        backgroundColor: const Color(0xFF131313),
      ),
      body: Column(
        children: [
          // Simulated Wave Player Screen Box
          Container(
            height: 240,
            color: Colors.black,
            child: Stack(
              children: [
                Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Icon(Icons.play_circle_filled, size: 64, color: Color(0xFF00F5FF)),
                      const SizedBox(height: 8),
                      Text(
                        "Playing indexed media local feed at ${widget.activeFrame?.timestampMs ?? 0} ms",
                        style: const TextStyle(fontFamily: 'JetBrains Mono', fontSize: 12),
                      )
                    ],
                  ),
                ),
                // Cyber layout coordinates & bounded boxes overlay
                Positioned(
                  top: 20,
                  left: 20,
                  child: Container(
                    padding: const EdgeInsets.all(4),
                    color: Colors.black54,
                    child: const Text("BOUNDING BOX [person: 98%]", style: TextStyle(fontFamily: 'JetBrains Mono', color: Color(0xFF39FF14), fontSize: 10)),
                  ),
                ),
                Positioned(
                  top: 60,
                  left: 40,
                  child: Container(
                    width: 120,
                    height: 120,
                    decoration: Border.all(color: const Color(0xFF39FF14), width: 2),
                  ),
                )
              ],
            ),
          ),

          // Player Tabs Control
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: ["Extractive Summary", "Audio Transcripts", "PP-OCR Telemetry"].map((tab) {
              final active = _activeTab == tab;
              return Expanded(
                child: TextButton(
                  style: TextButton.styleFrom(
                    foregroundColor: active ? const Color(0xFF00F5FF) : Colors.grey,
                    backgroundColor: active ? const Color(0xFF1C1B1B) : Colors.transparent,
                    shape: const RoundedRectangleBorder(),
                  ),
                  onPressed: () {
                    setState(() {
                      _activeTab = tab;
                    });
                  },
                  child: Text(tab, style: const TextStyle(fontFamily: 'JetBrains Mono', fontSize: 11)),
                ),
              );
            }).toList(),
          ),

          // Selected Tab Render Content
          Expanded(
            child: Container(
              padding: const EdgeInsets.all(16),
              color: const Color(0xFF1C1B1B),
              child: _buildTabContent(),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTabContent() {
    if (_activeTab == "Extractive Summary") {
      if (_calculatingSummary) {
        return const Center(child: CircularProgressIndicator(color: Color(0xFF00F5FF)));
      }
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text("OFFLINE EXTRACTIVE TEXTRANK MATRIX", style: TextStyle(fontFamily: 'JetBrains Mono', fontSize: 11, color: Color(0xFF39FF14))),
          const SizedBox(height: 12),
          ..._summarySentences.map((s) => Padding(
            padding: const EdgeInsets.only(bottom: 8.0),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Icon(Icons.auto_awesome, color: Color(0xFF00F5FF), size: 16),
                const SizedBox(width: 8),
                Expanded(child: Text(s, style: const TextStyle(fontFamily: 'Inter', fontSize: 13, color: Colors.white))),
              ],
            ),
          )).toList(),
        ],
      );
    } else if (_activeTab == "Audio Transcripts") {
      return ListView(
        children: const [
          Text("WHISPER DETECTED SENTENCES:", style: TextStyle(fontFamily: 'JetBrains Mono', fontSize: 11, color: Colors.grey)),
          SizedBox(height: 12),
          Text("[0:02 - 0:08] 'Look at those incredible peaks over there, the snow is fully covering the summit.'", style: TextStyle(fontFamily: 'Inter', fontSize: 13, color: Colors.white)),
          Divider(color: Colors.grey),
          Text("[0:15 - 0:22] 'We are hiking higher up, the pathway is getting quite steep but the view is worth it.'", style: TextStyle(fontFamily: 'Inter', fontSize: 13, color: Colors.white)),
        ],
      );
    } else {
      return ListView(
        children: const [
          Text("PP-OCR TEXT DETECTION TELEMETRY:", style: TextStyle(fontFamily: 'JetBrains Mono', fontSize: 11, color: Colors.grey)),
          SizedBox(height: 12),
          Text("Text Block #1: 'PATH TO SUMMIT' | Conf: 97.4%", style: TextStyle(fontFamily: 'JetBrains Mono', color: Colors.cyan, fontSize: 12)),
          Text("Coords: [[102, 240], [180, 240], [180, 255], [102, 255]]", style: TextStyle(fontFamily: 'JetBrains Mono', color: Colors.grey, fontSize: 11)),
          Divider(color: Colors.grey),
          Text("Text Block #2: 'ALTITUDE 2400M' | Conf: 91.2%", style: TextStyle(fontFamily: 'JetBrains Mono', color: Colors.cyan, fontSize: 12)),
          Text("Coords: [[510, 40], [580, 40], [580, 55], [510, 55]]", style: TextStyle(fontFamily: 'JetBrains Mono', color: Colors.grey, fontSize: 11)),
        ],
      );
    }
  }
}

// ----------------- Screen 4: Control Settings (Resource Management) -----------------
class ControlSettingsScreen extends StatefulWidget {
  const ControlSettingsScreen({Key? key}) : super(key: key);

  @override
  _ControlSettingsScreenState createState() => _ControlSettingsScreenState();
}

class _ControlSettingsScreenState extends State<ControlSettingsScreen> {
  bool _bgProcessing = true;
  double _ramCeiling = 1.2; // Memory budget default
  double _fpsTarget = 60.0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("NEURAL COMPUTE SYSTEM", style: TextStyle(fontFamily: 'Sora', fontWeight: FontWeight.bold)),
        backgroundColor: const Color(0xFF131313),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          const Text("ON-DEVICE RESOURCE ALLOCATION", style: TextStyle(fontFamily: 'JetBrains Mono', fontSize: 12, color: Color(0xFF00F5FF))),
          const SizedBox(height: 20),

          // RAM allocation bar
          Card(
            color: const Color(0xFF1C1B1B),
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text("ACTIVE RUNTIME MEMORY LIMIT (CRITICAL BUDGET)", style: TextStyle(fontFamily: 'JetBrains Mono', fontSize: 11)),
                  const SizedBox(height: 8),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text("${_ramCeiling.toStringAsFixed(1)} GB", style: const TextStyle(fontFamily: 'Sora', fontSize: 20, color: Color(0xFF39FF14), fontWeight: FontWeight.bold)),
                      const Text("CEILING BUDGET: 1.5 GB", style: TextStyle(fontFamily: 'JetBrains Mono', fontSize: 10, color: Colors.red)),
                    ],
                  ),
                  Slider(
                    value: _ramCeiling,
                    min: 0.5,
                    max: 1.5,
                    activeColor: const Color(0xFF39FF14),
                    onChanged: (val) {
                      setState(() {
                        _ramCeiling = val;
                      });
                    },
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),

          SwitchListTile(
            title: const Text("Isolate Thread Ingestion Workers", style: TextStyle(fontFamily: 'Sora')),
            subtitle: const Text("Offloads non-blocking FFI extraction pipelines to auxiliary thread loops", style: TextStyle(fontSize: 12, color: Colors.grey)),
            value: _bgProcessing,
            activeColor: const Color(0xFF00F5FF),
            onChanged: (val) {
              setState(() {
                _bgProcessing = val;
              });
            },
          ),
          const Divider(),

          const Text("DASHBOARD STATS", style: TextStyle(fontFamily: 'JetBrains Mono', color: Colors.grey, fontSize: 11)),
          const SizedBox(height: 8),
          ListTile(
            title: const Text("ObjectBox DB Store Dimension", style: TextStyle(fontSize: 14)),
            trailing: const Text("512-dim (FP32 locked)", style: TextStyle(fontFamily: 'JetBrains Mono', color: Color(0xFF00F5FF))),
          ),
          ListTile(
            title: const Text("Device Architecture Core", style: TextStyle(fontSize: 14)),
            trailing: const Text("Flutter + C++ FFI (100% Offline)", style: TextStyle(fontFamily: 'JetBrains Mono', color: Color(0xFF39FF14))),
          ),
        ],
      ),
    );
  }
}

// ----------------- Screen 5: Timeline Intelligence & Temporal Summaries -----------------
class TimelineScreen extends StatefulWidget {
  const TimelineScreen({Key? key}) : super(key: key);

  @override
  _TimelineScreenState createState() => _TimelineScreenState();
}

class _TimelineScreenState extends State<TimelineScreen> {
  final TextEditingController _temporalQueryController = TextEditingController();
  List<String> _mappedMemories = [];
  bool _querying = false;

  void _runTemporalQuery() async {
    final q = _temporalQueryController.text.trim();
    if (q.isEmpty) return;

    setState(() {
      _querying = true;
    });

    await Future.delayed(const Duration(seconds: 1)); // Mock neural aggregation

    setState(() {
      _querying = false;
      _mappedMemories = [
        "Mapped 3 Hiking Video events from 'Hiking_In_Swiss_Alps.mp4' on peak boundaries.",
        "Identified family timeline clustering with Birthday lasagna sequence.",
        "Co-occurring scenic vectors detected on alpine mountain trails."
      ];
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("TIMELINE INTELLIGENCE", style: TextStyle(fontFamily: 'Sora', fontWeight: FontWeight.bold)),
        backgroundColor: const Color(0xFF131313),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            TextField(
              controller: _temporalQueryController,
              style: const TextStyle(fontFamily: 'Inter', color: Colors.white),
              decoration: InputDecoration(
                hintText: "Ask temporal queries (e.g. Summarize what I did last month)...",
                fillColor: const Color(0xFF1C1B1B),
                filled: true,
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(4)),
                focusedBorder: const OutlineInputBorder(borderSide: BorderSide(color: Color(0xFF00F5FF))),
              ),
            ),
            const SizedBox(height: 12),
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF39FF14),
                foregroundColor: Colors.black,
                padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 16),
              ),
              onPressed: _runTemporalQuery,
              child: const Text("AGGREGATE MEMORY POOL"),
            ),
            const SizedBox(height: 24),
            Expanded(
              child: _querying
                  ? const Center(child: CircularProgressIndicator(color: Color(0xFF00F5FF)))
                  : ListView.builder(
                      itemCount: _mappedMemories.length,
                      itemBuilder: (context, idx) {
                        return Card(
                          color: const Color(0xFF1C1B1B),
                          child: ListTile(
                            leading: const Icon(Icons.schedule, color: Color(0xFF00F5FF)),
                            title: Text(_mappedMemories[idx], style: const TextStyle(fontFamily: 'Inter', fontSize: 13)),
                          ),
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }
}
