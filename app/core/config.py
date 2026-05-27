import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Absolute path to the project root (the folder that contains app/, models/, etc.)
# Works regardless of which directory the server is started from.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    PROJECT_NAME: str = "Semantic Video Search"
    API_V1_STR: str = "/api/v1"

    # ── Qdrant connection ──────────────────────────────────────────────────
    # QDRANT_USE_SERVER=false (default) → local embedded DB, no Docker needed.
    # QDRANT_USE_SERVER=true            → HTTP to QDRANT_HOST:QDRANT_PORT.
    QDRANT_USE_SERVER: bool = False
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "video_embeddings")

    # Local embedded DB path (used when QDRANT_USE_SERVER=false)
    QDRANT_LOCAL_PATH: str = str(_PROJECT_ROOT / "qdrant_local_db")

    # ── Model identity ────────────────────────────────────────────────────
    # HuggingFace model ID used for the processor (tokenizer + image processor).
    # The weights themselves come from the local ONNX file below.
    MODEL_NAME: str = "google/siglip-base-patch16-224"

    # ── ONNX model paths ──────────────────────────────────────────────────
    # Resolved to absolute paths so the app works from any working directory.
    ONNX_MODEL_DIR: str = str(_PROJECT_ROOT / "models" / "siglip_int8")
    # Set to "model_quantized.onnx" to use INT8; "model.onnx" for FP32.
    ONNX_MODEL_FILE: str = "model_quantized.onnx"

    # ── ONNX Runtime performance flags ───────────────────────────────────
    # Number of threads for intra-op parallelism (0 = ORT auto-detect).
    # On mobile / mid-range CPUs, 2-4 is usually optimal.  Keep 0 for auto.
    ORT_INTRA_OP_THREADS: int = 0
    # Number of threads for inter-op parallelism (0 = ORT auto-detect).
    ORT_INTER_OP_THREADS: int = 0
    # Enable ORT memory arena (reduces allocator overhead on repeated calls).
    ORT_ENABLE_MEMORY_ARENA: bool = True

    # ── Inference batching ────────────────────────────────────────────────
    # Frames per inference batch.  Lower = less RAM, higher = more throughput.
    # 8 is a good default for mid-range CPUs; increase if RAM allows.
    INFERENCE_BATCH_SIZE: int = 8

    # ── Frame / scene detection ───────────────────────────────────────────
    # FFmpeg scene-change score threshold (0.0 – 1.0 scale).
    # 0.3 is a good default: catches real cuts, ignores minor lighting shifts.
    # Lower = more frames extracted (more sensitive).
    FFMPEG_SCENE_SCORE: float = 0.3
    # FPS used for uniform-sampling fallback when scene detection produces 0 frames.
    FALLBACK_FPS: float = 1.0
    # LEGACY: kept for backward-compat; no longer used by pipeline.py.
    SCENE_THRESHOLD: float = 27.0

    # ── Legacy CUDA flag (kept for compatibility; ONNX always runs on CPU) ─
    USE_CUDA: bool = False

    @property
    def DEVICE(self) -> str:
        """Always returns 'cpu' – ONNX Runtime handles its own execution providers."""
        return "cpu"

    class Config:
        env_file = ".env"


settings = Settings()
