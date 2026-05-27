"""
app/engine/models.py
====================
Thread-safe singleton that loads the INT8 ONNX SigLIP model once and keeps it
in memory for the lifetime of the process.

Key design decisions
--------------------
* Vision encoder  → run via ORTModelForFeatureExtraction (image_embeds output).
* Text encoder    → run via ORTModelForFeatureExtraction with separate text inputs.
* Both use the *same* ONNX file exported by tools/quantize_model.py (the full
  SiglipModel export includes both vision and text heads).
* ORT session options (thread counts, memory arena) are read from config so they
  can be tuned per-device without code changes.
* Execution providers are auto-detected via utils/hardware.py (OpenVINO → NNAPI → CPU).
"""

from threading import Lock

import numpy as np
import onnxruntime as ort
from transformers import SiglipProcessor

from app.core.config import settings
from app.utils.hardware import get_execution_providers


class SigLipEngine:
    """
    Singleton ONNX-based SigLIP inference engine.

    Public API
    ----------
    get_components()        → (ort_session, processor, device_str)
    get_image_features(pixel_values_np)  → np.ndarray  [N, 768] float32 L2-normalised
    get_text_features(input_ids_np, attention_mask_np) → np.ndarray [N, 768] float32 L2-normalised
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._load_model()
        return cls._instance

    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        import os

        model_path = os.path.join(settings.ONNX_MODEL_DIR, settings.ONNX_MODEL_FILE)
        print(f"Loading INT8 SigLIP ONNX model from: {model_path}")

        # ── Session options ────────────────────────────────────────────
        sess_opts = ort.SessionOptions()

        # Thread counts (0 → ORT auto-detects optimal value)
        if settings.ORT_INTRA_OP_THREADS > 0:
            sess_opts.intra_op_num_threads = settings.ORT_INTRA_OP_THREADS
        if settings.ORT_INTER_OP_THREADS > 0:
            sess_opts.inter_op_num_threads = settings.ORT_INTER_OP_THREADS

        # Memory arena reduces repeated allocation overhead
        if settings.ORT_ENABLE_MEMORY_ARENA:
            sess_opts.enable_cpu_mem_arena = True

        # Graph optimisation: apply all ORT-level optimisations at load time
        # (safe for both FP32 and INT8 ONNX files)
        sess_opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # ── Load ONNX session ──────────────────────────────────────────
        # Auto-detect best execution provider (OpenVINO > NNAPI > CPU)
        providers, provider_options = get_execution_providers()
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=providers,
            provider_options=provider_options if provider_options else None,
        )

        # Collect the names of the model's inputs and outputs once so we
        # can route image / text tensors without string literals everywhere.
        self._input_names  = {inp.name for inp in self.session.get_inputs()}
        self._output_names = [out.name for out in self.session.get_outputs()]

        # ── Processor (tokeniser + image processor) ────────────────────
        self.processor = SiglipProcessor.from_pretrained(settings.MODEL_NAME)

        self.device = settings.DEVICE
        active_provider = self.session.get_providers()[0]
        print(f"INT8 ONNX model loaded via [{active_provider}]. Outputs: {self._output_names}")

    # ------------------------------------------------------------------
    # Compatibility accessor used by InferenceConsumer / routes.py
    # ------------------------------------------------------------------
    def get_components(self):
        """Return (session, processor, device_str) for callers that need them."""
        return self.session, self.processor, self.device

    # ------------------------------------------------------------------
    # Image embedding
    # ------------------------------------------------------------------
    def get_image_features(self, pixel_values) -> np.ndarray:
        """
        Run vision-encoder inference.

        Parameters
        ----------
        pixel_values : np.ndarray or torch.Tensor  [N, 3, H, W]  float32
            Pre-processed pixel values from SiglipProcessor.

        Returns
        -------
        np.ndarray  [N, 768]  float32, L2-normalised
        """
        # Accept both numpy arrays and PyTorch tensors
        if hasattr(pixel_values, "cpu"):
            pixel_values = pixel_values.cpu().numpy()
        pixel_values = pixel_values.astype(np.float32)

        # Build feed dict with only the inputs this model accepts
        feed = {"pixel_values": pixel_values}
        if "input_ids" in self._input_names:
            # SigLIP uses max_position_embeddings=64 — dummy must match exactly
            feed["input_ids"] = np.zeros((pixel_values.shape[0], 64), dtype=np.int64)

        if "image_embeds" in self._output_names:
            # Named output path – model has an explicit 'image_embeds' head
            outputs = self.session.run(["image_embeds"], feed)
            embeddings = outputs[0]  # [N, 768]
        else:
            # Generic path: run all outputs and pick the pooled one
            outputs = self.session.run(None, feed)
            # ORTModelForFeatureExtraction for SiglipModel exports:
            #   output[0] = last_hidden_state  [N, seq_len, hidden]
            #   output[1] = image_embeds / pooler_output  [N, hidden]  ← what we want
            if len(outputs) >= 2:
                embeddings = outputs[1]          # pooler_output / image_embeds [N, 768]
            else:
                # Fallback: mean-pool the token sequence (index 0)
                embeddings = outputs[0].mean(axis=1)

        return self._l2_normalise(embeddings)

    # ------------------------------------------------------------------
    # Text embedding
    # ------------------------------------------------------------------
    def get_text_features(self, input_ids, attention_mask=None) -> np.ndarray:
        """
        Run text-encoder inference via the same ONNX session.

        Parameters
        ----------
        input_ids      : np.ndarray  [N, seq_len]  int64
        attention_mask : np.ndarray  [N, seq_len]  int64  (optional)

        Returns
        -------
        np.ndarray  [N, 768]  float32, L2-normalised
        """
        # Accept torch tensors
        if hasattr(input_ids, "cpu"):
            input_ids = input_ids.cpu().numpy()
        input_ids = input_ids.astype(np.int64)

        feed = {"input_ids": input_ids}

        if "attention_mask" in self._input_names:
            if attention_mask is not None:
                if hasattr(attention_mask, "cpu"):
                    attention_mask = attention_mask.cpu().numpy()
                feed["attention_mask"] = attention_mask.astype(np.int64)
            else:
                # If model expects a mask but none provided, use all-ones (no masking)
                feed["attention_mask"] = np.ones(input_ids.shape, dtype=np.int64)

        if "pixel_values" in self._input_names:
            feed["pixel_values"] = np.zeros((input_ids.shape[0], 3, 224, 224), dtype=np.float32)

        if "text_embeds" in self._output_names:
            # Named output path – model has an explicit 'text_embeds' head
            outputs = self.session.run(["text_embeds"], feed)
            embeddings = outputs[0]  # [N, 768]
        else:
            # Generic path: run all outputs and pick the pooled one
            outputs = self.session.run(None, feed)
            # For text-only inference the session returns:
            #   output[0] = last_hidden_state  [N, seq_len, hidden]
            #   output[1] = text_embeds / pooler_output  [N, hidden]
            if len(outputs) >= 2:
                embeddings = outputs[1]
            else:
                embeddings = outputs[0].mean(axis=1)

        return self._l2_normalise(embeddings)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _l2_normalise(embeddings: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)   # avoid div-by-zero
        return (embeddings / norm).astype(np.float32)
