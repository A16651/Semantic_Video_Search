"""
app/utils/hardware.py
=====================
Hardware Auto-Detection Module (Item 4 of the optimisation plan).

Automatically selects the best ONNX Runtime Execution Provider based on
what is available in the current environment:

  Priority order
  --------------
  1. OpenVINOExecutionProvider  → Intel CPUs / Integrated GPUs (iGPU)
     - Targets the Intel HD Graphics in GPU_FP16 mode when available.
     - Falls back to CPU_FP32 inside OpenVINO if the iGPU is unsupported.
  2. NNAPIExecutionProvider     → Android / Exynos / ARM-based SoCs
  3. CPUExecutionProvider       → Universal fallback (always available)

Usage
-----
    from app.utils.hardware import get_execution_providers, get_inference_session

    # Option A – use in ort.InferenceSession directly
    providers, provider_options = get_execution_providers()
    session = ort.InferenceSession(model_path, providers=providers,
                                   provider_options=provider_options)

    # Option B – convenience wrapper that returns a ready-made session
    session = get_inference_session(model_path)
"""

import logging
import os
from typing import Optional

log = logging.getLogger(__name__)


def _is_openvino_runtime_available() -> bool:
    """
    Check whether the OpenVINO *runtime* is actually loadable on this machine,
    not just whether the ORT bridge DLL exists.

    Problem: ort.get_available_providers() lists OpenVINOExecutionProvider
    whenever the onnxruntime_providers_openvino.dll is present, but that DLL
    itself depends on openvino.dll which may be missing if the user never
    installed the full OpenVINO runtime.  ORT then emits a scary error at
    session-creation time and falls back to CPU anyway — but the error log
    confuses users.

    Fix: try to import openvino.runtime.Core before listing the provider.
    If this import fails for any reason (DLL missing, wrong PATH, etc.)
    we skip the provider entirely and ORT goes straight to CPU with zero noise.
    """
    try:
        from openvino.runtime import Core   # noqa: F401
        _ = Core()                          # forces the DLL chain to load now
        return True
    except Exception:
        return False


def get_execution_providers() -> tuple[list[str], list[dict]]:
    """
    Probe available ONNX Runtime Execution Providers and return the best
    ordered list together with any provider-specific options.

    Returns
    -------
    providers : list[str]
        Ordered list of provider names (highest-priority first).
        ORT will fall through the list until one succeeds.
    provider_options : list[dict]
        Parallel list of option dicts for each provider entry.
        Empty dicts mean "use defaults for this provider".
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except ImportError:
        log.error("onnxruntime is not installed.")
        return ["CPUExecutionProvider"], [{}]

    log.info("Available ORT providers: %s", available)

    providers: list[str] = []
    provider_options: list[dict] = [{}]  # CPU fallback options placeholder

    # ── 1. OpenVINO (Intel CPU / iGPU) ─────────────────────────────────────
    # Only add if both the ORT bridge *and* the openvino.dll runtime are present.
    if "OpenVINOExecutionProvider" in available and _is_openvino_runtime_available():
        log.info(
            "OpenVINOExecutionProvider detected + runtime OK → targeting Intel GPU_FP16."
        )
        providers.append("OpenVINOExecutionProvider")
        provider_options.insert(0, {
            "device_type": os.getenv("OPENVINO_DEVICE", "GPU_FP16"),
            "enable_opencl_throttling": "false",
            "cache_dir": os.getenv("OPENVINO_CACHE_DIR", ""),
        })
    elif "OpenVINOExecutionProvider" in available:
        log.info(
            "OpenVINOExecutionProvider listed by ORT but openvino runtime not "
            "loadable (openvino.dll missing?). Skipping — using CPU instead. "
            "Install the full OpenVINO runtime to enable iGPU acceleration."
        )

    # ── 2. NNAPI (Android / Exynos / ARM SoCs) ─────────────────────────────
    elif "NnapiExecutionProvider" in available:
        log.info("NnapiExecutionProvider detected → using NNAPI for hardware accel.")
        providers.append("NnapiExecutionProvider")
        provider_options.insert(0, {"NNAPI_FLAG_USE_FP16": "1"})

    # ── 3. CPU fallback (always appended last) ──────────────────────────────
    providers.append("CPUExecutionProvider")
    provider_options[-1] = {}   # CPU options: all defaults

    log.info("Selected provider chain: %s", providers)
    return providers, provider_options


def get_inference_session(
    model_path: str,
    intra_op_threads: int = 0,
    inter_op_threads: int = 0,
    enable_memory_arena: bool = True,
) -> "ort.InferenceSession":  # noqa: F821  (type-hint only, not importing at module level)
    """
    Convenience function: build and return a hardware-optimised ORT
    InferenceSession for *model_path*.

    Parameters
    ----------
    model_path          : path to a .onnx file
    intra_op_threads    : threads for op-level parallelism (0 = ORT auto)
    inter_op_threads    : threads for graph-level parallelism (0 = ORT auto)
    enable_memory_arena : reduce allocator overhead on repeated calls

    Returns
    -------
    ort.InferenceSession configured with the best available provider.
    """
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()

    if intra_op_threads > 0:
        sess_opts.intra_op_num_threads = intra_op_threads
    if inter_op_threads > 0:
        sess_opts.inter_op_num_threads = inter_op_threads
    if enable_memory_arena:
        sess_opts.enable_cpu_mem_arena = True

    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers, provider_options = get_execution_providers()

    session = ort.InferenceSession(
        model_path,
        sess_options=sess_opts,
        providers=providers,
        provider_options=provider_options if any(provider_options) else None,
    )

    active = session.get_providers()[0]
    log.info("ORT session created for '%s' using provider: %s", model_path, active)
    return session
