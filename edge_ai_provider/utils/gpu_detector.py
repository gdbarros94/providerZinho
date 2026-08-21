"""Auto-detect GPU availability and resolve ``n_gpu_layers`` for llama.cpp.

The logic:
1. If ``gpu_mode`` is ``"none"`` → 0 layers (CPU only).
2. If ``gpu_mode`` is a numeric string → use that fixed value.
3. If ``gpu_mode`` is ``"auto"`` → probe for an NVIDIA GPU via ``nvidia-smi``
   and set ``n_gpu_layers = -1`` (offload *all* layers) when VRAM is sufficient,
   otherwise fall back to CPU.
"""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)

# Minimum free VRAM (MB) required to consider GPU offloading worthwhile.
_MIN_VRAM_MB = 256


def detect_gpu_layers(gpu_mode: str, model_size_mb: float = 500.0) -> int:
    """Resolve the ``n_gpu_layers`` value to pass to ``llama_cpp.Llama``.

    Args:
        gpu_mode: One of ``"auto"``, ``"none"``, or an integer string.
        model_size_mb: Estimated model size in MB — used in *auto* mode to
            decide whether VRAM is large enough.

    Returns:
        Number of layers to offload to GPU.  ``0`` means CPU-only.
    """
    mode = gpu_mode.strip().lower()

    if mode == "none":
        logger.info("GPU mode 'none' → forcing CPU-only inference")
        return 0

    # Fixed numeric override
    try:
        layers = int(mode)
        logger.info("GPU mode fixed at %d layers", layers)
        return max(layers, 0)
    except ValueError:
        pass

    if mode != "auto":
        logger.warning("Unknown gpu_mode '%s' — falling back to CPU", gpu_mode)
        return 0

    # ── Auto-detect ─────────────────────────────────────────────────────────
    return _auto_detect(model_size_mb)


def _auto_detect(model_size_mb: float) -> int:
    """Probe NVIDIA GPU and decide how many layers to offload."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode != 0:
            logger.info("nvidia-smi returned non-zero — no NVIDIA GPU detected")
            return 0

        line = result.stdout.strip().split("\n")[0]
        parts = line.split(",")
        free_mb = float(parts[0].strip())
        total_mb = float(parts[1].strip())
        gpu_name = parts[2].strip() if len(parts) > 2 else "unknown"

        logger.info(
            "GPU detected: %s (%.0f MB free / %.0f MB total)",
            gpu_name,
            free_mb,
            total_mb,
        )

        if free_mb < _MIN_VRAM_MB:
            logger.info(
                "Insufficient free VRAM (%.0f MB < %d MB) — using CPU",
                free_mb,
                _MIN_VRAM_MB,
            )
            return 0

        # Enough VRAM: offload all layers
        if free_mb >= model_size_mb * 1.2:
            logger.info("VRAM sufficient — offloading ALL layers to GPU (n_gpu_layers=-1)")
            return -1  # -1 = all layers in llama.cpp

        # Partial offload: rough estimate — 1 layer ≈ model_size / 32
        layer_size = model_size_mb / 32
        feasible_layers = int(free_mb / layer_size) if layer_size > 0 else 0
        feasible_layers = max(feasible_layers, 1)
        logger.info(
            "Partial GPU offload: %d layers (estimated %.0f MB per layer)",
            feasible_layers,
            layer_size,
        )
        return feasible_layers

    except FileNotFoundError:
        logger.info("nvidia-smi not found — no NVIDIA GPU available, using CPU")
        return 0
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out — falling back to CPU")
        return 0
    except Exception:
        logger.exception("GPU detection failed — falling back to CPU")
        return 0
