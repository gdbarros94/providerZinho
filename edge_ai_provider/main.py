"""EdgeAI Micro-Provider — FastAPI application entry point.

Start with::

    uvicorn edge_ai_provider.main:app --host 0.0.0.0 --port 9880

Or simply::

    python -m edge_ai_provider.main
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from edge_ai_provider.api.routes import router
from edge_ai_provider.core.config import Settings, get_settings
from edge_ai_provider.core.hardware_monitor import init_hardware_monitor
from edge_ai_provider.models.registry import get_registry

logger = logging.getLogger("edge_ai_provider")


# ════════════════════════════════════════════════════════════════════════════
# Lifespan — startup / shutdown hooks
# ════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: load models on startup, unload on shutdown."""
    settings = get_settings()

    # ── Configure logging ───────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("═" * 60)
    logger.info("  EdgeAI Micro-Provider v0.1.0")
    logger.info("  Port: %d │ GPU mode: %s", settings.port, settings.gpu_mode)
    logger.info("═" * 60)

    # ── Init hardware monitor ───────────────────────────────────────────
    hw = init_hardware_monitor(settings)
    snap = hw.snapshot()
    logger.info(
        "Hardware: RAM %.0f/%.0f MB (%.1f%%) │ CPU %.1f%% │ GPU %s",
        snap.ram_total_mb - snap.ram_available_mb,
        snap.ram_total_mb,
        snap.ram_percent,
        snap.cpu_percent,
        f"{snap.gpu_memory_free_mb:.0f}/{snap.gpu_memory_total_mb:.0f} MB"
        if snap.gpu_available
        else "not available",
    )

    # ── Register models ─────────────────────────────────────────────────
    registry = get_registry()
    await _register_models(settings, registry, snap.gpu_available)

    # ── Load all registered models ──────────────────────────────────────
    await registry.load_all()
    logger.info("Models ready: %s", ", ".join(registry.model_ids) or "(none)")

    yield

    # ── Shutdown ────────────────────────────────────────────────────────
    logger.info("Shutting down — unloading models…")
    await registry.unload_all()
    logger.info("All models unloaded. Goodbye!")


async def _register_models(
    settings: Settings,
    registry: "ModelRegistry",
    gpu_available: bool,
) -> None:
    """Register all configured model adapters."""
    from edge_ai_provider.models.needle_adapter import NeedleAdapter
    from edge_ai_provider.models.text_adapter import TextModelAdapter
    from edge_ai_provider.utils.gpu_detector import detect_gpu_layers

    # ── Needle 2 ────────────────────────────────────────────────────────
    if settings.needle_enabled:
        needle = NeedleAdapter(model_id=settings.needle_model_id)
        registry.register(needle)
        logger.info("Registered Needle adapter as '%s'", settings.needle_model_id)

    # ── Text models from MODELS_DIR ─────────────────────────────────────
    models_dir = settings.models_dir
    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created models directory: %s", models_dir)

    # Explicitly configured models (alias:filename pairs)
    configured = settings.parse_default_text_models()

    # Auto-discover any .gguf files not already configured
    gguf_files = list(models_dir.glob("*.gguf"))
    for gguf_path in gguf_files:
        # Check if this file is already in the configured map
        stem = gguf_path.stem
        if stem not in configured and gguf_path.name not in configured.values():
            configured[stem] = gguf_path.name

    # Resolve GPU layers
    n_gpu_layers = detect_gpu_layers(settings.gpu_mode)
    if n_gpu_layers != 0:
        logger.info("GPU offloading enabled: n_gpu_layers=%d", n_gpu_layers)
    else:
        logger.info("Running in CPU-only mode")

    # Register each text model
    for alias, filename in configured.items():
        model_path = models_dir / filename
        if not model_path.exists():
            logger.warning(
                "Model file '%s' not found in %s — skipping '%s'",
                filename,
                models_dir,
                alias,
            )
            continue

        # Estimate model size for GPU decision
        model_size_mb = model_path.stat().st_size / (1024 * 1024)

        # Only register models under 500 MB RAM footprint
        # (GGUF compressed size is a reasonable proxy)
        if model_size_mb > 500:
            logger.warning(
                "Model '%s' (%.0f MB) exceeds 500 MB limit — skipping",
                alias,
                model_size_mb,
            )
            continue

        adapter = TextModelAdapter(
            model_id=alias,
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=n_gpu_layers,
        )
        registry.register(adapter)
        logger.info(
            "Registered text model '%s' (%.0f MB, %d GPU layers)",
            alias,
            model_size_mb,
            n_gpu_layers,
        )


# ════════════════════════════════════════════════════════════════════════════
# App factory
# ════════════════════════════════════════════════════════════════════════════


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="EdgeAI Micro-Provider",
        description=(
            "OpenAI-compatible API for edge AI models. "
            "Serves Needle 2 (tool calling) and GGUF text models "
            "with SSE streaming support."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # ── CORS (permissive for local / Tailnet use) ───────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Mount routes ────────────────────────────────────────────────────
    app.include_router(router)

    return app


# Uvicorn import target
app = create_app()


# ── Direct execution ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "edge_ai_provider.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,
    )
