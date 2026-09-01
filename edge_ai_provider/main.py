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
from edge_ai_provider.core.thermal import get_thermal_provider
from edge_ai_provider.core.router import AgenticRouter
from edge_ai_provider.models.registry import get_registry
from edge_ai_provider.utils.payload_processor import PayloadProcessor

logger = logging.getLogger("edge_ai_provider")


# ════════════════════════════════════════════════════════════════════════════
# Lifespan — startup / shutdown hooks
# ════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: load models on startup, unload on shutdown."""
    settings = get_settings()

    # Init Payload Processor
    processor = PayloadProcessor()
    app.state.payload_processor = processor

    # Init Hardware Monitor
    hw = init_hardware_monitor(settings)
    app.state.hw_monitor = hw

    # Init Agentic Router
    registry = get_registry()
    router = AgenticRouter(registry, hw)
    app.state.router = router

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
    from edge_ai_provider.models.llama_cpp_adapter import LlamaCPPAdapter
    from edge_ai_provider.utils.gpu_detector import detect_gpu_layers

    # ── Needle 2 ────────────────────────────────────────────────────────
    if settings.needle_enabled:
        needle = NeedleAdapter(model_id=settings.needle_model_id)
        registry.register(needle)
        logger.info("Registered Needle adapter as '%s'", settings.needle_model_id)

    # ── Text models from MODELS_DIR ─────────────────────────────────────
    models_dir = settings.models_dir
    if not models_dir.exists():
        logger.warning("Models directory %s not found", models_dir)
        return

    # ── SLM Fleet Registration ──────────────────────────────────────────
    # Mapping based on RFC v2
    slm_configs = [
        {"id": "qwen-0.5b", "file": "qwen2.5-0.5b-instruct-q6_k.gguf", "ctx": 32768, "t": 4},
        {"id": "gemma-2b", "file": "gemma-2-2b-it-q4_k_m.gguf", "ctx": 8192, "t": 4},
        {"id": "llama-3.2-3b", "file": "llama-3.2-3b-instruct-iq3_m.gguf", "ctx": 8192, "t": 6},
        {"id": "phi-3.5-mini", "file": "phi-3.5-mini-instruct-q4_k_m.gguf", "ctx": 8192, "t": 6},
    ]

    for cfg in slm_configs:
        path = models_dir / cfg["file"]
        if path.exists():
            # Calculate GPU layers if available
            ngl = detect_gpu_layers(path) if gpu_available else 0
            adapter = LlamaCPPAdapter(
                model_id=cfg["id"],
                model_path=str(path),
                n_ctx=cfg["ctx"],
                n_threads=cfg["t"],
                n_gpu_layers=ngl,
            )
            registry.register(adapter)
            logger.info(
                "Registered SLM: %s (ctx: %d, t: %d, ngl: %d)",
                cfg["id"],
                cfg["ctx"],
                cfg["t"],
                ngl,
            )
        else:
            logger.warning("SLM file missing: %s", path)


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
