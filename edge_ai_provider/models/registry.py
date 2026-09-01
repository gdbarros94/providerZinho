"""Model Registry — singleton that manages adapter lifecycle and lookup."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from fastapi import HTTPException, status

from edge_ai_provider.api.schemas import ModelInfo
from edge_ai_provider.models.base import BaseModelAdapter

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central registry of available model adapters.

    Adapters are registered by their ``model_id`` and can be looked up
    by the ``model`` field in incoming API requests.
    """

    def __init__(self) -> None:
        self._adapters: dict[str, BaseModelAdapter] = {}
        self._created_at: int = int(time.time())

    # ── Registration ────────────────────────────────────────────────────────

    def register(self, adapter: BaseModelAdapter) -> None:
        """Register a model adapter under its ``model_id``."""
        model_id = adapter.model_id
        if model_id in self._adapters:
            logger.warning("Overwriting existing adapter for model '%s'", model_id)
        self._adapters[model_id] = adapter
        logger.info("Registered model adapter: %s", model_id)

    def unregister(self, model_id: str) -> None:
        """Remove an adapter from the registry."""
        self._adapters.pop(model_id, None)
        logger.info("Unregistered model adapter: %s", model_id)

    # ── Lookup ──────────────────────────────────────────────────────────────

    def get(self, model_id: str) -> BaseModelAdapter:
        """Retrieve an adapter by model ID.

        Raises:
            HTTPException(404): If the model is not registered.
        """
        adapter = self._adapters.get(model_id)
        if adapter is None:
            available = ", ".join(sorted(self._adapters.keys())) or "(none)"
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": (
                            f"Model '{model_id}' not found. "
                            f"Available models: {available}"
                        ),
                        "type": "invalid_request_error",
                        "code": "model_not_found",
                    }
                },
            )
        return adapter

    async def switch_to_model(self, model_id: str) -> BaseModelAdapter:
        """
        Ensures only one model is loaded in RAM at a time.
        Unloads current model before loading the target.
        """
        target = self.get(model_id)

        # Find currently loaded model
        current_loaded = next(
            (a for a in self._adapters.values() if a.is_loaded), None
        )

        if current_loaded and current_loaded.model_id != model_id:
            logger.info("Swapping models: %s -> %s", current_loaded.model_id, model_id)
            await current_loaded.unload()

        if not target.is_loaded:
            await target.load()

        return target

    def has(self, model_id: str) -> bool:
        """Check whether a model is registered."""
        return model_id in self._adapters

    # ── Listing ─────────────────────────────────────────────────────────────

    def list_models(self) -> list[ModelInfo]:
        """Return :class:`ModelInfo` for every registered adapter."""
        return [
            ModelInfo(id=model_id, created=self._created_at)
            for model_id in sorted(self._adapters.keys())
        ]

    @property
    def model_ids(self) -> list[str]:
        """Sorted list of registered model IDs."""
        return sorted(self._adapters.keys())

    # ── Lifecycle ───────────────────────────────────────────────────────────

    async def load_all(self) -> None:
        """Load every registered adapter that is not yet loaded."""
        for model_id, adapter in self._adapters.items():
            if not adapter.is_loaded:
                logger.info("Loading model: %s", model_id)
                await adapter.load()
                logger.info("Model loaded: %s", model_id)

    async def unload_all(self) -> None:
        """Unload every registered adapter."""
        for model_id, adapter in self._adapters.items():
            if adapter.is_loaded:
                logger.info("Unloading model: %s", model_id)
                await adapter.unload()
                logger.info("Model unloaded: %s", model_id)


# ── Singleton ───────────────────────────────────────────────────────────────

_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Return the cached :class:`ModelRegistry` singleton."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
