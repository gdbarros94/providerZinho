"""Abstract base class that every model adapter must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator

from edge_ai_provider.api.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
)


class BaseModelAdapter(ABC):
    """Contract for model adapters served by the :class:`ModelRegistry`.

    Each adapter wraps a specific inference backend (Needle, llama.cpp, etc.)
    and exposes a uniform interface for both synchronous and streaming
    generation.
    """

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        """The model identifier used in API requests (e.g. ``"needle2-edge"``)."""
        return self._model_id

    # ── Lifecycle ───────────────────────────────────────────────────────────

    @abstractmethod
    async def load(self) -> None:
        """Load the model weights / engine into memory."""

    @abstractmethod
    async def unload(self) -> None:
        """Release all resources held by this adapter."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is ready to serve requests."""

    # ── Inference ───────────────────────────────────────────────────────────

    @abstractmethod
    async def generate(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Run inference and return the full completion response."""

    @abstractmethod
    async def generate_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Run inference and yield streaming delta chunks."""

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def extract_last_user_message(request: ChatCompletionRequest) -> str:
        """Extract the last user message content from the request."""
        for msg in reversed(request.messages):
            if msg.role == "user" and msg.content:
                return msg.content
        return ""

    @staticmethod
    def messages_to_llama_format(
        request: ChatCompletionRequest,
    ) -> list[dict[str, str]]:
        """Convert request messages to the dict format expected by llama.cpp."""
        return [
            {"role": m.role, "content": m.content or ""}
            for m in request.messages
        ]
