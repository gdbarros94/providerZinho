"""Text model adapter — llama.cpp inference with real SSE streaming.

Wraps ``llama-cpp-python`` to serve any GGUF model as an OpenAI-compatible
chat completion endpoint with true token-by-token streaming.
"""

from __future__ import annotations

import asyncio
import gc
import logging
from pathlib import Path
from typing import Any, AsyncGenerator

from edge_ai_provider.api.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    Usage,
)
from edge_ai_provider.models.base import BaseModelAdapter
from edge_ai_provider.utils.stream_parser import (
    generate_completion_id,
    make_content_chunk,
    make_finish_chunk,
    unix_timestamp,
)

logger = logging.getLogger(__name__)


class TextModelAdapter(BaseModelAdapter):
    """Adapter for GGUF text models served via ``llama-cpp-python``.

    Supports both synchronous (``generate``) and true streaming
    (``generate_stream``) inference.
    """

    def __init__(
        self,
        model_id: str,
        model_path: str | Path,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ) -> None:
        super().__init__(model_id)
        self._model_path = Path(model_path)
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._verbose = verbose
        self._llm: Any = None  # llama_cpp.Llama instance

    # ── Lifecycle ───────────────────────────────────────────────────────────

    async def load(self) -> None:
        """Load the GGUF model into memory."""
        if self._llm is not None:
            logger.warning("Model %s already loaded — skipping", self._model_id)
            return

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self._model_path}. "
                f"Download a GGUF model to this path or update MODELS_DIR."
            )

        logger.info(
            "Loading model %s from %s (n_ctx=%d, n_gpu_layers=%d)",
            self._model_id,
            self._model_path,
            self._n_ctx,
            self._n_gpu_layers,
        )

        # Run the blocking load in a thread to avoid stalling the event loop
        self._llm = await asyncio.to_thread(self._create_llm)

        logger.info("Model %s loaded successfully", self._model_id)

    def _create_llm(self) -> Any:
        """Instantiate the llama_cpp.Llama model (blocking)."""
        from llama_cpp import Llama

        return Llama(
            model_path=str(self._model_path),
            n_ctx=self._n_ctx,
            n_gpu_layers=self._n_gpu_layers,
            verbose=self._verbose,
            # Use half the available cores to leave headroom
            n_threads=self._get_thread_count(),
        )

    @staticmethod
    def _get_thread_count() -> int:
        """Choose a sensible thread count — half of available cores, minimum 1."""
        import os

        cpus = os.cpu_count() or 2
        return max(1, cpus // 2)

    async def unload(self) -> None:
        """Release the model and reclaim memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            gc.collect()
            logger.info("Model %s unloaded", self._model_id)

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    # ── Inference ───────────────────────────────────────────────────────────

    async def generate(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Run a full (non-streaming) chat completion."""
        self._ensure_loaded()

        completion_id = generate_completion_id()
        created = unix_timestamp()

        messages = self.messages_to_llama_format(request)
        params = self._build_params(request)

        # Run blocking inference in a thread
        raw = await asyncio.to_thread(
            self._llm.create_chat_completion,
            messages=messages,
            **params,
        )

        # Parse llama.cpp response → OpenAI response
        content = raw["choices"][0]["message"].get("content", "")
        finish_reason = raw["choices"][0].get("finish_reason", "stop")

        usage_raw = raw.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
        )

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=self._model_id,
            choices=[
                Choice(
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )

    async def generate_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Stream tokens as they are generated by llama.cpp."""
        self._ensure_loaded()

        completion_id = generate_completion_id()
        created = unix_timestamp()

        messages = self.messages_to_llama_format(request)
        params = self._build_params(request)

        # The llama.cpp stream is a synchronous generator — we wrap it with
        # asyncio.to_thread per-chunk by running the whole iteration in a
        # thread and pushing chunks through a queue.
        queue: asyncio.Queue[ChatCompletionChunk | None] = asyncio.Queue()

        async def _producer() -> None:
            """Run the blocking generator in a thread, feeding the queue."""

            def _iterate() -> None:
                stream = self._llm.create_chat_completion(
                    messages=messages,
                    stream=True,
                    **params,
                )

                first = True
                for raw_chunk in stream:
                    delta = raw_chunk["choices"][0].get("delta", {})
                    content = delta.get("content")
                    finish = raw_chunk["choices"][0].get("finish_reason")

                    if content is not None:
                        chunk = make_content_chunk(
                            completion_id,
                            self._model_id,
                            created,
                            content,
                            role="assistant" if first else None,
                        )
                        first = False
                        queue.put_nowait(chunk)

                    if finish is not None:
                        chunk = make_finish_chunk(
                            completion_id, self._model_id, created, finish
                        )
                        queue.put_nowait(chunk)

                # Signal end of stream
                queue.put_nowait(None)

            await asyncio.to_thread(_iterate)

        # Start producer in background
        producer_task = asyncio.create_task(_producer())

        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            # Ensure producer is cleaned up
            if not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass

    # ── Private helpers ─────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Raise if the model hasn't been loaded yet."""
        if self._llm is None:
            raise RuntimeError(
                f"Model '{self._model_id}' is not loaded. Call load() first."
            )

    @staticmethod
    def _build_params(request: ChatCompletionRequest) -> dict[str, Any]:
        """Map OpenAI request parameters to llama.cpp kwargs."""
        params: dict[str, Any] = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
        }
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens
        if request.stop is not None:
            params["stop"] = request.stop
        return params
