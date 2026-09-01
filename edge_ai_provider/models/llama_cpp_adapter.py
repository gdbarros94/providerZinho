"""Generic Llama.cpp Adapter for SLMs.
Wraps llama-cpp-python to provide a uniform interface for various GGUF models.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator
from llama_cpp import Llama

from edge_ai_provider.models.base import BaseModelAdapter
from edge_ai_provider.api.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
)

logger = logging.getLogger(__name__)

class LlamaCPPAdapter(BaseModelAdapter):
    def __init__(
        self, 
        model_id: str, 
        model_path: str, 
        n_ctx: int = 2048, 
        n_threads: int = 4, 
        n_gpu_layers: int = 0
    ) -> None:
        super().__init__(model_id)
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self._llm: Llama | None = None

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    async def load(self) -> None:
        """Load the GGUF model into memory."""
        if self.is_loaded:
            return

        try:
            logger.info("Loading Llama.cpp model %s from %s", self.model_id, self.model_path)
            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False
            )
            logger.info("Model %s loaded successfully", self.model_id)
        except Exception as e:
            logger.error("Failed to load model %s: %s", self.model_id, str(e))
            raise

    async def unload(self) -> None:
        """Release model resources."""
        if self._llm:
            # llama-cpp-python doesn't have an explicit unload, 
            # but deleting the reference helps GC
            self._llm = None
            logger.info("Model %s unloaded", self.model_id)

    async def generate(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        if not self.is_loaded:
            await self.load()

        formatted_msgs = self.messages_to_llama_format(request)
        
        # Simple non-streaming call
        response = self._llm.create_chat_completion(
            messages=formatted_msgs,
            options={"temperature": 0.7, "top_p": 0.9}
        )
        
        # Map llama-cpp response to our schema
        content = response["choices"][0]["message"]["content"]
        
        # In a real implementation, we'd map the full ChatCompletionResponse schema
        # For brevity, returning a mock-compatible response
        return ChatCompletionResponse(
            id="chatcmpl-llama",
            object="chat.completion",
            created=0,
            model=self.model_id,
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }]
        )

    async def generate_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        if not self.is_loaded:
            await self.load()

        formatted_msgs = self.messages_to_llama_format(request)
        
        stream = self._llm.create_chat_completion(
            messages=formatted_msgs,
            stream=True,
            options={"temperature": 0.7, "top_p": 0.9}
        )

        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            yield ChatCompletionChunk(
                id="chatcmpl-llama",
                object="chat.completion.chunk",
                created=0,
                model=self.model_id,
                choices=[{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": None
                }]
            )
