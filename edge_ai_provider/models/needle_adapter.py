"""Needle 2 adapter — structured tool calling on the edge.

Converts OpenAI-format ``tools`` to raw JSON schema dicts accepted by
``cactus-needle``, executes the agent, and returns the result as
``tool_calls`` in OpenAI response format.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncGenerator

from edge_ai_provider.api.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    FunctionCall,
    ToolCall,
    ToolDefinition,
    Usage,
)
from edge_ai_provider.models.base import BaseModelAdapter
from edge_ai_provider.utils.stream_parser import (
    generate_completion_id,
    simulate_tool_call_stream,
    unix_timestamp,
)

logger = logging.getLogger(__name__)


class NeedleAdapter(BaseModelAdapter):
    """Adapter for the **Needle 2** tool-calling model via ``cactus-needle``.

    Needle is instantiated per-request with the tools declared in the payload,
    because each request may define a different set of functions.
    """

    def __init__(self, model_id: str = "needle2-edge") -> None:
        super().__init__(model_id)
        self._loaded = False

    # ── Lifecycle ───────────────────────────────────────────────────────────

    async def load(self) -> None:
        """Verify that ``cactus-needle`` is importable.

        Needle itself is lightweight enough (~28 MB) that we create the
        ``Needle`` instance per-request with the tools from the payload,
        rather than keeping a persistent agent in memory.
        """
        try:
            import needle  # noqa: F401

            self._loaded = True
            logger.info("Needle 2 engine available and ready")
        except ImportError as exc:
            logger.error("cactus-needle not installed: %s", exc)
            raise RuntimeError(
                "cactus-needle package is required for the Needle adapter. "
                "Install with: pip install cactus-needle"
            ) from exc

    async def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Inference ───────────────────────────────────────────────────────────

    async def generate(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Run Needle inference and return a full tool-call response."""
        import needle

        completion_id = generate_completion_id()
        created = unix_timestamp()

        # Extract user prompt
        prompt = self.extract_last_user_message(request)

        # Convert OpenAI tools → Needle schema dicts
        needle_tools = self._convert_tools(request.tools or [])

        if not needle_tools:
            # No tools — return a simple text message
            return self._make_text_response(
                completion_id, created, "No tools provided in the request."
            )

        # Create Needle agent with per-request tools
        agent = needle.Needle(tools=needle_tools)

        # Run inference
        try:
            result = agent.run(prompt)
        except Exception as exc:
            logger.exception("Needle inference failed")
            return self._make_text_response(
                completion_id, created, f"Needle inference error: {exc}"
            )

        # Parse Needle response into OpenAI tool_calls
        return self._build_tool_call_response(
            completion_id, created, result, needle_tools
        )

    async def generate_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Simulate streaming for a Needle tool-call response.

        Needle is fast enough (~ms latency) that true token streaming is
        unnecessary.  We execute synchronously and then fragment the response
        into OpenAI-compatible SSE chunks.
        """
        # Get the full response first
        response = await self.generate(request)

        completion_id = response.id
        created = response.created

        choice = response.choices[0]
        message = choice.message

        if message.tool_calls:
            # Stream tool calls
            for tc in message.tool_calls:
                async for chunk in simulate_tool_call_stream(
                    completion_id=completion_id,
                    model=self._model_id,
                    created=created,
                    tool_call_id=tc.id,
                    function_name=tc.function.name,
                    arguments_json=tc.function.arguments,
                ):
                    yield chunk
        else:
            # Simple text — stream the content
            from edge_ai_provider.utils.stream_parser import (
                make_content_chunk,
                make_finish_chunk,
            )

            content = message.content or ""

            # First chunk with role
            yield make_content_chunk(
                completion_id, self._model_id, created, "", role="assistant"
            )

            # Content chunks
            for i in range(0, len(content), 20):
                yield make_content_chunk(
                    completion_id, self._model_id, created, content[i : i + 20]
                )

            yield make_finish_chunk(completion_id, self._model_id, created)

    # ── Private helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _convert_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert OpenAI tool definitions to raw JSON schema dicts for Needle.

        OpenAI format::

            {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

        Needle format::

            {"name": ..., "description": ..., "parameters": ...}
        """
        return [
            {
                "name": t.function.name,
                "description": t.function.description or "",
                "parameters": t.function.parameters
                or {"type": "object", "properties": {}},
            }
            for t in tools
        ]

    def _build_tool_call_response(
        self,
        completion_id: str,
        created: int,
        result: Any,
        needle_tools: list[dict[str, Any]],
    ) -> ChatCompletionResponse:
        """Map a Needle result into an OpenAI ``ChatCompletionResponse`` with ``tool_calls``."""

        # Needle typically returns an object with .name and .arguments,
        # or a dict with the same structure.  We handle both.
        tool_calls: list[ToolCall] = []

        try:
            calls = self._parse_needle_result(result, needle_tools)
            for name, arguments in calls:
                tool_calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:24]}",
                        function=FunctionCall(
                            name=name,
                            arguments=(
                                json.dumps(arguments)
                                if isinstance(arguments, dict)
                                else str(arguments)
                            ),
                        ),
                    )
                )
        except Exception:
            logger.exception("Failed to parse Needle result: %r", result)
            return self._make_text_response(
                completion_id, created, f"Failed to parse Needle output: {result}"
            )

        if not tool_calls:
            return self._make_text_response(
                completion_id, created, str(result)
            )

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=self._model_id,
            choices=[
                Choice(
                    message=ChatMessage(
                        role="assistant",
                        tool_calls=tool_calls,
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=Usage(),
        )

    @staticmethod
    def _parse_needle_result(
        result: Any, needle_tools: list[dict[str, Any]]
    ) -> list[tuple[str, Any]]:
        """Extract ``(function_name, arguments)`` pairs from a Needle result.

        Handles multiple possible return formats from the ``cactus-needle``
        library:
        - Object with ``.name`` / ``.arguments`` attrs
        - Dict with ``"name"`` / ``"arguments"`` keys
        - List of the above
        - Raw dict that matches a single tool (name inferred)
        """
        calls: list[tuple[str, Any]] = []

        # Normalise to list
        items = result if isinstance(result, list) else [result]

        for item in items:
            # Object with attributes
            if hasattr(item, "name") and hasattr(item, "arguments"):
                calls.append((item.name, item.arguments))
            elif hasattr(item, "name") and hasattr(item, "args"):
                calls.append((item.name, item.args))
            # Dict with name/arguments
            elif isinstance(item, dict) and "name" in item:
                calls.append((item["name"], item.get("arguments", item.get("args", {}))))
            # Raw dict — try to match against known tools
            elif isinstance(item, dict) and len(needle_tools) == 1:
                calls.append((needle_tools[0]["name"], item))
            # Fallback: stringify
            elif isinstance(item, dict):
                # Multiple tools and raw dict — wrap as first tool's args
                calls.append((needle_tools[0]["name"], item))

        return calls

    def _make_text_response(
        self, completion_id: str, created: int, content: str
    ) -> ChatCompletionResponse:
        """Build a simple text completion response (no tool calls)."""
        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=self._model_id,
            choices=[
                Choice(
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=Usage(),
        )
