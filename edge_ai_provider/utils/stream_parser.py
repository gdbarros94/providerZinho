"""Utilities for formatting Server-Sent Events (SSE) and OpenAI delta chunks."""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncGenerator

from edge_ai_provider.api.schemas import (
    ChatCompletionChunk,
    DeltaContent,
    FunctionCallDelta,
    StreamChoice,
    ToolCallDelta,
)


# ── ID / timestamp helpers ──────────────────────────────────────────────────


def generate_completion_id() -> str:
    """Generate a unique completion ID in the OpenAI format."""
    return f"chatcmpl-{uuid.uuid4().hex[:29]}"


def unix_timestamp() -> int:
    """Return the current Unix timestamp as an integer."""
    return int(time.time())


# ── SSE formatting ──────────────────────────────────────────────────────────


def format_sse_event(data: str) -> str:
    """Wrap a JSON string in the SSE ``data:`` envelope."""
    return f"data: {data}\n\n"


def format_sse_chunk(chunk: ChatCompletionChunk) -> str:
    """Serialise a :class:`ChatCompletionChunk` into an SSE line."""
    return format_sse_event(chunk.model_dump_json(exclude_none=True))


def format_sse_done() -> str:
    """Return the SSE terminator for OpenAI streaming."""
    return "data: [DONE]\n\n"


# ── Chunk builders ──────────────────────────────────────────────────────────


def make_content_chunk(
    completion_id: str,
    model: str,
    created: int,
    content: str,
    *,
    role: str | None = None,
    finish_reason: str | None = None,
) -> ChatCompletionChunk:
    """Build a streaming chunk carrying text content."""
    delta = DeltaContent(role=role, content=content)
    choice = StreamChoice(index=0, delta=delta, finish_reason=finish_reason)
    return ChatCompletionChunk(
        id=completion_id,
        model=model,
        created=created,
        choices=[choice],
    )


def make_finish_chunk(
    completion_id: str,
    model: str,
    created: int,
    finish_reason: str = "stop",
) -> ChatCompletionChunk:
    """Build the final chunk that signals generation is complete."""
    delta = DeltaContent()
    choice = StreamChoice(index=0, delta=delta, finish_reason=finish_reason)
    return ChatCompletionChunk(
        id=completion_id,
        model=model,
        created=created,
        choices=[choice],
    )


# ── Tool-call simulated streaming ───────────────────────────────────────────


async def simulate_tool_call_stream(
    completion_id: str,
    model: str,
    created: int,
    tool_call_id: str,
    function_name: str,
    arguments_json: str,
    *,
    chunk_size: int = 20,
) -> AsyncGenerator[ChatCompletionChunk, None]:
    """Simulate streaming for a tool call response from Needle.

    Needle responds synchronously, but when the client requests ``stream=True``
    we need to fragment the response into OpenAI-compatible delta chunks:

    1. First chunk: ``role=assistant`` + tool_calls[0] with ``function.name``
    2. Middle chunks: ``tool_calls[0].function.arguments`` fragmented
    3. Final chunk: ``finish_reason="tool_calls"``
    """
    # Chunk 1: role + function name
    yield ChatCompletionChunk(
        id=completion_id,
        model=model,
        created=created,
        choices=[
            StreamChoice(
                index=0,
                delta=DeltaContent(
                    role="assistant",
                    tool_calls=[
                        ToolCallDelta(
                            index=0,
                            id=tool_call_id,
                            type="function",
                            function=FunctionCallDelta(name=function_name, arguments=""),
                        )
                    ],
                ),
            )
        ],
    )

    # Chunks 2..N: arguments fragmented
    for i in range(0, len(arguments_json), chunk_size):
        fragment = arguments_json[i : i + chunk_size]
        yield ChatCompletionChunk(
            id=completion_id,
            model=model,
            created=created,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaContent(
                        tool_calls=[
                            ToolCallDelta(
                                index=0,
                                function=FunctionCallDelta(arguments=fragment),
                            )
                        ],
                    ),
                )
            ],
        )

    # Final chunk: finish reason
    yield make_finish_chunk(completion_id, model, created, finish_reason="tool_calls")
