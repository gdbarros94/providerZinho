"""Pydantic v2 models replicating the OpenAI chat completions API spec.

Covers request, non-streaming response, streaming delta chunks,
tool/function definitions, and the ``/v1/models`` listing.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ════════════════════════════════════════════════════════════════════════════
# Tool / Function definitions
# ════════════════════════════════════════════════════════════════════════════


class FunctionDefinition(BaseModel):
    """Schema for a callable function declared by the client."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolDefinition(BaseModel):
    """Wrapper around :class:`FunctionDefinition` — OpenAI always nests under ``type``."""

    type: Literal["function"] = "function"
    function: FunctionDefinition


class FunctionCall(BaseModel):
    """A resolved function call in an assistant message."""

    name: str
    arguments: str  # JSON-encoded string


class ToolCall(BaseModel):
    """A single tool invocation returned by the model."""

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


# ════════════════════════════════════════════════════════════════════════════
# Request
# ════════════════════════════════════════════════════════════════════════════


class ChatMessage(BaseModel):
    """A single message in the conversation history."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    """``POST /v1/chat/completions`` request body."""

    model: str
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = None
    stream: bool = False
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict[str, Any] | None = None
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stop: str | list[str] | None = None
    n: int = Field(default=1, ge=1, le=1)  # we only support n=1


# ════════════════════════════════════════════════════════════════════════════
# Non-streaming response
# ════════════════════════════════════════════════════════════════════════════


class Usage(BaseModel):
    """Token usage counters."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    """A single completion choice (non-streaming)."""

    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "tool_calls", "length"] | None = None


class ChatCompletionResponse(BaseModel):
    """Full (non-streamed) response for ``/v1/chat/completions``."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


# ════════════════════════════════════════════════════════════════════════════
# Streaming (SSE delta chunks)
# ════════════════════════════════════════════════════════════════════════════


class FunctionCallDelta(BaseModel):
    """Partial function call in a streaming delta."""

    name: str | None = None
    arguments: str | None = None


class ToolCallDelta(BaseModel):
    """Partial tool call in a streaming delta."""

    index: int = 0
    id: str | None = None
    type: Literal["function"] | None = None
    function: FunctionCallDelta | None = None


class DeltaContent(BaseModel):
    """The ``delta`` object inside each streaming chunk's choice."""

    role: str | None = None
    content: str | None = None
    tool_calls: list[ToolCallDelta] | None = None


class StreamChoice(BaseModel):
    """A single choice in a streaming chunk."""

    index: int = 0
    delta: DeltaContent
    finish_reason: Literal["stop", "tool_calls", "length"] | None = None


class ChatCompletionChunk(BaseModel):
    """A single SSE ``data:`` payload during streaming."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]


# ════════════════════════════════════════════════════════════════════════════
# /v1/models listing
# ════════════════════════════════════════════════════════════════════════════


class ModelInfo(BaseModel):
    """Description of a single available model."""

    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "edge-local"


class ModelList(BaseModel):
    """Response for ``GET /v1/models``."""

    object: Literal["list"] = "list"
    data: list[ModelInfo]


# ════════════════════════════════════════════════════════════════════════════
# Error response (OpenAI format)
# ════════════════════════════════════════════════════════════════════════════


class ErrorDetail(BaseModel):
    """Inner error body matching OpenAI error envelope."""

    message: str
    type: str
    code: str | None = None
    param: str | None = None


class ErrorResponse(BaseModel):
    """Top-level error response."""

    error: ErrorDetail
