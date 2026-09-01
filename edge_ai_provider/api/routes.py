"""API routes — OpenAI-compatible endpoints.

* ``POST /v1/chat/completions`` — chat completion (streaming & non-streaming)
* ``GET  /v1/models``           — list available models
* ``GET  /v1/health``           — server & hardware health check
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from edge_ai_provider.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelList,
)
from edge_ai_provider.core.hardware_monitor import HardwareMonitor, get_hardware_monitor
from edge_ai_provider.core.security import verify_api_key
from edge_ai_provider.models.registry import ModelRegistry, get_registry
from edge_ai_provider.utils.stream_parser import format_sse_chunk, format_sse_done

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")


# ════════════════════════════════════════════════════════════════════════════
# POST /v1/chat/completions
# ════════════════════════════════════════════════════════════════════════════


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    app_state: Request,
    _api_key: str | None = Depends(verify_api_key),
    registry: ModelRegistry = Depends(get_registry),
    hw: HardwareMonitor = Depends(get_hardware_monitor),
) -> Any:
    """Handle chat completion requests — supports both streaming and non-streaming."""

    # 1. Check hardware capacity
    ok, reason = await hw.acquire()
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "message": f"Server overloaded: {reason}",
                    "type": "server_error",
                    "code": "capacity_exceeded",
                }
            },
        )

    try:
        # 2. Pre-process payload (Commands, Compression, Token Count)
        processor = app_state.app.state.payload_processor
        static_resp, modified_request = processor.process_request(request)
        
        if static_resp:
            return JSONResponse(content={
                "choices": [{"message": {"role": "assistant", "content": static_resp}}]
            })

        token_count = processor.count_tokens(
            "".join([m.content or "" for m in modified_request.messages])
        )

        # 3. Route and Activate (Exclusive Swap)
        router = app_state.app.state.router
        adapter = await router.route_and_activate(modified_request, token_count)

        # 4. Branch on streaming
        if modified_request.stream:
            return await _handle_streaming(modified_request, adapter, hw)
        else:
            return await _handle_non_streaming(modified_request, adapter)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in chat_completions")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Internal server error: {exc}",
                    "type": "server_error",
                    "code": "internal_error",
                }
            },
        ) from exc
    finally:
        if ok:
            hw.release()


async def _handle_non_streaming(
    request: ChatCompletionRequest,
    adapter: Any,
) -> ChatCompletionResponse:
    """Execute non-streaming inference and return the full response."""
    logger.info(
        "Non-streaming request: model=%s, messages=%d",
        request.model,
        len(request.messages),
    )
    response = await adapter.generate(request)
    return response


async def _handle_streaming(
    request: ChatCompletionRequest,
    adapter: Any,
    hw: HardwareMonitor,
) -> EventSourceResponse:
    """Execute streaming inference and return an SSE response.

    Note: the hardware slot is released by the ``finally`` block in the
    calling function, which runs after the response is fully sent.
    """
    logger.info(
        "Streaming request: model=%s, messages=%d",
        request.model,
        len(request.messages),
    )

    async def _event_generator():
        try:
            async for chunk in adapter.generate_stream(request):
                yield format_sse_chunk(chunk)
            yield format_sse_done()
        except Exception:
            logger.exception("Error during streaming generation")
            yield format_sse_done()

    return EventSourceResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


# ════════════════════════════════════════════════════════════════════════════
# GET /v1/models
# ════════════════════════════════════════════════════════════════════════════


@router.get("/models")
async def list_models(
    _api_key: str | None = Depends(verify_api_key),
    registry: ModelRegistry = Depends(get_registry),
) -> ModelList:
    """Return the list of available models."""
    return ModelList(data=registry.list_models())


# ════════════════════════════════════════════════════════════════════════════
# GET /v1/health
# ════════════════════════════════════════════════════════════════════════════


@router.get("/health")
async def health_check(
    registry: ModelRegistry = Depends(get_registry),
    hw: HardwareMonitor = Depends(get_hardware_monitor),
) -> dict:
    """Return server health including hardware status and loaded models.

    This endpoint does **not** require an API key so monitoring tools can
    ping it freely.
    """
    snap = hw.snapshot()
    return {
        "status": "healthy",
        "hardware": snap.to_dict(),
        "active_inferences": hw.active_inferences,
        "models": registry.model_ids,
    }
