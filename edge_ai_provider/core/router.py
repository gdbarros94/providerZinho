"""Agentic Router for SLM selection.
Routes requests to specific models based on token count, 
intent (tool calling), and thermal state.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from edge_ai_provider.api.schemas import ChatCompletionRequest

if TYPE_CHECKING:
    from edge_ai_provider.models.registry import ModelRegistry
    from edge_ai_provider.core.hardware_monitor import HardwareMonitor

logger = logging.getLogger(__name__)

class AgenticRouter:
    def __init__(self, registry: ModelRegistry, hw_monitor: HardwareMonitor):
        self.registry = registry
        self.hw_monitor = hw_monitor
        
        # Model mapping for different capabilities
        self.capability_map = {
            "fast": "qwen-0.5b",
            "balanced": "gemma-2b",
            "reasoning": "llama-3.2-3b",
            "tools": "phi-3.5-mini"
        }

    async def route_and_activate(self, request: ChatCompletionRequest, token_count: int) -> BaseModelAdapter:
        """
        Determines the best model and ensures it is loaded in RAM.
        """
        model_id = self.route(request, token_count)
        logger.info("Router selected model: %s", model_id)
        
        # Exclusive swap: unload others, load target
        return await self.registry.switch_to_model(model_id)

    def route(self, request: ChatCompletionRequest, token_count: int) -> str:
        """
        Determines the best model for the request.
        """
        # 1. Thermal Check: If critical, force the smallest model
        snap = self.hw_monitor.snapshot()
        if snap.temperature and snap.temperature > 60:
            logger.warning("Thermal Critical: Routing to smallest model")
            return self.capability_map["fast"]

        # 2. Tool Calling Detection
        if self._detect_tool_intent(request):
            return self.capability_map["tools"]

        # 3. Token-based Routing
        if token_count < 1000:
            return self.capability_map["fast"]
        elif token_count < 8000:
            return self.capability_map["balanced"]
        else:
            return self.capability_map["reasoning"]

    def _detect_tool_intent(self, request: ChatCompletionRequest) -> bool:
        """
        Robust scanner for tool-calling intent.
        """
        last_msg = self._get_last_user_message(request)
        if not last_msg:
            return False
            
        # Check for JSON-like structures or tool keywords
        tool_keywords = ["calculate", "search", "api", "function", "tool"]
        if any(kw in last_msg.lower() for kw in tool_keywords):
            return True
            
        if "{" in last_msg and "}" in last_msg:
            return True
            
        return False

    def _get_last_user_message(self, request: ChatCompletionRequest) -> str:
        for msg in reversed(request.messages):
            if msg.role == "user" and msg.content:
                return msg.content
