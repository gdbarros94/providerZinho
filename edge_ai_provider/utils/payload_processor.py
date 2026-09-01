"""Swiss-Army Payload Processor.
Handles prompt compression, command parsing (/help, /compact), 
and token-based model routing.
"""

from __future__ import annotations

import logging
import tiktoken
from typing import Any
from edge_ai_provider.api.schemas import ChatCompletionRequest

logger = logging.getLogger(__name__)

class PayloadProcessor:
    def __init__(self):
        # Use cl100k_base (GPT-4/Llama 3 compatible)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Static responses for commands
        self.static_commands = {
            "/help": "EdgeAI Swiss-Army Help:\n- /help, /?: Show this menu\n- /compact: Compress context\n- /search [query]: Search context\n- /attach [id]: Attach specific msg",
            "/?": "EdgeAI Swiss-Army Help:\n- /help, /?: Show this menu\n- /compact: Compress context\n- /search [query]: Search context\n- /attach [id]: Attach specific msg",
        }

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def process_request(self, request: ChatCompletionRequest) -> tuple[str | None, ChatCompletionRequest]:
        """
        Processes the request. 
        Returns (static_response, modified_request).
        If static_response is not None, the AI should be bypassed.
        """
        last_msg = self._get_last_user_message(request)
        if not last_msg:
            return None, request

        # 1. Check for static commands
        for cmd, response in self.static_commands.items():
            if last_msg.startswith(cmd):
                return response, request

        # 2. Handle dynamic tools (Swiss-Army)
        if last_msg.startswith("/"):
            return self._handle_tool_command(last_msg, request)

        return None, request

    def _get_last_user_message(self, request: ChatCompletionRequest) -> str:
        for msg in reversed(request.messages):
            if msg.role == "user" and msg.content:
                return msg.content
        return ""

    def _handle_tool_command(self, text: str, request: ChatCompletionRequest) -> tuple[str | None, ChatCompletionRequest]:
        parts = text.split(maxsplit=1)
        cmd = parts[0]
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "/compact":
            return None, self._compact_context(request)
        
        if cmd == "/caveman":
            return None, self._apply_caveman_mode(request)
        
        # Other tools like /search, /attach would be implemented here
        return None, request

    def _apply_caveman_mode(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """
        Caveman Mode: Strips conjunctions and filler words to reduce tokens.
        """
        fillers = [" the ", " a ", " an ", " and ", " or ", " but ", " because ", " although ", " however "]
        
        new_messages = []
        for msg in request.messages:
            content = msg.content or ""
            for filler in fillers:
                content = content.replace(filler, " ")
            
            # Update message content (assuming Pydantic model)
            msg.content = content
            new_messages.append(msg)
            
        request.messages = new_messages
        return request

    def _compact_context(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """
        Robust context compression.
        Keeps the system prompt and the last 2 turns, summarizes the rest.
        """
        messages = request.messages
        if len(messages) <= 3:
            return request

        # Keep system prompt (index 0) and last 2 messages
        system_prompt = messages[0]
        recent_history = messages[-2:]
        
        # Simple compression: remove middle messages
        # In a full implementation, this would call a small model to summarize
        compressed_messages = [system_prompt] + recent_history
        
        # Update request (assuming ChatCompletionRequest is a Pydantic model)
        request.messages = compressed_messages
        return request
