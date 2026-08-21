"""API Key validation — lightweight auth suitable for Tailnet / local use."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from edge_ai_provider.core.config import Settings, get_settings

# Optional bearer scheme — auto_error=False so we handle missing tokens ourselves
_bearer_scheme = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> str | None:
    """Validate the ``Authorization: Bearer <token>`` header.

    * If ``settings.api_key`` is **not set** (``None`` or empty), all requests
      are accepted — useful during development.
    * If ``settings.api_key`` **is** set, the bearer token must match exactly.

    Returns the validated token (or ``None`` when auth is disabled).
    """
    expected = settings.api_key

    # Auth disabled — passthrough
    if not expected:
        return None

    if credentials is None or credentials.credentials != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Invalid API key. Provide a valid key via 'Authorization: Bearer <key>'.",
                    "type": "invalid_api_key",
                    "code": "invalid_api_key",
                }
            },
        )

    return credentials.credentials
