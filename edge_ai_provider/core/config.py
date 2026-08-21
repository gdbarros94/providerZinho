"""Application settings loaded from environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """EdgeAI Micro-Provider configuration.

    Values are read from environment variables (case-insensitive) or a `.env`
    file located in the project root.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Server ──────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 9880
    log_level: str = "info"

    # ── Authentication ──────────────────────────────────────────────────────
    api_key: str | None = None

    # ── Model storage ───────────────────────────────────────────────────────
    models_dir: Path = Path.home() / ".edge_ai_models"

    # ── Hardware limits ─────────────────────────────────────────────────────
    max_ram_percent: float = 85.0
    max_cpu_percent: float = 90.0
    max_concurrent_inferences: int = 2

    # ── GPU ──────────────────────────────────────────────────────────────────
    # "auto" → detect & use if available
    # "none" → force CPU only
    # integer string (e.g. "35") → fixed n_gpu_layers value
    gpu_mode: str = "auto"

    # ── Default models to pre-register on startup ───────────────────────────
    # Comma-separated list: "alias:filename,alias2:filename2"
    # Example: "smollm-135m:smollm-135m-instruct-v0.2.Q4_K_M.gguf"
    default_text_models: str = ""

    # ── Needle ──────────────────────────────────────────────────────────────
    needle_enabled: bool = True
    needle_model_id: str = "needle2-edge"

    # ── Derived helpers (not from env) ──────────────────────────────────────

    @field_validator("models_dir", mode="before")
    @classmethod
    def _expand_models_dir(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()

    def parse_default_text_models(self) -> dict[str, str]:
        """Return ``{alias: filename}`` parsed from the comma-separated env var."""
        if not self.default_text_models.strip():
            return {}
        pairs: dict[str, str] = {}
        for entry in self.default_text_models.split(","):
            entry = entry.strip()
            if ":" not in entry:
                continue
            alias, filename = entry.split(":", maxsplit=1)
            pairs[alias.strip()] = filename.strip()
        return pairs

    @property
    def gpu_layers(self) -> int:
        """Resolve gpu_mode into an ``n_gpu_layers`` integer.

        * ``"auto"`` → handled externally by :mod:`utils.gpu_detector`
        * ``"none"`` → ``0``
        * Numeric string → the parsed int
        """
        if self.gpu_mode.lower() == "none":
            return 0
        if self.gpu_mode.lower() == "auto":
            return -1  # sentinel; gpu_detector will resolve
        try:
            return int(self.gpu_mode)
        except ValueError:
            return 0


# ── Singleton accessor ──────────────────────────────────────────────────────

_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached :class:`Settings` singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
