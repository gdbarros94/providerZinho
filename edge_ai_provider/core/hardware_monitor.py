"""Real-time hardware monitor that gates inference requests.

The monitor checks RAM, CPU, and (optionally) GPU utilisation before allowing
a new inference to proceed.  When the device is under heavy load the request
is rejected with HTTP 503 so the client can retry later — preventing OOM
crashes and system freezes on constrained hardware.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from edge_ai_provider.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class HardwareStatus:
    """Snapshot of the device's current resource utilisation."""

    ram_percent: float
    ram_available_mb: float
    ram_total_mb: float
    cpu_percent: float
    gpu_available: bool
    gpu_memory_free_mb: float | None
    gpu_memory_total_mb: float | None
    timestamp: float = field(default_factory=time.time)

    @property
    def ram_ok(self) -> bool:
        return True  # checked externally with threshold

    def to_dict(self) -> dict:
        return {
            "ram_percent": round(self.ram_percent, 1),
            "ram_available_mb": round(self.ram_available_mb, 1),
            "ram_total_mb": round(self.ram_total_mb, 1),
            "cpu_percent": round(self.cpu_percent, 1),
            "gpu_available": self.gpu_available,
            "gpu_memory_free_mb": (
                round(self.gpu_memory_free_mb, 1) if self.gpu_memory_free_mb is not None else None
            ),
            "gpu_memory_total_mb": (
                round(self.gpu_memory_total_mb, 1) if self.gpu_memory_total_mb is not None else None
            ),
        }


class HardwareMonitor:
    """Monitors device resources and controls inference admission.

    Uses an :class:`asyncio.Semaphore` for hard concurrency limits **plus**
    live RAM/CPU checks so the system won't accept work it can't handle.
    """

    def __init__(self, settings: Settings) -> None:
        self._max_ram = settings.max_ram_percent
        self._max_cpu = settings.max_cpu_percent
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_inferences)
        self._active_inferences: int = 0

        # Warm up psutil CPU counter (first call always returns 0.0)
        psutil.cpu_percent(interval=None)

    # ── Public API ──────────────────────────────────────────────────────────

    def snapshot(self) -> HardwareStatus:
        """Take a point-in-time reading of device resources."""
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)  # non-blocking after warm-up

        gpu_avail, gpu_free, gpu_total = self._probe_gpu()

        return HardwareStatus(
            ram_percent=mem.percent,
            ram_available_mb=mem.available / (1024 * 1024),
            ram_total_mb=mem.total / (1024 * 1024),
            cpu_percent=cpu,
            gpu_available=gpu_avail,
            gpu_memory_free_mb=gpu_free,
            gpu_memory_total_mb=gpu_total,
        )

    def check_capacity(self) -> tuple[bool, str]:
        """Check whether the device can accept another inference.

        Returns:
            ``(True, "ok")`` when capacity is available, or
            ``(False, reason)`` when the request should be rejected.
        """
        snap = self.snapshot()

        if snap.ram_percent > self._max_ram:
            reason = (
                f"RAM usage {snap.ram_percent:.1f}% exceeds limit {self._max_ram:.0f}% "
                f"({snap.ram_available_mb:.0f} MB available)"
            )
            logger.warning("Inference rejected: %s", reason)
            return False, reason

        if snap.cpu_percent > self._max_cpu:
            reason = f"CPU usage {snap.cpu_percent:.1f}% exceeds limit {self._max_cpu:.0f}%"
            logger.warning("Inference rejected: %s", reason)
            return False, reason

        return True, "ok"

    async def acquire(self) -> tuple[bool, str]:
        """Try to acquire an inference slot.

        Checks hardware first, then the concurrency semaphore.  Returns the
        same ``(ok, reason)`` tuple as :meth:`check_capacity`.
        """
        ok, reason = self.check_capacity()
        if not ok:
            return False, reason

        acquired = self._semaphore._value > 0  # noqa: SLF001 — peek at availability
        if not acquired:
            return False, (
                f"Max concurrent inferences reached ({self._active_inferences} running)"
            )

        await self._semaphore.acquire()
        self._active_inferences += 1
        logger.debug(
            "Inference slot acquired (%d/%d active)",
            self._active_inferences,
            self._active_inferences + self._semaphore._value,  # noqa: SLF001
        )
        return True, "ok"

    def release(self) -> None:
        """Release a previously acquired inference slot."""
        self._semaphore.release()
        self._active_inferences = max(0, self._active_inferences - 1)
        logger.debug("Inference slot released (%d active)", self._active_inferences)

    @property
    def active_inferences(self) -> int:
        return self._active_inferences

    # ── GPU detection ───────────────────────────────────────────────────────

    @staticmethod
    def _probe_gpu() -> tuple[bool, float | None, float | None]:
        """Best-effort GPU memory probe via ``nvidia-smi``.

        Returns ``(available, free_mb, total_mb)``.  Falls back gracefully
        if no NVIDIA GPU is present.
        """
        try:
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0:
                line = result.stdout.strip().split("\n")[0]
                free_str, total_str = line.split(",")
                return True, float(free_str.strip()), float(total_str.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass

        return False, None, None


# ── Singleton ───────────────────────────────────────────────────────────────

_monitor: HardwareMonitor | None = None


def get_hardware_monitor() -> HardwareMonitor:
    """Return the cached :class:`HardwareMonitor` singleton.

    Must be initialised via :func:`init_hardware_monitor` first (done during
    app startup).
    """
    if _monitor is None:
        raise RuntimeError("HardwareMonitor not initialised — call init_hardware_monitor() first")
    return _monitor


def init_hardware_monitor(settings: Settings) -> HardwareMonitor:
    """Create and cache the :class:`HardwareMonitor` singleton."""
    global _monitor
    _monitor = HardwareMonitor(settings)
    return _monitor
