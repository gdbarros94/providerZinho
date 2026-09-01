"""Cross-platform thermal and resource provider.
Handles different OS environments (Linux, Android, Docker) to provide
consistent temperature and load metrics.
"""

from __future__ import annotations

import os
import logging
import platform
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ThermalProvider(ABC):
    @abstractmethod
    def get_temperature(self) -> float | None:
        """Returns temperature in Celsius."""
        pass

class LinuxThermalProvider(ThermalProvider):
    def get_temperature(self) -> float | None:
        # Try common thermal zones
        for zone in range(10):
            path = f"/sys/class/thermal/thermal_zone{zone}/temp"
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        return int(f.read().strip()) / 1000.0
                except (ValueError, IOError):
                    continue
        return None

class AndroidThermalProvider(LinuxThermalProvider):
    def get_temperature(self) -> float | None:
        # Android often uses different paths or requires dumpsys
        # Fallback to Linux provider first
        temp = super().get_temperature()
        if temp is not None:
            return temp
        return None

class GenericThermalProvider(ThermalProvider):
    def get_temperature(self) -> float | None:
        # Fallback for Docker/AMD64 where /sys/class/thermal might be missing
        # Returns None to indicate temperature monitoring is unavailable
        return None

def get_thermal_provider() -> ThermalProvider:
    system = platform.system().lower()
    if system == "linux":
        # Simple check for Android
        if "android" in platform.version().lower():
            return AndroidThermalProvider()
        return LinuxThermalProvider()
    return GenericThermalProvider()
