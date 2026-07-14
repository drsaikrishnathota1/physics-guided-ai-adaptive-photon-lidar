"""Common interface for photon-allocation policies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.environment.adaptive_lidar_env import (
    AdaptiveLidarEnvironment,
    AdaptiveObservation,
)


class PhotonAllocationPolicy(ABC):
    """Abstract interface implemented by every baseline policy."""

    name: str

    @abstractmethod
    def select_region(
        self,
        observation: AdaptiveObservation,
        environment: AdaptiveLidarEnvironment,
    ) -> int:
        """Return the next candidate region to measure."""
        raise NotImplementedError
