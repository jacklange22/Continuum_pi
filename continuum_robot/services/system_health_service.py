"""Aggregate subsystem health for diagnostics and GUI status surfaces."""

from __future__ import annotations

import copy

from continuum_robot.services.models import (
    HEALTH_DEGRADED,
    HEALTH_FAILED,
    HEALTH_HEALTHY,
    ServiceHealthSnapshot,
    SystemHealthSnapshot,
)
from continuum_robot.services.registration_service import RegistrationService
from continuum_robot.services.tracking_service import TrackingService
from continuum_robot.utils.time_utils import utc_now_iso


class SystemHealthService:
    """Aggregate tracking and registration health into one operator-facing summary."""

    def __init__(
        self,
        tracking_service: TrackingService,
        registration_service: RegistrationService,
        *,
        config_source: str,
    ) -> None:
        self.tracking_service = tracking_service
        self.registration_service = registration_service
        self.config_source = config_source

    def start(self) -> None:
        """Start the health service.

        This service is computed on demand, so start is intentionally a no-op.
        """

    def stop(self) -> None:
        """Stop the health service.

        This service is computed on demand, so stop is intentionally a no-op.
        """

    def get_snapshot(self) -> SystemHealthSnapshot:
        """Return an aggregated health snapshot."""
        tracking_snapshot = self.tracking_service.get_snapshot()
        tracking = tracking_snapshot.health
        registration = self.registration_service.get_snapshot().health

        if HEALTH_FAILED in {tracking.health, registration.health}:
            overall_health = HEALTH_FAILED
        elif HEALTH_DEGRADED in {tracking.health, registration.health}:
            overall_health = HEALTH_DEGRADED
        else:
            overall_health = HEALTH_HEALTHY

        overall = ServiceHealthSnapshot(
            name="system_health_service",
            health=overall_health,
            state="running",
            status=self._compose_status(tracking.status, registration.status),
            current_config_source=self.config_source,
            details={
                "tracking_state": tracking.state,
                "registration_state": registration.state,
            },
        )
        return SystemHealthSnapshot(
            generated_at_utc=utc_now_iso(),
            health=overall,
            tracking=copy.deepcopy(tracking),
            registration=copy.deepcopy(registration),
            summary={
                "tracking_faults": list(tracking.details.get("faults", [])),
                "tracking_backend_identity": tracking_snapshot.backend_identity,
                "registration_fre_mm": registration.details.get("fre_mm"),
                "registration_pending_accept": registration.details.get("pending_accept"),
            },
        )

    @staticmethod
    def _compose_status(tracking_status: str, registration_status: str) -> str:
        return f"Tracking: {tracking_status} | Registration: {registration_status}"
