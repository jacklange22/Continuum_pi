"""Minimal motor-control policy helper for torque lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Callable


@dataclass
class TorqueDisarmAttempt:
    """One per-servo torque disable attempt result."""

    servo_id: int
    success: bool
    error: str | None = None


@dataclass
class TorqueDisarmReport:
    """Structured best-effort torque disable report."""

    owner: str
    reason: str
    target_servo_ids: list[int]
    attempts: list[TorqueDisarmAttempt] = field(default_factory=list)
    best_effort: bool = True

    @property
    def success_count(self) -> int:
        return sum(1 for attempt in self.attempts if attempt.success)

    @property
    def failure_count(self) -> int:
        return sum(1 for attempt in self.attempts if not attempt.success)


@dataclass
class PretensionTorquePolicyOutcome:
    """Torque cleanup policy result for one pretension terminal status."""

    policy: str
    action: str
    attempted: bool
    success: bool | None
    error: str | None = None


class MotorControlSupervisor:
    """Small policy gateway for deterministic torque lifecycle behavior."""

    def __init__(
        self,
        *,
        configured_servo_ids_provider: Callable[[], list[int]],
        last_commanded_servo_ids_provider: Callable[[], list[int]],
        write_torque_enable: Callable[[int, bool], None],
        logger: logging.Logger | None = None,
    ) -> None:
        self._configured_servo_ids_provider = configured_servo_ids_provider
        self._last_commanded_servo_ids_provider = last_commanded_servo_ids_provider
        self._write_torque_enable = write_torque_enable
        self._log = logger or logging.getLogger(__name__)

    def known_servo_ids(self, additional_servo_ids: list[int] | None = None) -> list[int]:
        """Return deterministic known servo IDs for torque cleanup."""
        configured = [int(value) for value in (self._configured_servo_ids_provider() or [])]
        commanded = [int(value) for value in (self._last_commanded_servo_ids_provider() or [])]
        additional = [int(value) for value in (additional_servo_ids or [])]
        ordered: list[int] = []
        for servo_id in configured + additional + commanded:
            if int(servo_id) not in ordered:
                ordered.append(int(servo_id))
        return ordered

    def disarm_all_known(
        self,
        *,
        reason: str,
        owner: str,
        best_effort: bool = True,
        additional_servo_ids: list[int] | None = None,
    ) -> TorqueDisarmReport:
        """Attempt torque-off on all known IDs, continuing on failures."""
        targets = self.known_servo_ids(additional_servo_ids=additional_servo_ids)
        report = TorqueDisarmReport(
            owner=str(owner),
            reason=str(reason),
            target_servo_ids=list(targets),
            best_effort=bool(best_effort),
        )
        self._log.info(
            "Torque disarm start | owner=%s | reason=%s | targets=%s | best_effort=%s",
            report.owner,
            report.reason,
            report.target_servo_ids,
            report.best_effort,
        )
        for servo_id in report.target_servo_ids:
            try:
                self._write_torque_enable(int(servo_id), False)
            except Exception as exc:
                error = str(exc)
                report.attempts.append(
                    TorqueDisarmAttempt(
                        servo_id=int(servo_id),
                        success=False,
                        error=error,
                    )
                )
                lowered_error = error.lower()
                expected_disconnect_failure = (
                    str(report.reason).strip().lower() == "disconnect"
                    and any(
                        token in lowered_error
                        for token in (
                            "not connected",
                            "disconnected",
                            "port is not open",
                            "port not open",
                            "incorrect status packet",
                            "txrxresult",
                            "no status packet",
                        )
                    )
                )
                log_fn = self._log.info if expected_disconnect_failure else self._log.warning
                log_fn(
                    "Torque disarm failed | owner=%s | servo_id=%s | reason=%s | expected_disconnect_failure=%s | error=%s",
                    report.owner,
                    int(servo_id),
                    report.reason,
                    expected_disconnect_failure,
                    error,
                )
                if not report.best_effort:
                    raise RuntimeError(
                        f"Torque disable failed for servo {int(servo_id)} during {report.reason}: {error}"
                    ) from exc
            else:
                report.attempts.append(
                    TorqueDisarmAttempt(
                        servo_id=int(servo_id),
                        success=True,
                        error=None,
                    )
                )
                self._log.info(
                    "Torque disarm ok | owner=%s | servo_id=%s | reason=%s",
                    report.owner,
                    int(servo_id),
                    report.reason,
                )
        self._log.info(
            "Torque disarm complete | owner=%s | reason=%s | success=%s | failure=%s",
            report.owner,
            report.reason,
            report.success_count,
            report.failure_count,
        )
        return report

    def apply_pretension_terminal_policy(
        self,
        *,
        servo_id: int,
        result_success: bool,
        result_status: str,
        armed_torque_during_run: bool,
        owner: str,
        keep_torque_on_after_success: bool = True,
    ) -> PretensionTorquePolicyOutcome:
        """Apply deterministic torque cleanup for pretension terminal states."""
        status = str(result_status or "")
        if bool(result_success) and bool(keep_torque_on_after_success):
            self._log.info(
                "Pretension torque policy | owner=%s | servo_id=%s | status=%s | action=leave_on_after_success",
                str(owner),
                int(servo_id),
                status,
            )
            return PretensionTorquePolicyOutcome(
                policy="pretension_terminal_cleanup",
                action="left_on_after_success",
                attempted=False,
                success=True,
                error=None,
            )
        self._log.info(
            "Pretension torque policy | owner=%s | servo_id=%s | status=%s | action=disarm_after_non_success | armed_during_run=%s",
            str(owner),
            int(servo_id),
            status,
            bool(armed_torque_during_run),
        )
        try:
            self._write_torque_enable(int(servo_id), False)
        except Exception as exc:
            error = str(exc)
            self._log.warning(
                "Pretension torque cleanup failed | owner=%s | servo_id=%s | status=%s | error=%s",
                str(owner),
                int(servo_id),
                status,
                error,
            )
            return PretensionTorquePolicyOutcome(
                policy="pretension_terminal_cleanup",
                action="disarm_after_terminal_state",
                attempted=True,
                success=False,
                error=error,
            )
        self._log.info(
            "Pretension torque cleanup ok | owner=%s | servo_id=%s | status=%s | action=disarm",
            str(owner),
            int(servo_id),
            status,
        )
        return PretensionTorquePolicyOutcome(
            policy="pretension_terminal_cleanup",
            action="disarm_after_terminal_state",
            attempted=True,
            success=True,
            error=None,
        )
