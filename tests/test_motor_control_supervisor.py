from __future__ import annotations

from continuum_robot.servos.motor_control_supervisor import MotorControlSupervisor


def test_motor_control_supervisor_disarm_all_known_continues_after_failure() -> None:
    calls: list[tuple[int, bool]] = []

    def _write_torque_enable(servo_id: int, enabled: bool) -> None:
        calls.append((int(servo_id), bool(enabled)))
        if int(servo_id) == 2:
            raise RuntimeError("mock failure")

    supervisor = MotorControlSupervisor(
        configured_servo_ids_provider=lambda: [1, 2],
        last_commanded_servo_ids_provider=lambda: [3],
        write_torque_enable=_write_torque_enable,
    )

    report = supervisor.disarm_all_known(
        reason="shutdown",
        owner="test",
        best_effort=True,
    )

    assert report.target_servo_ids == [1, 2, 3]
    assert calls == [(1, False), (2, False), (3, False)]
    assert report.success_count == 2
    assert report.failure_count == 1
    assert report.attempts[1].servo_id == 2
    assert report.attempts[1].success is False


def test_motor_control_supervisor_pretension_failure_disarms_even_if_not_armed_by_routine() -> None:
    calls: list[tuple[int, bool]] = []

    def _write_torque_enable(servo_id: int, enabled: bool) -> None:
        calls.append((int(servo_id), bool(enabled)))

    supervisor = MotorControlSupervisor(
        configured_servo_ids_provider=lambda: [1],
        last_commanded_servo_ids_provider=lambda: [],
        write_torque_enable=_write_torque_enable,
    )

    outcome = supervisor.apply_pretension_terminal_policy(
        servo_id=1,
        result_success=False,
        result_status="timeout",
        armed_torque_during_run=False,
        owner="test",
        keep_torque_on_after_success=True,
    )

    assert calls == [(1, False)]
    assert outcome.action == "disarm_after_terminal_state"
    assert outcome.attempted is True
    assert outcome.success is True
