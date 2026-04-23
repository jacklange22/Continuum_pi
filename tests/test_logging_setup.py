from __future__ import annotations

import logging
from pathlib import Path

from continuum_robot.utils.logging_setup import configure_session_logging, current_session_log_path


def test_configure_session_logging_creates_fresh_log_file_per_call(tmp_path: Path) -> None:
    first_log = configure_session_logging(project_root=tmp_path)
    logging.getLogger("tests.logging").info("first session message")

    second_log = configure_session_logging(project_root=tmp_path)
    logging.getLogger("tests.logging").info("second session message")

    recent_dir = tmp_path / "data" / "logs" / "recent"
    rotated_first_log = recent_dir / first_log.name
    assert rotated_first_log.exists()
    assert second_log.exists()
    assert first_log != second_log
    assert current_session_log_path() == second_log
    assert rotated_first_log.parent.name == "recent"
    assert second_log.parent.name == "current"

    first_text = rotated_first_log.read_text(encoding="utf-8")
    second_text = second_log.read_text(encoding="utf-8")
    assert "first session message" in first_text
    assert "second session message" not in first_text
    assert "second session message" in second_text


def test_configure_session_logging_trims_recent_logs_to_30(tmp_path: Path) -> None:
    recent_dir = tmp_path / "data" / "logs" / "recent"
    recent_dir.mkdir(parents=True)
    for index in range(35):
        (recent_dir / f"operator_gui_20260101_0000{index:02d}.log").write_text("old\n", encoding="utf-8")

    current_log = configure_session_logging(project_root=tmp_path)

    recent_logs = sorted(recent_dir.glob("operator_gui_*.log"))
    assert current_log.exists()
    assert len(recent_logs) == 30
