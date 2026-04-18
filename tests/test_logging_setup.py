from __future__ import annotations

import logging
from pathlib import Path

from continuum_robot.utils.logging_setup import configure_session_logging, current_session_log_path


def test_configure_session_logging_creates_fresh_log_file_per_call(tmp_path: Path) -> None:
    first_log = configure_session_logging(project_root=tmp_path)
    logging.getLogger("tests.logging").info("first session message")

    second_log = configure_session_logging(project_root=tmp_path)
    logging.getLogger("tests.logging").info("second session message")

    assert first_log.exists()
    assert second_log.exists()
    assert first_log != second_log
    assert current_session_log_path() == second_log

    first_text = first_log.read_text(encoding="utf-8")
    second_text = second_log.read_text(encoding="utf-8")
    assert "first session message" in first_text
    assert "second session message" not in first_text
    assert "second session message" in second_text
