"""Logging setup helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
import sys


_SESSION_LOG_PATH: Path | None = None


def configure_session_logging(
    *,
    project_root: Path | None = None,
    level: int = logging.INFO,
) -> Path:
    """Configure root logging for one GUI/app launch and return the session log path.

    A new log file is created on every call so each launch gets a clean session log.
    """
    global _SESSION_LOG_PATH
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    root.setLevel(int(level))

    repo_root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    logs_dir = repo_root / "data" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    session_log_path = logs_dir / f"operator_gui_{timestamp}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(threadName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_handler.setLevel(int(level))
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(session_log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(int(level))
    file_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
    root.addHandler(file_handler)
    _SESSION_LOG_PATH = session_log_path
    logging.getLogger(__name__).info("Session logging initialized at %s", session_log_path)
    return session_log_path


def current_session_log_path() -> Path | None:
    """Return the current per-launch session log path when initialized."""
    return _SESSION_LOG_PATH


def configure_logging(level: int = logging.INFO) -> None:
    """Backward-compatible wrapper used by older call sites/tests."""
    configure_session_logging(level=level)
