"""Experiment lifecycle abstractions shared by all canonical experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
import time
from typing import Any, Callable

from continuum_robot.config.settings import Settings
from continuum_robot.experiments.schemas import ExperimentDatasetPaths, ExperimentMetadata, ExperimentTimeseriesSample


def _config_to_dict(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    raise TypeError(f"Unsupported experiment config type: {type(config)!r}")


@dataclass
class ExperimentHardwareRequirements:
    """Declared hardware/runtime needs for one experiment."""

    tracking_required: bool = False
    servo_required: bool = False
    registration_required: bool = False
    mock_compatible: bool = True


@dataclass
class ExperimentContext:
    """Shared runtime context given to every experiment run."""

    project_root: Path
    settings: Settings
    tracking_service: Any
    servo_service: Any
    registration_path: Path
    output_root: Path
    run_output_dir: Path | None = None
    monotonic_fn: Callable[[], float] = time.monotonic
    sleep_fn: Callable[[float], None] = time.sleep
    penprobe_live_gui_hz: Callable[[], float | None] | None = None


@dataclass
class ExperimentSession:
    """Mutable run session passed through experiment lifecycle hooks."""

    context: ExperimentContext
    metadata: ExperimentMetadata
    stop_requested: Callable[[], bool] | None = None
    progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None
    sample_callback: Callable[[ExperimentTimeseriesSample], None] | None = None
    samples: list[ExperimentTimeseriesSample] = field(default_factory=list)
    stage_pass_fail: dict[str, str] = field(default_factory=dict)
    stage_messages: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    warning_messages: list[str] = field(default_factory=list)
    error_messages: list[str] = field(default_factory=list)
    total_progress_steps: int = 0
    completed_progress_steps: int = 0
    started_monotonic: float = field(init=False)

    def __post_init__(self) -> None:
        self.started_monotonic = self.context.monotonic_fn()

    def config_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable config payload for this run."""
        return dict(self.metadata.config_used)

    def set_stage(self, stage: str, status: str, message: str = "") -> None:
        """Record one lifecycle stage outcome."""
        self.stage_pass_fail[stage] = status
        if message:
            self.stage_messages[stage] = message

    def add_sample(self, sample: ExperimentTimeseriesSample) -> None:
        """Append one canonical sample."""
        self.samples.append(sample)
        if self.sample_callback is not None:
            self.sample_callback(sample)

    def add_warning(self, message: str) -> None:
        """Record one warning message."""
        if message not in self.warning_messages:
            self.warning_messages.append(message)

    def add_error(self, message: str) -> None:
        """Record one error message."""
        if message not in self.error_messages:
            self.error_messages.append(message)

    def set_metric(self, key: str, value: Any) -> None:
        """Store one experiment-specific metric."""
        self.metrics[key] = value

    def elapsed_s(self) -> float:
        """Return elapsed monotonic time since the run started."""
        return float(self.context.monotonic_fn() - self.started_monotonic)

    def raise_if_stop_requested(self) -> None:
        """Abort if the caller has requested stop."""
        if self.stop_requested is not None and self.stop_requested():
            raise RuntimeError("Experiment run stopped by operator.")

    def update_progress(self, current: int, total: int, last_payload: dict[str, Any] | None = None) -> None:
        """Emit progress updates to the caller."""
        self.completed_progress_steps = int(current)
        self.total_progress_steps = int(total)
        if self.progress_callback is not None:
            self.progress_callback(int(current), int(total), dict(last_payload or {}))


class BaseExperiment:
    """Base class for canonical experiments."""

    name = ""
    description = ""
    hardware_requirements = ExperimentHardwareRequirements()

    def __init__(self, config: Any) -> None:
        self.config = config

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "BaseExperiment":
        """Construct the experiment from a dictionary config payload."""
        return cls(config=payload or {})

    def config_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable config payload."""
        return _config_to_dict(self.config)

    def setup(self, session: ExperimentSession) -> None:
        """Prepare runtime state before precheck."""

    def precheck(self, session: ExperimentSession) -> None:
        """Validate prerequisites before execute."""

    def execute(self, session: ExperimentSession) -> None:
        """Run the main experiment body."""
        raise NotImplementedError

    def finalize(self, session: ExperimentSession) -> None:
        """Perform cleanup after execute or failure."""

    def summarize(self, session: ExperimentSession) -> dict[str, Any]:
        """Return experiment-specific summary metrics."""
        return dict(session.metrics)

    def write_outputs(
        self,
        session: ExperimentSession,
        paths: ExperimentDatasetPaths,
        summary,
    ) -> None:
        """Write any additional experiment-specific artifacts into the canonical run directory."""
