"""Experiment loop coordinator scaffold."""

from continuum_robot.experiments.experiment_models import ExperimentPoint


class ExperimentRunner:
    """Coordinates point execution, settle, sampling, and logging."""

    def run(self, points: list[ExperimentPoint]) -> None:
        """Scaffold no-op loop."""
        for _point in points:
            # Intentionally scaffold-only; hardware behavior added later.
            pass
