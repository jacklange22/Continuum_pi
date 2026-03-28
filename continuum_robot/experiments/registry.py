"""Registry for canonical experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from continuum_robot.experiments.framework import BaseExperiment


@dataclass(frozen=True)
class ExperimentDescriptor:
    """Registered experiment metadata."""

    name: str
    description: str
    factory: Callable[[dict[str, Any] | None], BaseExperiment]


class ExperimentRegistry:
    """Map experiment names to constructors."""

    def __init__(self) -> None:
        self._descriptors: dict[str, ExperimentDescriptor] = {}

    def register(
        self,
        *,
        name: str,
        description: str,
        factory: Callable[[dict[str, Any] | None], BaseExperiment],
    ) -> None:
        """Register one experiment by name."""
        key = str(name).strip()
        if not key:
            raise ValueError("Experiment name must not be empty")
        self._descriptors[key] = ExperimentDescriptor(
            name=key,
            description=str(description).strip(),
            factory=factory,
        )

    def create(self, name: str, config: dict[str, Any] | None = None) -> BaseExperiment:
        """Instantiate one registered experiment."""
        descriptor = self.get(name)
        return descriptor.factory(config or {})

    def get(self, name: str) -> ExperimentDescriptor:
        """Return one registered experiment descriptor."""
        key = str(name).strip()
        if key not in self._descriptors:
            raise KeyError(f"Unknown experiment: {name}")
        return self._descriptors[key]

    def list_descriptors(self) -> list[ExperimentDescriptor]:
        """Return all registered experiments in sorted order."""
        return [self._descriptors[key] for key in sorted(self._descriptors)]
