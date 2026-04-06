"""Registry for canonical experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from continuum_robot.experiments.framework import BaseExperiment


@dataclass(frozen=True)
class ExperimentDescriptor:
    """Registered experiment metadata."""

    name: str
    title: str
    description: str
    category: str
    tags: tuple[str, ...]
    workspace_visible: bool
    default_config_path: str | None
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
        title: str | None = None,
        category: str = "validation",
        tags: list[str] | tuple[str, ...] | None = None,
        workspace_visible: bool = True,
        default_config_path: str | None = None,
        factory: Callable[[dict[str, Any] | None], BaseExperiment],
    ) -> None:
        """Register one experiment by name."""
        key = str(name).strip()
        if not key:
            raise ValueError("Experiment name must not be empty")
        self._descriptors[key] = ExperimentDescriptor(
            name=key,
            title=str(title or key.replace("_", " ").title()).strip(),
            description=str(description).strip(),
            category=str(category).strip() or "validation",
            tags=tuple(str(tag).strip() for tag in (tags or []) if str(tag).strip()),
            workspace_visible=bool(workspace_visible),
            default_config_path=(
                str(default_config_path).strip()
                if default_config_path not in (None, "")
                else None
            ),
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
