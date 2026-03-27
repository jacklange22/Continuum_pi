"""Simple service registry for dependency wiring.

This intentionally stays minimal for v1 scaffold.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ServiceRegistry:
    """Name-to-service container."""

    services: dict[str, Any] = field(default_factory=dict)

    def register(self, name: str, service: Any) -> None:
        self.services[name] = service

    def get(self, name: str) -> Any:
        if name not in self.services:
            available = ", ".join(sorted(self.services))
            raise KeyError(f"Service {name!r} is not registered. Available services: {available}")
        return self.services[name]
