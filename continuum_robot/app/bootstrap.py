"""Application bootstrap helpers."""

from dataclasses import dataclass

from continuum_robot.app.service_registry import ServiceRegistry
from continuum_robot.config.config_loader import ConfigLoader


@dataclass
class AppContext:
    """Container for top-level services used by the GUI and scripts."""

    config_loader: ConfigLoader
    services: ServiceRegistry


def build_app_context() -> AppContext:
    """Build and return application context with lightweight service stubs."""
    config_loader = ConfigLoader()
    services = ServiceRegistry()
    return AppContext(config_loader=config_loader, services=services)
