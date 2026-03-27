"""Main window integration scaffold for controllers and tabs."""

from __future__ import annotations

from continuum_robot.app.bootstrap import AppContext
from continuum_robot.gui.controllers.registration_controller import RegistrationController
from continuum_robot.gui.controllers.tracking_controller import TrackingController
from continuum_robot.gui.tabs.registration_tab import RegistrationTab
from continuum_robot.gui.tabs.tracking_tab import TrackingTab


class AppWindow:
    """Lightweight container that wires controllers and tab states."""

    def __init__(self, context: AppContext) -> None:
        settings = context.config_loader.load_settings()
        tracker_manager = context.services.get("tracker_manager")
        live_registration = context.services.get("live_registration")

        self.tracking_controller = TrackingController(tracker_manager=tracker_manager, settings=settings)
        self.registration_controller = RegistrationController(
            live_registration=live_registration,
            registration_config_path=context.config_loader.base_dir / "registration.yaml",
        )

        self.tracking_tab = TrackingTab()
        self.registration_tab = RegistrationTab()

    def show(self) -> None:
        self.refresh()
        print("GUI controllers wired. Attach PySide widgets to tab/controller state as needed.")

    def refresh(self) -> None:
        self.tracking_tab.update(self.tracking_controller.refresh())
        self.registration_tab.update(self.registration_controller.state)

    def shutdown(self) -> None:
        self.tracking_controller.shutdown()
