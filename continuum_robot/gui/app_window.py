"""Main window for the operator GUI."""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMainWindow, QTabWidget

from continuum_robot.app.bootstrap import AppContext
from continuum_robot.gui.controllers.experiment_controller import ExperimentController
from continuum_robot.gui.controllers.registration_controller import RegistrationController
from continuum_robot.gui.controllers.servos_controller import ServosController
from continuum_robot.gui.controllers.system_controller import SystemController
from continuum_robot.gui.controllers.tracking_controller import TrackingController
from continuum_robot.gui.tabs.experiment_tab import ExperimentTab
from continuum_robot.gui.tabs.registration_tab import RegistrationTab
from continuum_robot.gui.tabs.servos_tab import ServosTab
from continuum_robot.gui.tabs.system_tab import SystemTab
from continuum_robot.gui.tabs.tracking_tab import TrackingTab


class AppWindow(QMainWindow):
    """Main operator window with all platform tabs."""

    def __init__(self, context: AppContext) -> None:
        super().__init__()
        self.context = context
        settings = context.settings
        tracker_manager = context.services.get("tracker_manager")
        tracking_service = context.services.get("tracking_service")
        registration_service = context.services.get("registration_service")
        servo_service = context.services.get("servo_service")
        openrb_client = context.services.get("openrb_client")
        experiment_loader = context.services.get("experiment_loader")
        experiment_runner = context.services.get("experiment_runner")

        self.system_controller = SystemController(
            tracking_service=tracking_service,
            openrb_client=openrb_client,
            servo_service=servo_service,
            settings=settings,
        )
        self.servos_controller = ServosController(servo_service=servo_service, settings=settings)
        self.tracking_controller = TrackingController(
            tracking_service=tracking_service,
            settings=settings,
            registration_path=context.project_root / settings.calibration.latest_registration_path,
        )
        self.registration_controller = RegistrationController(
            registration_service=registration_service,
            registration_config=settings.registration,
        )
        self.registration_controller.load_latest_result()
        self.experiment_controller = ExperimentController(
            experiment_loader=experiment_loader,
            experiment_runner=experiment_runner,
            registration_path=self.tracking_controller.registration_path,
            servo_service=servo_service,
            tracker_manager=tracker_manager,
        )

        self.tab_widget = QTabWidget()
        self.system_tab = SystemTab(self.system_controller)
        self.servos_tab = ServosTab(self.servos_controller)
        self.tracking_tab = TrackingTab(self.tracking_controller)
        self.registration_tab = RegistrationTab(self.registration_controller)
        self.experiment_tab = ExperimentTab(self.experiment_controller)
        self.tab_widget.addTab(self.system_tab, "System")
        self.tab_widget.addTab(self.servos_tab, "Servos")
        self.tab_widget.addTab(self.tracking_tab, "Tracking")
        self.tab_widget.addTab(self.registration_tab, "Registration")
        self.tab_widget.addTab(self.experiment_tab, "Experiment")
        self.setCentralWidget(self.tab_widget)

        self.setWindowTitle("Continuum Robot Operator Console")
        self.resize(1280, 900)
        self.statusBar().showMessage("Ready")

        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self.refresh)
        interval_ms = max(100, int(1000 / max(1, settings.runtime.poll_rate_hz)))
        self._refresh_timer.start(interval_ms)
        self.refresh()

    def refresh(self) -> None:
        self.system_tab.update(self.system_controller.refresh())
        self.servos_tab.update(self.servos_controller.refresh())
        self.tracking_tab.update(self.tracking_controller.refresh())
        self.registration_tab.update(self.registration_controller.state)
        self.experiment_tab.update(self.experiment_controller.refresh_prerequisites())
        self.statusBar().showMessage(self.system_controller.state.status_message)

    def shutdown(self) -> None:
        self._refresh_timer.stop()
        self.experiment_controller.shutdown()
        self.tracking_controller.shutdown()
