"""Main window for the operator GUI."""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMainWindow, QTabWidget

from continuum_robot.app.bootstrap import AppContext, build_app_context
from continuum_robot.gui.controllers.experiment_controller import ExperimentController
from continuum_robot.gui.controllers.registration_controller import RegistrationController
from continuum_robot.gui.controllers.servos_controller import ServosController
from continuum_robot.gui.controllers.system_controller import SystemController
from continuum_robot.gui.controllers.tracker_mvp_controller import TrackerMvpController
from continuum_robot.gui.controllers.tracking_controller import TrackingController
from continuum_robot.gui.tabs.experiment_tab import ExperimentTab
from continuum_robot.gui.tabs.registration_tab import RegistrationTab
from continuum_robot.gui.tabs.servos_tab import ServosTab
from continuum_robot.gui.tabs.system_tab import SystemTab
from continuum_robot.gui.tabs.tracking_tab import TrackingTab


class AppWindow(QMainWindow):
    """Main operator window with all platform tabs."""

    MIN_REFRESH_INTERVAL_MS = 50
    SERVO_FULL_REFRESH_DIVISOR = 2

    def __init__(self, context: AppContext) -> None:
        super().__init__()
        self.context = context
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self.refresh)
        self._servo_full_refresh_due = True
        self._servo_refresh_cycle = 0

        self._build_workspace(context, selected_tab_label="System")
        self.setWindowTitle("Continuum Robot Operator Console")
        self.resize(1280, 900)
        self.statusBar().showMessage("Ready")
        self.refresh()

    def refresh(self) -> None:
        system_state = self.system_controller.refresh()
        current_widget = self.tab_widget.currentWidget()
        if current_widget is self.tracking_tab:
            tracker_mvp_state = self.tracker_mvp_controller.refresh()
            tracking_state = self.tracking_controller.refresh()
            self.tracking_tab.update(tracker_mvp_state, tracking_state)
            self.statusBar().showMessage(tracker_mvp_state.status_message)
            return
        if current_widget is self.registration_tab:
            registration_state = self.registration_controller.refresh()
            tracker_mvp_state = self.tracker_mvp_controller.refresh()
            self.registration_tab.update(registration_state, tracker_mvp_state)
            self.statusBar().showMessage(registration_state.status_message)
            return
        if current_widget is self.system_tab:
            if self.system_controller.state.dynamixel_connected:
                self.servos_controller.refresh_selected_servo()
            self.system_tab.update(system_state)
        elif current_widget is self.servos_tab:
            if self._servo_full_refresh_due or self.servos_controller.state.single_servo_mode:
                servo_state = self.servos_controller.refresh()
                self._servo_full_refresh_due = False
                self._servo_refresh_cycle = 0
            else:
                self._servo_refresh_cycle = (self._servo_refresh_cycle + 1) % self.SERVO_FULL_REFRESH_DIVISOR
                if self._servo_refresh_cycle == 0:
                    servo_state = self.servos_controller.refresh()
                else:
                    servo_state = self.servos_controller.refresh_selected_servo()
            self.servos_tab.update(servo_state)
        elif current_widget is self.experiment_tab:
            self.experiment_tab.update(self.experiment_controller.refresh_prerequisites())
        self.statusBar().showMessage(system_state.status_message)

    def _handle_tab_changed(self, _index: int) -> None:
        if self.tab_widget.currentWidget() in {self.system_tab, self.servos_tab}:
            self._servo_full_refresh_due = True
        self.refresh()

    def shutdown(self) -> None:
        self._refresh_timer.stop()
        self._shutdown_workspace()

    def _build_workspace(self, context: AppContext, *, selected_tab_label: str = "System") -> None:
        self.context = context
        settings = context.settings
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
            config_loader=context.config_loader,
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
            tracking_service=tracking_service,
        )
        self.tracker_mvp_controller = TrackerMvpController(
            tracking_service=tracking_service,
            registration_service=registration_service,
            registration_controller=self.registration_controller,
            experiment_runner=experiment_runner,
            settings=settings,
            project_root=context.project_root,
        )

        new_tab_widget = QTabWidget()
        self.system_tab = SystemTab(
            self.system_controller,
            apply_runtime_parameters=self._save_and_apply_runtime_parameters,
        )
        self.tracking_tab = TrackingTab(self.tracking_controller, workflow_controller=self.tracker_mvp_controller)
        self.registration_tab = RegistrationTab(
            self.registration_controller,
            workflow_controller=self.tracker_mvp_controller,
        )
        self.servos_tab = ServosTab(self.servos_controller)
        self.experiment_tab = ExperimentTab(self.experiment_controller)
        for widget, label in (
            (self.system_tab, "System"),
            (self.tracking_tab, "Tracking"),
            (self.registration_tab, "Registration"),
            (self.servos_tab, "Servos"),
            (self.experiment_tab, "Experiment"),
        ):
            new_tab_widget.addTab(widget, label)
        new_tab_widget.currentChanged.connect(self._handle_tab_changed)

        old_tab_widget = getattr(self, "tab_widget", None)
        self.tab_widget = new_tab_widget
        self.setCentralWidget(self.tab_widget)
        if old_tab_widget is not None:
            old_tab_widget.deleteLater()

        selected_index = 0
        for index in range(self.tab_widget.count()):
            if self.tab_widget.tabText(index) == selected_tab_label:
                selected_index = index
                break
        self.tab_widget.setCurrentIndex(selected_index)
        self._servo_full_refresh_due = True
        self._servo_refresh_cycle = 0
        self._refresh_timer.start(self._refresh_interval_ms(settings.runtime.poll_rate_hz))

    def _save_and_apply_runtime_parameters(self, **parameters) -> None:
        saved_path = self.system_controller.save_runtime_parameters(**parameters)
        try:
            reloaded_context = build_app_context()
        except Exception as exc:
            self.system_controller.state.last_error = str(exc)
            self.system_controller.state.status_message = (
                f"Saved runtime parameters to {saved_path}, but runtime reload failed: {exc}"
            )
            self.system_tab.update(self.system_controller.refresh())
            self.statusBar().showMessage(self.system_controller.state.status_message)
            raise

        self._refresh_timer.stop()
        self._shutdown_workspace()
        self._build_workspace(reloaded_context, selected_tab_label="System")
        self.system_controller.state.saved_overrides_path = str(saved_path)
        self.system_controller.state.status_message = (
            f"Applied runtime parameters from {saved_path}. Runtime reloaded; reconnect hardware if needed."
        )
        self.system_controller.state.last_error = None
        self.refresh()

    def _shutdown_workspace(self) -> None:
        for attribute in ("servos_controller", "experiment_controller", "tracking_controller"):
            controller = getattr(self, attribute, None)
            if controller is None:
                continue
            try:
                controller.shutdown()
            except Exception:
                pass
        system_controller = getattr(self, "system_controller", None)
        if system_controller is not None:
            try:
                system_controller.disconnect_openrb()
            except Exception:
                pass

    @classmethod
    def _refresh_interval_ms(cls, poll_rate_hz: int) -> int:
        return max(cls.MIN_REFRESH_INTERVAL_MS, int(1000 / max(1, int(poll_rate_hz))))
