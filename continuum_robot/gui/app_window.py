"""Main window for the operator GUI."""

from __future__ import annotations

import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

from continuum_robot.app.bootstrap import AppContext, build_app_context
from continuum_robot.gui.controllers.data_management_controller import DataManagementController
from continuum_robot.gui.controllers.experiment_controller import ExperimentController
from continuum_robot.gui.controllers.modeling_controller import ModelingController
from continuum_robot.gui.controllers.pretension_controller import PretensionController
from continuum_robot.gui.controllers.registration_controller import RegistrationController
from continuum_robot.gui.controllers.runtime_tip_calibration_controller import RuntimeTipCalibrationController
from continuum_robot.gui.controllers.servos_controller import ServosController
from continuum_robot.gui.controllers.system_controller import SystemController
from continuum_robot.gui.controllers.tracker_mvp_controller import TrackerMvpController
from continuum_robot.gui.controllers.tracking_controller import TrackingController
from continuum_robot.gui.tabs.data_management_tab import DataManagementTab
from continuum_robot.gui.tabs.experiment_tab import ExperimentTab
from continuum_robot.gui.tabs.modeling_tab import ModelingTab
from continuum_robot.gui.tabs.pretension_tab import PretensionTab
from continuum_robot.gui.tabs.registration_tab import RegistrationTab
from continuum_robot.gui.tabs.servos_tab import ServosTab
from continuum_robot.gui.tabs.system_tab import SystemTab
from continuum_robot.gui.tabs.tracking_tab import TrackingTab
from continuum_robot.gui.theme import apply_dark_theme
from continuum_robot.gui.widgets.runtime_tip_calibration_dialog import RuntimeTipCalibrationDialog
from continuum_robot.servos.telemetry_diagnostics import (
    DEFAULT_SERVO_FULL_REFRESH_DIVISOR,
    DEFAULT_SYSTEM_SUMMARY_REFRESH_DIVISOR,
)


class AppWindow(QMainWindow):
    """Main operator window with all platform tabs."""

    MIN_REFRESH_INTERVAL_MS = 50
    SERVO_FULL_REFRESH_DIVISOR = DEFAULT_SERVO_FULL_REFRESH_DIVISOR
    SYSTEM_SUMMARY_REFRESH_DIVISOR = DEFAULT_SYSTEM_SUMMARY_REFRESH_DIVISOR

    def __init__(self, context: AppContext) -> None:
        super().__init__()
        apply_dark_theme(QApplication.instance())
        self.context = context
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self.refresh)
        self._servo_full_refresh_due = True
        self._servo_refresh_cycle = 0
        self._system_summary_refresh_due = True
        self._system_summary_refresh_cycle = 0

        self._build_workspace(context, selected_tab_label="System")
        self.setWindowTitle("Continuum Robot Operator Console")
        self.resize(1280, 900)
        self.statusBar().showMessage("Ready")
        self.refresh()

    def refresh(self) -> None:
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
            system_state = self.system_controller.refresh()
            if self.system_controller.state.dynamixel_connected:
                if self._system_summary_refresh_due:
                    system_state = self.system_controller.refresh_readiness(include_scan=False)
                    self._system_summary_refresh_due = False
                    self._system_summary_refresh_cycle = 0
                else:
                    self._system_summary_refresh_cycle = (
                        self._system_summary_refresh_cycle + 1
                    ) % self.SYSTEM_SUMMARY_REFRESH_DIVISOR
                    if self._system_summary_refresh_cycle == 0:
                        system_state = self.system_controller.refresh_readiness(include_scan=False)
                    else:
                        system_state = self.system_controller.refresh()
            self.system_tab.update(system_state)
            self.statusBar().showMessage(system_state.status_message)
            return
        elif current_widget is self.servos_tab:
            servo_state = self._refresh_servo_state()
            if getattr(self.servos_controller, "latest_runtime_snapshot", None) is not None:
                self.system_controller.sync_servo_runtime_snapshot(self.servos_controller.latest_runtime_snapshot)
            else:
                self.system_controller.sync_servo_bringup_state(servo_state)
            self.servos_tab.update(servo_state)
            self.statusBar().showMessage(servo_state.status_message)
            return
        elif current_widget is self.pretension_tab:
            pretension_state = self.pretension_controller.refresh()
            if getattr(self.pretension_controller, "latest_runtime_snapshot", None) is not None:
                self.system_controller.sync_servo_runtime_snapshot(self.pretension_controller.latest_runtime_snapshot)
            self.pretension_tab.update(pretension_state)
            self.statusBar().showMessage(pretension_state.status_message)
            return
        elif current_widget is self.experiment_tab:
            if self.experiment_controller.should_periodically_refresh_selected_experiment():
                experiment_state = self.experiment_controller.refresh_prerequisites()
                self.experiment_tab.update(experiment_state)
            else:
                experiment_state = self.experiment_controller.state
                if self.experiment_tab.needs_state_update(experiment_state):
                    self.experiment_tab.update(experiment_state)
            self.statusBar().showMessage(experiment_state.status_message)
            return
        elif current_widget is self.modeling_tab:
            modeling_state = self.modeling_controller.refresh()
            self.modeling_tab.update(modeling_state)
            self.statusBar().showMessage(modeling_state.status_message)
            return
        elif current_widget is self.data_management_tab:
            data_management_state = self.data_management_controller.refresh()
            self.data_management_tab.update(data_management_state)
            self.statusBar().showMessage(data_management_state.status_message)
            return
        system_state = self.system_controller.refresh()
        self.statusBar().showMessage(system_state.status_message)

    def _refresh_servo_state(self):
        if self._servo_full_refresh_due or self.servos_controller.state.single_servo_mode:
            servo_state = self.servos_controller.refresh()
            self._servo_full_refresh_due = False
            self._servo_refresh_cycle = 0
            return servo_state
        self._servo_refresh_cycle = (self._servo_refresh_cycle + 1) % self.SERVO_FULL_REFRESH_DIVISOR
        if self._servo_refresh_cycle == 0:
            return self.servos_controller.refresh()
        return self.servos_controller.refresh_selected_servo()

    def _handle_tab_changed(self, _index: int) -> None:
        if self.tab_widget.currentWidget() in {self.system_tab, self.servos_tab, self.pretension_tab}:
            self._servo_full_refresh_due = True
        if self.tab_widget.currentWidget() is self.system_tab:
            self._system_summary_refresh_due = True
        self.refresh()

    def shutdown(self) -> None:
        self._refresh_timer.stop()
        self._shutdown_workspace()

    def _build_workspace(self, context: AppContext, *, selected_tab_label: str = "System") -> None:
        self.context = context
        settings = context.settings
        tracking_service = context.services.get("tracking_service")
        registration_service = context.services.get("registration_service")
        runtime_tip_calibration_service = context.services.get("runtime_tip_calibration_service")
        servo_service = context.services.get("servo_service")
        openrb_client = context.services.get("openrb_client")
        experiment_loader = context.services.get("experiment_loader")
        experiment_runner = context.services.get("experiment_runner")
        try:
            experiment_dataset_writer = context.services.get("experiment_dataset_writer")
        except KeyError:
            experiment_dataset_writer = None

        self.system_controller = SystemController(
            tracking_service=tracking_service,
            openrb_client=openrb_client,
            servo_service=servo_service,
            settings=settings,
            config_loader=context.config_loader,
            session_log_path=(str(context.session_log_path) if context.session_log_path is not None else None),
        )
        self.servos_controller = ServosController(servo_service=servo_service, settings=settings)
        self.pretension_controller = PretensionController(
            servo_service=servo_service,
            settings=settings,
            config_loader=context.config_loader,
        )
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
        self.runtime_tip_calibration_controller = RuntimeTipCalibrationController(
            runtime_tip_calibration_service=runtime_tip_calibration_service,
        )
        self.runtime_tip_calibration_dialog = RuntimeTipCalibrationDialog(
            self.runtime_tip_calibration_controller,
            parent=self,
        )
        self.experiment_controller = ExperimentController(
            experiment_loader=experiment_loader,
            experiment_runner=experiment_runner,
            registration_path=self.tracking_controller.registration_path,
            servo_service=servo_service,
            tracking_service=tracking_service,
        )
        self.modeling_controller = ModelingController(
            project_root=context.project_root,
            dataset_output_root=getattr(experiment_dataset_writer, "output_root", context.project_root / "data" / "experiments"),
            artifact_root=context.project_root / "data" / "models" / "ann",
            results_root=context.project_root / "data" / "modeling_results",
        )
        self.data_management_controller = DataManagementController(project_root=context.project_root)
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
            open_runtime_tip_calibration=self._open_runtime_tip_calibration,
        )
        self.servos_tab = ServosTab(
            self.servos_controller,
            apply_runtime_parameters=self._save_and_apply_servo_jog_settings,
        )
        self.pretension_tab = PretensionTab(self.pretension_controller)
        self.experiment_tab = ExperimentTab(self.experiment_controller)
        self.modeling_tab = ModelingTab(self.modeling_controller)
        self.data_management_tab = DataManagementTab(self.data_management_controller)
        for widget, label in (
            (self.system_tab, "System"),
            (self.tracking_tab, "Tracking"),
            (self.registration_tab, "Registration"),
            (self.servos_tab, "Servos"),
            (self.pretension_tab, "Pretension"),
            (self.experiment_tab, "Experiment"),
            (self.modeling_tab, "Modeling"),
            (self.data_management_tab, "Data"),
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
        self._system_summary_refresh_due = True
        self._system_summary_refresh_cycle = 0
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
        settle_s = float(self.context.settings.serial.openrb_settings.get("port_settle_time_s", 0.15) or 0.0)
        if settle_s > 0.0:
            time.sleep(settle_s)
        self._build_workspace(reloaded_context, selected_tab_label="System")
        self.system_controller.state.saved_overrides_path = str(saved_path)
        self.system_controller.state.status_message = (
            f"Applied runtime parameters from {saved_path}. Runtime reloaded; reconnect hardware if needed."
        )
        self.system_controller.state.last_error = None
        self.refresh()

    def _save_and_apply_servo_jog_settings(self, *, fine_jog_step_ticks: int, coarse_jog_step_ticks: int) -> None:
        self._save_and_apply_runtime_parameters(
            mock_mode=bool(self.system_controller.state.mock_mode),
            robot_config=str(self.system_controller.state.robot_config),
            openrb_port=str(self.system_controller.state.openrb_port),
            baudrate=int(self.system_controller.state.baudrate),
            poll_rate_hz=int(self.system_controller.state.poll_rate_hz),
            fine_jog_step_ticks=int(fine_jog_step_ticks),
            coarse_jog_step_ticks=int(coarse_jog_step_ticks),
            telemetry_freshness_timeout_s=float(self.system_controller.state.telemetry_freshness_timeout_s),
        )

    def _shutdown_workspace(self) -> None:
        dialog = getattr(self, "runtime_tip_calibration_dialog", None)
        if dialog is not None:
            try:
                dialog.close()
            except Exception:
                pass
        experiment_tab = getattr(self, "experiment_tab", None)
        if experiment_tab is not None:
            try:
                experiment_tab.shutdown()
            except Exception:
                pass
        for attribute in ("servos_controller", "pretension_controller", "experiment_controller", "modeling_controller", "tracking_controller"):
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

    def _open_runtime_tip_calibration(self) -> None:
        dialog = getattr(self, "runtime_tip_calibration_dialog", None)
        if dialog is None:
            return
        registration_controller = getattr(self, "registration_controller", None)
        registration_state = getattr(registration_controller, "state", None)
        runtime_tip_mode = str(getattr(registration_state, "runtime_tip_mode", "latest_accepted") or "latest_accepted")
        if runtime_tip_mode == "quick_4_point":
            dialog.set_preferred_session_mode("quick_4_point")
        elif runtime_tip_mode == "latest_accepted":
            dialog.set_preferred_session_mode("full_hat")
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        dialog.refresh()
