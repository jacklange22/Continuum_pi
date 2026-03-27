"""Servos tab widget."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.gui.controllers.servos_controller import ServosViewState


class ServosTab(QWidget):
    """Servo scan, calibration, and manual control UI."""

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.displacement_inputs: list[QDoubleSpinBox] = []

        self.connection_label = QLabel()
        self.ids_label = QLabel()
        self.neutral_label = QLabel()
        self.pretension_label = QLabel()

        self.assign_from_spin = QSpinBox()
        self.assign_from_spin.setRange(1, 252)
        self.assign_to_spin = QSpinBox()
        self.assign_to_spin.setRange(1, 252)
        self.assign_button = QPushButton("Assign ID")
        self.assign_button.clicked.connect(self._assign_id)

        self.jog_servo_spin = QSpinBox()
        self.jog_servo_spin.setRange(1, 252)
        self.jog_delta_spin = QSpinBox()
        self.jog_delta_spin.setRange(-4096, 4096)
        self.jog_minus_button = QPushButton("Jog -")
        self.jog_plus_button = QPushButton("Jog +")
        self.jog_minus_button.clicked.connect(lambda: self._jog(-abs(self.jog_delta_spin.value() or 20)))
        self.jog_plus_button.clicked.connect(lambda: self._jog(abs(self.jog_delta_spin.value() or 20)))

        self.scan_button = QPushButton("Scan Servos")
        self.scan_button.clicked.connect(lambda: self._safe_call(self.controller.scan))
        self.capture_neutral_button = QPushButton("Capture Neutral")
        self.capture_neutral_button.clicked.connect(lambda: self._safe_call(self.controller.capture_neutral_setpoints))
        self.save_neutral_button = QPushButton("Save Neutral")
        self.save_neutral_button.clicked.connect(lambda: self._safe_call(self.controller.save_neutral_setpoints))
        self.load_neutral_button = QPushButton("Load Neutral")
        self.load_neutral_button.clicked.connect(lambda: self._safe_call(self.controller.load_neutral_setpoints))
        self.pretension_button = QPushButton("Validate Pretension")
        self.pretension_button.clicked.connect(lambda: self._safe_call(self.controller.validate_pretension))
        self.apply_displacement_button = QPushButton("Apply Displacement")
        self.apply_displacement_button.clicked.connect(self._apply_displacement)

        summary_box = QGroupBox("Servo Summary")
        summary_layout = QFormLayout(summary_box)
        summary_layout.addRow("Connected", self.connection_label)
        summary_layout.addRow("Servo IDs", self.ids_label)
        summary_layout.addRow("Neutral setpoints", self.neutral_label)
        summary_layout.addRow("Pretension", self.pretension_label)

        scan_row = QHBoxLayout()
        scan_row.addWidget(self.scan_button)
        scan_row.addWidget(self.capture_neutral_button)
        scan_row.addWidget(self.save_neutral_button)
        scan_row.addWidget(self.load_neutral_button)
        scan_row.addWidget(self.pretension_button)

        jog_box = QGroupBox("ID and Jog Tools")
        jog_layout = QGridLayout(jog_box)
        jog_layout.addWidget(QLabel("Rename from"), 0, 0)
        jog_layout.addWidget(self.assign_from_spin, 0, 1)
        jog_layout.addWidget(QLabel("to"), 0, 2)
        jog_layout.addWidget(self.assign_to_spin, 0, 3)
        jog_layout.addWidget(self.assign_button, 0, 4)
        jog_layout.addWidget(QLabel("Jog servo"), 1, 0)
        jog_layout.addWidget(self.jog_servo_spin, 1, 1)
        jog_layout.addWidget(QLabel("Delta ticks"), 1, 2)
        jog_layout.addWidget(self.jog_delta_spin, 1, 3)
        jog_layout.addWidget(self.jog_minus_button, 1, 4)
        jog_layout.addWidget(self.jog_plus_button, 1, 5)

        displacement_box = QGroupBox("Tendon Displacement Command (cm)")
        self.displacement_layout = QGridLayout(displacement_box)
        self._rebuild_displacement_inputs(len(self.controller.state.tendon_displacements_cm))
        self.displacement_layout.addWidget(self.apply_displacement_button, 1, 0, 1, 4)

        self.telemetry_table = QTableWidget(0, 5)
        self.telemetry_table.setHorizontalHeaderLabels(["Servo ID", "Position", "Current mA", "Voltage mV", "Fault"])

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.addWidget(summary_box)
        layout.addLayout(scan_row)
        layout.addWidget(jog_box)
        layout.addWidget(displacement_box)
        layout.addWidget(self.telemetry_table)
        layout.addWidget(self.status_text)

    def update(self, state: ServosViewState) -> None:
        self.connection_label.setText("yes" if state.connected else "no")
        self.ids_label.setText(", ".join(str(sid) for sid in state.servo_ids) or "none")
        self.neutral_label.setText(
            ", ".join(f"{sid}:{tick}" for sid, tick in sorted(state.neutral_setpoints.items())) or "not saved"
        )
        self.pretension_label.setText(state.pretension_message)
        has_neutral = bool(state.neutral_setpoints)
        self.assign_button.setEnabled(state.connected)
        self.jog_minus_button.setEnabled(state.connected)
        self.jog_plus_button.setEnabled(state.connected)
        self.scan_button.setEnabled(state.connected)
        self.capture_neutral_button.setEnabled(state.connected)
        self.save_neutral_button.setEnabled(has_neutral)
        self.load_neutral_button.setEnabled(True)
        self.pretension_button.setEnabled(state.connected and has_neutral)
        self.apply_displacement_button.setEnabled(state.connected and has_neutral)
        if len(self.displacement_inputs) != len(state.tendon_displacements_cm):
            self._rebuild_displacement_inputs(len(state.tendon_displacements_cm))
        for spin, value in zip(self.displacement_inputs, state.tendon_displacements_cm):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

        self.telemetry_table.setRowCount(len(state.telemetry))
        for row, servo_id in enumerate(sorted(state.telemetry)):
            item = state.telemetry[servo_id]
            self.telemetry_table.setItem(row, 0, QTableWidgetItem(str(servo_id)))
            self.telemetry_table.setItem(row, 1, QTableWidgetItem(str(item["position"])))
            self.telemetry_table.setItem(row, 2, QTableWidgetItem(str(item["current_ma"])))
            self.telemetry_table.setItem(row, 3, QTableWidgetItem(str(item["voltage_mv"])))
            self.telemetry_table.setItem(row, 4, QTableWidgetItem(str(item["error"])))

        lines = [state.status_message]
        if state.last_error:
            lines.append(f"Error: {state.last_error}")
        self.status_text.setPlainText("\n".join(lines))

    def _rebuild_displacement_inputs(self, count: int) -> None:
        for spin in self.displacement_inputs:
            spin.setParent(None)
        self.displacement_inputs = []
        for idx in range(count):
            label = QLabel(f"dl_{idx + 1}")
            spin = QDoubleSpinBox()
            spin.setRange(-25.0, 25.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
            self.displacement_layout.addWidget(label, 0, idx * 2)
            self.displacement_layout.addWidget(spin, 0, idx * 2 + 1)
            self.displacement_inputs.append(spin)

    def _assign_id(self) -> None:
        self._safe_call(
            lambda: self.controller.assign_servo_id(self.assign_from_spin.value(), self.assign_to_spin.value())
        )

    def _jog(self, delta: int) -> None:
        self._safe_call(lambda: self.controller.jog_servo(self.jog_servo_spin.value(), delta))

    def _apply_displacement(self) -> None:
        values = [spin.value() for spin in self.displacement_inputs]
        self.controller.set_tendon_displacements(values)
        self._safe_call(self.controller.apply_displacement)

    def _safe_call(self, fn) -> None:
        try:
            fn()
        except Exception:
            return
