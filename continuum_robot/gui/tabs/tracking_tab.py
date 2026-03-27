"""Tracking tab state holder.

PySide widgets are intentionally not hardcoded here; this class can be adapted by
the real GUI layer while preserving controller contract.
"""

from continuum_robot.gui.controllers.tracking_controller import TrackingViewState


class TrackingTab:
    """Live Aurora tool and tip pose UI."""

    def __init__(self) -> None:
        self.last_state = TrackingViewState(device_path="", socket_path="")

    def update(self, state: TrackingViewState) -> None:
        self.last_state = state
