"""Registration tab state holder."""

from continuum_robot.gui.controllers.registration_controller import RegistrationViewState


class RegistrationTab:
    """Guided landmark capture and registration UI."""

    def __init__(self) -> None:
        self.last_state = RegistrationViewState()

    def update(self, state: RegistrationViewState) -> None:
        self.last_state = state
