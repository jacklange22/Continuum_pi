from pathlib import Path

import pytest

from continuum_robot.app.bootstrap import build_app_context
from continuum_robot.hardware.dxl_bus import DxlBus
from continuum_robot.hardware.openrb_client import OpenRbClient


def test_real_dxl_bus_refuses_fake_connection() -> None:
    bus = DxlBus()
    with pytest.raises(RuntimeError, match="not implemented"):
        bus.connect("/dev/ttyUSB0", 115200)


def test_real_openrb_client_refuses_fake_connection() -> None:
    client = OpenRbClient()
    with pytest.raises(RuntimeError, match="not implemented"):
        client.connect("/dev/ttyUSB1", 115200)


def test_build_app_context_uses_absolute_registration_path() -> None:
    context = build_app_context()
    registration_path = context.project_root / context.settings.calibration.latest_registration_path
    assert registration_path.is_absolute()
