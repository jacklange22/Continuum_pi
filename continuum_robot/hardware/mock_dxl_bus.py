"""Mock DYNAMIXEL bus for non-hardware test and GUI development."""

from continuum_robot.hardware.dxl_bus import DxlBus


class MockDxlBus(DxlBus):
    """Mock implementation of low-level bus operations."""
