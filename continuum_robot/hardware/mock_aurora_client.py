"""Mock Aurora client for GUI and pipeline testing without hardware."""

from continuum_robot.hardware.aurora_client import AuroraClient


class MockAuroraClient(AuroraClient):
    """Aurora mock that yields no-op data by default."""
