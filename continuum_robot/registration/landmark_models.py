"""Registration landmarks and persistence payload models."""

from dataclasses import dataclass


@dataclass
class LandmarkCapture:
    """One landmark and all repeated captured samples in robot frame."""

    label: str
    nominal_robot_xyz: list[float]
    raw_captured_robot_xyz: list[list[float]]
    averaged_robot_xyz: list[float]
