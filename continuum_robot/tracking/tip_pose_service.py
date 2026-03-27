"""Compute robot tip pose from calibrated transform chain."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from continuum_robot.tracking.tool_models import AuroraToolMeasurement
from continuum_robot.tracking.transforms import (
    assert_transform_matrix,
    compose_T_A_C,
    make_transform_A_B,
)


@dataclass
class TipPoseInputs:
    """Runtime transforms needed for tip pose computation."""

    T_robot_aurora: np.ndarray
    T_coil_tip: np.ndarray


class TipPoseService:
    """Provides strict-frame tip pose computation and config loading."""

    def __init__(self, inputs: TipPoseInputs) -> None:
        assert_transform_matrix(inputs.T_robot_aurora, "T_robot_aurora")
        assert_transform_matrix(inputs.T_coil_tip, "T_coil_tip")
        self.inputs = inputs

    @classmethod
    def from_registration_file(cls, registration_path: Path) -> "TipPoseService":
        """Load transforms from current or legacy registration JSON schemas."""
        if not registration_path.exists():
            raise FileNotFoundError(
                f"Missing registration file: {registration_path}. "
                "Run registration first to generate latest_registration.json."
            )

        payload = json.loads(registration_path.read_text(encoding="utf-8"))
        T_robot_aurora = cls._load_T_robot_aurora(payload, registration_path)
        T_coil_tip = cls._load_T_coil_tip(payload, registration_path)

        return cls(TipPoseInputs(T_robot_aurora=T_robot_aurora, T_coil_tip=T_coil_tip))

    @staticmethod
    def _load_T_robot_aurora(payload: dict, registration_path: Path) -> np.ndarray:
        if "T_robot_aurora" in payload:
            return np.asarray(payload["T_robot_aurora"], dtype=float)
        if "T_aurora_2_model" in payload:
            # Legacy naming maps to the same direction under T_A_B convention.
            return np.asarray(payload["T_aurora_2_model"], dtype=float)
        raise KeyError(
            f"Registration file {registration_path} is missing T_robot_aurora/T_aurora_2_model"
        )

    @staticmethod
    def _load_T_coil_tip(payload: dict, registration_path: Path) -> np.ndarray:
        if "T_coil_tip" in payload:
            return np.asarray(payload["T_coil_tip"], dtype=float)
        if "T_tip_2_coil" in payload:
            T_tip_coil = np.asarray(payload["T_tip_2_coil"], dtype=float)
            return np.linalg.inv(T_tip_coil)
        raise KeyError(
            f"Registration file {registration_path} is missing T_coil_tip/T_tip_2_coil"
        )

    @staticmethod
    def compute_T_robot_tip(
        T_robot_aurora: np.ndarray,
        T_aurora_coil: np.ndarray,
        T_coil_tip: np.ndarray,
    ) -> np.ndarray:
        """Compute ``T_robot_tip = T_robot_aurora @ T_aurora_coil @ T_coil_tip``."""
        assert_transform_matrix(T_robot_aurora, "T_robot_aurora")
        assert_transform_matrix(T_aurora_coil, "T_aurora_coil")
        assert_transform_matrix(T_coil_tip, "T_coil_tip")
        return compose_T_A_C(compose_T_A_C(T_robot_aurora, T_aurora_coil), T_coil_tip)

    def compute_T_robot_tip_from_0A(self, tool_0A: AuroraToolMeasurement) -> np.ndarray:
        """Compute tip pose in robot frame from Aurora tool ``0A`` sample."""
        if tool_0A.tool_id != "0A":
            raise ValueError("Expected tool 0A measurement")
        if not tool_0A.valid:
            raise ValueError(f"Tool 0A is invalid: {tool_0A.status_text}")

        T_aurora_coil = make_transform_A_B(tool_0A.quat_wxyz, tool_0A.translation_xyz)
        return self.compute_T_robot_tip(
            T_robot_aurora=self.inputs.T_robot_aurora,
            T_aurora_coil=T_aurora_coil,
            T_coil_tip=self.inputs.T_coil_tip,
        )
