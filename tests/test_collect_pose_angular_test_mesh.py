"""Tests for the Wolfe-style angular test-mesh dataset_mode.

Grounded in Wolfe MS thesis §3.2.3 p85 — model comparison uses a θ × φ angular
grid evaluated at fixed segment length, with multiple samples per cell. This
module verifies that the new ``dataset_mode = 'angular_test_mesh'`` schedule
generator emits the right number of cells, in the right order, with pair
commands tracing constant-curvature targets bounded by the workspace amplitude.
"""

from __future__ import annotations

import math

import pytest

from continuum_robot.experiments.builtins import (
    CollectPoseCommandDatasetConfig,
    _build_collect_pose_command_steps,
)


def _bounds(amplitude: float = 0.5) -> dict:
    return {"pair_bounds_cm": [(-amplitude, amplitude), (-amplitude, amplitude)]}


def test_angular_test_mesh_step_count_matches_theta_x_phi() -> None:
    cfg = CollectPoseCommandDatasetConfig(
        dataset_mode="angular_test_mesh",
        test_mesh_theta_count=4,
        test_mesh_phi_count=6,
        workspace_amplitude_cm=0.5,
    )
    steps = _build_collect_pose_command_steps(config=cfg, pair_limits=_bounds())
    assert len(steps) == 24, "4 * 6 grid"
    # Default Wolfe grid emits 12 * 24 = 288 cells.
    cfg_wolfe = CollectPoseCommandDatasetConfig(
        dataset_mode="angular_test_mesh", workspace_amplitude_cm=0.5
    )
    wolfe_steps = _build_collect_pose_command_steps(config=cfg_wolfe, pair_limits=_bounds())
    assert len(wolfe_steps) == 288


def test_angular_test_mesh_commands_stay_within_amplitude_bounds() -> None:
    """Per-pair magnitude must never exceed the workspace-amplitude bound."""
    amplitude = 0.4
    cfg = CollectPoseCommandDatasetConfig(
        dataset_mode="angular_test_mesh",
        test_mesh_theta_count=12,
        test_mesh_phi_count=24,
        workspace_amplitude_cm=amplitude,
    )
    steps = _build_collect_pose_command_steps(config=cfg, pair_limits=_bounds(amplitude))
    for step in steps:
        pair_x, pair_y = step.pair_command_cm
        assert abs(pair_x) <= amplitude + 1e-9
        assert abs(pair_y) <= amplitude + 1e-9
        # Radial magnitude bound (sin(θ) ≤ 1, so radial ≤ amplitude).
        assert math.hypot(pair_x, pair_y) <= amplitude + 1e-9


def test_angular_test_mesh_phase_and_metadata() -> None:
    """Each cell carries Wolfe-grade provenance metadata for later analysis."""
    cfg = CollectPoseCommandDatasetConfig(
        dataset_mode="angular_test_mesh",
        test_mesh_theta_count=3,
        test_mesh_phi_count=4,
        workspace_amplitude_cm=0.5,
    )
    steps = _build_collect_pose_command_steps(config=cfg, pair_limits=_bounds())
    for step in steps:
        assert step.phase == "angular_test_mesh"
        assert step.metadata["mode_family"] == "angular_test_mesh"
        assert "theta_rad" in step.metadata
        assert "phi_rad" in step.metadata
        assert isinstance(step.metadata["theta_index"], int)
        assert isinstance(step.metadata["phi_index"], int)


def test_angular_test_mesh_amplitude_override_takes_precedence() -> None:
    """``test_mesh_amplitude_cm`` overrides ``workspace_amplitude_cm`` if set."""
    cfg = CollectPoseCommandDatasetConfig(
        dataset_mode="angular_test_mesh",
        test_mesh_theta_count=2,
        test_mesh_phi_count=4,
        workspace_amplitude_cm=1.0,
        test_mesh_amplitude_cm=0.2,
    )
    steps = _build_collect_pose_command_steps(
        config=cfg,
        pair_limits={"pair_bounds_cm": [(-1.0, 1.0), (-1.0, 1.0)]},
    )
    for step in steps:
        pair_x, pair_y = step.pair_command_cm
        assert math.hypot(pair_x, pair_y) <= 0.2 + 1e-9
        assert math.hypot(pair_x, pair_y) < 0.5  # well under workspace_amplitude


def test_angular_test_mesh_first_cell_is_low_amplitude() -> None:
    """θ_1 = π/2 · 1/N produces the smallest commanded radius; sanity-check it's small."""
    cfg = CollectPoseCommandDatasetConfig(
        dataset_mode="angular_test_mesh",
        test_mesh_theta_count=12,
        test_mesh_phi_count=24,
        workspace_amplitude_cm=1.0,
    )
    steps = _build_collect_pose_command_steps(
        config=cfg,
        pair_limits={"pair_bounds_cm": [(-1.0, 1.0), (-1.0, 1.0)]},
    )
    first = steps[0]
    last = steps[-1]
    first_radius = math.hypot(*first.pair_command_cm)
    last_radius = math.hypot(*last.pair_command_cm)
    # First ring should be much smaller than last ring (12 θ steps).
    assert first_radius < last_radius
    assert first_radius < 0.2
    # Last θ ring sits at θ = π/2 → radial = sin(π/2) * amplitude = amplitude.
    assert last_radius == pytest.approx(1.0, abs=1e-9)


def test_angular_test_mesh_phi_traversal_in_order() -> None:
    """Within one θ ring, φ should sweep 0 → 2π."""
    theta_count = 2
    phi_count = 8
    cfg = CollectPoseCommandDatasetConfig(
        dataset_mode="angular_test_mesh",
        test_mesh_theta_count=theta_count,
        test_mesh_phi_count=phi_count,
        workspace_amplitude_cm=0.5,
    )
    steps = _build_collect_pose_command_steps(config=cfg, pair_limits=_bounds())
    # First phi_count cells belong to θ_1 ring; φ must be monotonically increasing.
    first_ring = steps[:phi_count]
    phis = [step.metadata["phi_rad"] for step in first_ring]
    assert all(phis[i] < phis[i + 1] for i in range(len(phis) - 1))
    # And every ring has phi_count cells.
    assert len(steps) == theta_count * phi_count
