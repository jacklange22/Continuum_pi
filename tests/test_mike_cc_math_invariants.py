"""Mike constant-curvature math invariants.

These tests pin the analytical correctness of the constant-curvature transform
and the two-segment composition. They do NOT depend on any operator data,
hardware, or sign-convention choice — they verify that:

1. The CC transform with kx=ky=0 is identity with Z=length (the "straight" case).
2. The CC transform with non-zero curvature places the endpoint at the
   analytical (radius * sin/cos) position in the curvature plane.
3. The two-segment composition T_distal = T_bottom @ T_top is consistent
   with applying the top segment in the bottom-tip's rotated frame.
4. Curvature plane angle phi rotates the endpoint in the X-Y plane as expected.

If a future refactor breaks the Webster DH parameterization or the
composition order, these tests catch it immediately.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from continuum_robot.modeling.two_segment.physics import (
    _constant_curvature_transform,
    two_segment_constant_curvature_prediction,
)


def _analytical_cc_endpoint(length_mm: float, kappa: float, phi: float) -> np.ndarray:
    """Closed-form analytical endpoint for a constant-curvature segment.

    Standard textbook constant-curvature parameterisation:
      - The arc lies in the plane at angle phi from +X.
      - r = 1/kappa is the radius of curvature.
      - In-plane displacement = r * (1 - cos(kappa * L)).
      - Axial (Z) displacement = r * sin(kappa * L).
    """
    if abs(kappa) <= 1e-12:
        return np.asarray([0.0, 0.0, float(length_mm)], dtype=float)
    radius = 1.0 / kappa
    in_plane = radius * (1.0 - math.cos(kappa * length_mm))
    z_disp = radius * math.sin(kappa * length_mm)
    return np.asarray(
        [
            in_plane * math.cos(phi),
            in_plane * math.sin(phi),
            z_disp,
        ],
        dtype=float,
    )


def test_constant_curvature_transform_with_zero_curvature_is_pure_z_translation() -> None:
    transform = _constant_curvature_transform(length_mm=66.0, kx=0.0, ky=0.0)

    # Rotation block should be identity.
    np.testing.assert_allclose(transform[:3, :3], np.eye(3), atol=1e-9)
    # Translation should be (0, 0, length).
    np.testing.assert_allclose(transform[:3, 3], [0.0, 0.0, 66.0], atol=1e-9)
    np.testing.assert_allclose(transform[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-12)


@pytest.mark.parametrize(
    "kappa, phi",
    [
        (0.005, 0.0),
        (0.010, 0.0),
        (0.005, math.pi / 2.0),
        (0.005, math.pi),
        (0.005, -math.pi / 4.0),
        (0.020, math.pi / 6.0),
    ],
)
def test_constant_curvature_transform_endpoint_matches_analytical_formula(kappa: float, phi: float) -> None:
    length = 66.0
    kx = kappa * math.cos(phi)
    ky = kappa * math.sin(phi)

    transform = _constant_curvature_transform(length_mm=length, kx=kx, ky=ky)
    endpoint = transform[:3, 3]
    expected = _analytical_cc_endpoint(length, kappa, phi)

    np.testing.assert_allclose(endpoint, expected, atol=1e-6, rtol=1e-6)


def test_constant_curvature_transform_endpoint_lies_on_circle_of_radius_one_over_kappa() -> None:
    """The endpoint should lie on a circle of radius 1/kappa in the curvature plane."""
    length = 66.0
    kappa = 0.015
    phi = math.pi / 3.0
    kx = kappa * math.cos(phi)
    ky = kappa * math.sin(phi)

    transform = _constant_curvature_transform(length_mm=length, kx=kx, ky=ky)
    endpoint = transform[:3, 3]

    # The arc center is offset by 1/kappa along the in-plane direction perpendicular to start tangent.
    # |endpoint - center|^2 = (1/kappa)^2. center = (cos(phi)/kappa, sin(phi)/kappa, 0).
    radius = 1.0 / kappa
    center = np.asarray([radius * math.cos(phi), radius * math.sin(phi), 0.0], dtype=float)
    distance_from_center = float(np.linalg.norm(endpoint - center))
    np.testing.assert_allclose(distance_from_center, radius, atol=1e-6, rtol=1e-6)


def test_constant_curvature_transform_rotation_advances_by_kappa_times_length() -> None:
    """The tangent direction at the endpoint should rotate by (kappa * length) radians around the bend axis."""
    length = 50.0
    kappa = 0.020
    phi = 0.0  # bend in +X plane
    kx = kappa
    ky = 0.0

    transform = _constant_curvature_transform(length_mm=length, kx=kx, ky=ky)
    # Tangent at endpoint = R @ z_hat
    tangent = transform[:3, 2]
    expected_tangent = np.asarray(
        [math.sin(kappa * length), 0.0, math.cos(kappa * length)],
        dtype=float,
    )
    np.testing.assert_allclose(tangent, expected_tangent, atol=1e-6, rtol=1e-6)


def _config_with_symmetric_geometry() -> dict:
    """Both segments identical: 66 mm long, 4.12 mm tendon radius, [+X,+Y,-X,-Y] arrangement."""
    return {
        "physics_models": {
            "global": {
                "tendon_displacement_sign_convention": {"value": "positive_shortens_tendon", "source": "design"},
            },
            "segments": {
                "segment_a": {
                    "segment_length_mm": {"value": 66.0, "units": "mm", "source": "design"},
                    "tendon_positions_mm": {
                        "value": [[4.12, 0.0], [0.0, 4.12], [-4.12, 0.0], [0.0, -4.12]],
                        "units": "mm",
                        "source": "design",
                    },
                },
                "segment_b": {
                    "segment_length_mm": {"value": 66.0, "units": "mm", "source": "design"},
                    "tendon_positions_mm": {
                        "value": [[4.12, 0.0], [0.0, 4.12], [-4.12, 0.0], [0.0, -4.12]],
                        "units": "mm",
                        "source": "design",
                    },
                },
            },
        }
    }


def test_two_segment_zero_command_stacks_to_axial_double_length() -> None:
    """Zero command on both segments should put the distal tip at (0, 0, 2L)."""
    config = _config_with_symmetric_geometry()
    command = np.zeros(8, dtype=float)

    prediction = two_segment_constant_curvature_prediction(command, config)

    np.testing.assert_allclose(prediction["intermediate_xyz"], [0.0, 0.0, 66.0], atol=1e-9)
    np.testing.assert_allclose(prediction["distal_xyz"], [0.0, 0.0, 132.0], atol=1e-9)
    np.testing.assert_allclose(prediction["distal_tangent"], [0.0, 0.0, 1.0], atol=1e-9)


def test_two_segment_top_only_command_starts_from_bottom_endpoint() -> None:
    """If bottom is zero and top bends, the distal endpoint = bottom endpoint + top transform applied in bottom-tip frame.

    Since bottom is straight (R_bottom = I, translation = (0, 0, L_bottom)), the
    distal should be the top segment's standalone endpoint shifted by (0, 0, L_bottom).
    """
    config = _config_with_symmetric_geometry()
    # Apply only top-segment X-tendon shortening; bottom stays zero.
    command = np.zeros(8, dtype=float)
    command[4] = 0.5  # top +X tendon shorten
    command[6] = -0.5  # top -X tendon lengthen (anti-correlated for pure bend)

    prediction = two_segment_constant_curvature_prediction(command, config)

    np.testing.assert_allclose(prediction["intermediate_xyz"], [0.0, 0.0, 66.0], atol=1e-9)
    # Distal should be bottom-tip + top-only-prediction.
    # We can verify by predicting with only the top command interpreted as a single segment.
    top_only_command = command[4:].copy()  # 4-vector
    top_only_command_8 = np.concatenate([top_only_command, np.zeros(4)])
    top_only_prediction = two_segment_constant_curvature_prediction(top_only_command_8, config)
    standalone_top_endpoint = top_only_prediction["intermediate_xyz"]
    expected_distal = np.asarray([standalone_top_endpoint[0], standalone_top_endpoint[1], standalone_top_endpoint[2] + 66.0])
    np.testing.assert_allclose(prediction["distal_xyz"], expected_distal, atol=1e-6, rtol=1e-6)


def test_two_segment_composition_t_distal_equals_t_bottom_times_t_top() -> None:
    """T_distal must equal T_bottom @ T_top by construction."""
    config = _config_with_symmetric_geometry()
    rng = np.random.default_rng(42)
    command = rng.uniform(-0.3, 0.3, size=8)
    # Force the anti-correlated tendon convention: tendon i and i+2 negate each other per segment.
    command[2] = -command[0]
    command[3] = -command[1]
    command[6] = -command[4]
    command[7] = -command[5]

    prediction = two_segment_constant_curvature_prediction(command, config)

    # Reconstruct T_bottom @ T_top from the prediction's matrices.
    composed = prediction["T_intermediate"] @ _solo_segment_transform(command[4:], config, segment_key="segment_b")
    # The prediction already returns T_distal = T_bottom @ T_top.
    np.testing.assert_allclose(prediction["T_distal"], composed, atol=1e-9)


def _solo_segment_transform(command_4: np.ndarray, config: dict, *, segment_key: str) -> np.ndarray:
    """Compute the transform for a single segment using the same internals as the predictor."""
    from continuum_robot.modeling.two_segment.physics import _segment_transform

    segment = config["physics_models"]["segments"][segment_key]
    return _segment_transform(command_4, segment=segment, sign_convention="positive_shortens_tendon")
