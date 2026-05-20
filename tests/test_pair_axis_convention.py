"""Cross-path sign-convention tests for single-segment pair commands.

Before the 2026-05-20 unification, two code paths disagreed about which
direction the tip moves for a positive pair command. These tests pin the
canonical tip-target convention and verify both call sites agree.
"""

from __future__ import annotations

import math

import pytest

from continuum_robot.experiments.builtins import (
    _expand_pair_command_cm,
    _pair_command_from_cable_deltas,
)
from continuum_robot.experiments.pair_axis_convention import (
    PAIR_AXIS_CONVENTION_DOC,
    PAIR_AXIS_CONVENTION_VERSION,
    expand_pair_command_to_cable_deltas,
    pair_command_from_cable_deltas,
)
from continuum_robot.experiments.workspace_repeatability_map import (
    build_workspace_repeatability_targets,
)


class TestCanonicalConvention:
    """The canonical helper itself: positive X pair command shortens the +X-side cable."""

    def test_positive_x_pair_shortens_plus_x_cable(self) -> None:
        cable_deltas = expand_pair_command_to_cable_deltas([1.0, 0.0])
        # +X-side cable (index 0) must SHORTEN (negative delta)
        assert cable_deltas[0] == -1.0
        # -X-side cable (index 2) must LENGTHEN (positive delta)
        assert cable_deltas[2] == 1.0
        # Y axis untouched
        assert cable_deltas[1] == 0.0
        assert cable_deltas[3] == 0.0

    def test_positive_y_pair_shortens_plus_y_cable(self) -> None:
        cable_deltas = expand_pair_command_to_cable_deltas([0.0, 1.0])
        assert cable_deltas[1] == -1.0
        assert cable_deltas[3] == 1.0
        assert cable_deltas[0] == 0.0
        assert cable_deltas[2] == 0.0

    def test_inverse_round_trip(self) -> None:
        for pair in [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (-0.5, 0.7), (2.3, -1.1)]:
            cable_deltas = expand_pair_command_to_cable_deltas(list(pair))
            recovered = pair_command_from_cable_deltas(cable_deltas)
            assert recovered[0] == pytest.approx(pair[0])
            assert recovered[1] == pytest.approx(pair[1])

    def test_zero_input_produces_zero_cable_deltas(self) -> None:
        assert expand_pair_command_to_cable_deltas([0.0, 0.0]) == [0.0, 0.0, 0.0, 0.0]

    def test_short_input_zero_pads(self) -> None:
        assert expand_pair_command_to_cable_deltas([1.0]) == [-1.0, 0.0, 1.0, 0.0]
        assert expand_pair_command_to_cable_deltas([]) == [0.0, 0.0, 0.0, 0.0]

    def test_convention_version_string_is_stable(self) -> None:
        # Pin the version so downstream tooling (audit script, evidence
        # index, etc.) can identify pre-unification runs by absence of the
        # field. Changing this string must be a deliberate breaking change.
        assert PAIR_AXIS_CONVENTION_VERSION == "tip_target_2026_05_20"
        assert "tip-target" in PAIR_AXIS_CONVENTION_DOC


class TestBuiltinsThinWrappersUseCanonicalConvention:
    """The collect-pose helpers in builtins.py must now produce the same
    cable signs as the workspace_repeatability_map target generator."""

    def test_builtins_expand_uses_canonical_signs(self) -> None:
        # +X pair → +X-side cable shortens. Before the 2026-05-20 fix
        # builtins.py's helper produced [+1, 0, -1, 0] (the OPPOSITE).
        assert _expand_pair_command_cm([1.0, 0.0]) == [-1.0, 0.0, 1.0, 0.0]
        assert _expand_pair_command_cm([0.0, 1.0]) == [0.0, -1.0, 0.0, 1.0]

    def test_builtins_inverse_is_negation(self) -> None:
        # The inverse maps cable_deltas[0] to -cable_deltas[0] under the
        # canonical convention. Pre-2026-05-20 it returned cable_deltas[0]
        # directly, which was the inverse of the OPPOSITE expansion.
        assert _pair_command_from_cable_deltas([-1.0, 0.0, 1.0, 0.0]) == [1.0, 0.0]
        assert _pair_command_from_cable_deltas([0.0, -1.0, 0.0, 1.0]) == [0.0, 1.0]


class TestWorkspaceMapAgreesWithCollectPose:
    """The two code paths must produce identical cable signs for the
    same intended tip-target position. This was the bug the audit
    flagged: same `(x_mm, y_mm)` → opposite cable signs."""

    def test_plus_x_target_matches_across_paths(self) -> None:
        # Workspace-map target generator at (+5 mm, 0)
        targets = build_workspace_repeatability_targets(target_count=8, max_amplitude_mm=5.0)
        # Find the target at +X (largest x_mm)
        plus_x = max((t for t in targets if abs(t.x_mm) > 0.1), key=lambda t: t.x_mm)
        assert plus_x.x_mm > 0
        # Cable delta from the workspace-map path:
        wsm_cable_mm = list(plus_x.cable_deltas_mm)
        # Cable delta from the collect-pose path with the same tip-target:
        cp_cable_mm = expand_pair_command_to_cable_deltas([plus_x.x_mm, plus_x.y_mm])
        # Both paths must agree on every cable sign:
        for i, (wsm, cp) in enumerate(zip(wsm_cable_mm, cp_cable_mm)):
            assert wsm == pytest.approx(cp), (
                f"cable[{i}] disagrees between workspace_map={wsm} and collect_pose={cp}"
            )

    def test_plus_y_target_matches_across_paths(self) -> None:
        targets = build_workspace_repeatability_targets(target_count=16, max_amplitude_mm=5.0)
        # Find a near +Y target
        plus_y = max(targets, key=lambda t: t.y_mm)
        assert plus_y.y_mm > 0
        wsm_cable_mm = list(plus_y.cable_deltas_mm)
        cp_cable_mm = expand_pair_command_to_cable_deltas([plus_y.x_mm, plus_y.y_mm])
        for i, (wsm, cp) in enumerate(zip(wsm_cable_mm, cp_cable_mm)):
            assert wsm == pytest.approx(cp, abs=1e-9), (
                f"cable[{i}] disagrees between workspace_map={wsm} and collect_pose={cp}"
            )

    def test_workspace_map_plus_x_shortens_plus_x_cable(self) -> None:
        # Sanity: the workspace_map target generator itself, post-unification,
        # produces the canonical sign for +X tip target.
        targets = build_workspace_repeatability_targets(target_count=8, max_amplitude_mm=5.0)
        plus_x = max(targets, key=lambda t: t.x_mm)
        assert plus_x.cable_deltas_mm[0] < 0  # +X-side cable SHORTENS
        assert plus_x.cable_deltas_mm[2] > 0  # -X-side cable LENGTHENS
