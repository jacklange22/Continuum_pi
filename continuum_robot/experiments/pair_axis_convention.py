"""Canonical pair-axis ↔ cable-delta convention for single-segment commands.

This module is the single source of truth for the relationship between a
two-component "pair command" `(px, py)` and the four-tendon cable-delta
vector `[c0, c1, c2, c3]` on the 4-cable single-segment continuum robot.
It exists because two earlier code paths used opposite sign conventions:

* The Vogel-disk target generator in
  ``continuum_robot/experiments/workspace_repeatability_map.py`` produced
  ``cable_deltas = [-x_mm, -y_mm, +x_mm, +y_mm]`` for a tip target at
  ``(+x_mm, +y_mm)`` — i.e., the +X-side cable shortens (negative cable
  delta) when the tip is pulled toward +X. This matches the project
  hardware convention described in ``docs/hardware_day_runbook.md``:
  "5 = +x, 6 = +y, 7 = -x, 8 = -y; lower ticks mean more tension /
  tendon shortening".
* The collect-pose helper ``_expand_pair_command_cm`` in
  ``continuum_robot/experiments/builtins.py`` produced
  ``cable_deltas = [+px, +py, -px, -py]`` for a pair command ``(+px, +py)``
  — i.e., the +X-side cable LENGTHENED when the user asked for a +X
  pair, which made the tip move toward -X.

Both functions were mutual inverses of themselves, so single-path runs
were internally self-consistent. They diverged only when the two files
were compared side-by-side; thesis figures comparing collect-pose tip
commands to workspace-map tip commands appeared mirrored.

The canonical convention recorded here is **tip-target space**: the
pair command ``(px, py)`` is the position the tip should be pulled
toward, in the same units as the cable deltas. The +X-side cable
(cable index 0, physical servo 5 on segment B) shortens by ``px``
to pull the tip toward +X:

    pair (+px, +py)  →  cable_deltas = [-px, -py, +px, +py]
    cable_deltas [c0, c1, c2, c3]  →  pair = (-c0, -c1)

Equivalently, when the antagonistic pair structure is intact
``(c0 == -c2)`` and ``(c1 == -c3)``, the inverse can also be read as
``pair = (+c2, +c3)``.

The version string ``PAIR_AXIS_CONVENTION_VERSION`` is stamped into
every new experiment's metrics so that downstream consumers can detect
runs that pre-date the unification and avoid silently re-interpreting
them. Historical runs that lack the field were saved under the old
mixed convention; treat their stored ``pair_command_cm`` values as
opaque rather than as tip targets.
"""

from __future__ import annotations

from typing import Sequence


PAIR_AXIS_CONVENTION_VERSION = "tip_target_2026_05_20"
"""Stamp this into ``session.metrics['pair_axis_convention']`` for every
new run that uses the canonical helpers below. Absence implies the
legacy mixed convention that this module replaces."""


PAIR_AXIS_CONVENTION_DOC = (
    "pair (px, py) is the tip-target displacement; cable_deltas "
    "[c0, c1, c2, c3] = [-px, -py, +px, +py]; +X-side cable "
    "(index 0) shortens (negative delta) when tip is pulled +X"
)


def expand_pair_command_to_cable_deltas(pair_command: Sequence[float]) -> list[float]:
    """Map a tip-target pair command ``(px, py)`` to a 4-cable delta vector.

    The +X-side cable (index 0) shortens by ``px`` (negative delta) so
    that the tip is pulled toward ``+x`` when ``px > 0``. The -X-side
    cable (index 2) lengthens by the same amount. Y axis is analogous.

    Inputs shorter than 2 are zero-padded; inputs longer than 2 are
    truncated to ``(px, py)``. Returns a list with exactly 4 floats.
    """
    px = float(pair_command[0]) if len(pair_command) > 0 else 0.0
    py = float(pair_command[1]) if len(pair_command) > 1 else 0.0
    return [-px, -py, px, py]


def pair_command_from_cable_deltas(cable_deltas: Sequence[float]) -> list[float]:
    """Inverse of :func:`expand_pair_command_to_cable_deltas`.

    Given cable deltas ``[c0, c1, c2, c3]``, return the tip-target pair
    ``(px, py) = (-c0, -c1)``. When the antagonistic structure is
    intact (``c0 == -c2`` and ``c1 == -c3``), this also equals
    ``(+c2, +c3)`` — both forms are equivalent.

    For shorter inputs the missing components are treated as zero.
    """
    if len(cable_deltas) == 0:
        return [0.0, 0.0]
    px = -float(cable_deltas[0])
    py = -float(cable_deltas[1]) if len(cable_deltas) > 1 else 0.0
    return [px, py]
