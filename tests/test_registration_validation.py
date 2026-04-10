import pytest

from continuum_robot.registration.validation import (
    build_registration_history_summary,
    compute_fre_mm,
    compute_geometry_diagnostics,
)


def test_compute_fre_mm_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        compute_fre_mm([])


def test_compute_fre_mm_returns_rms_norm() -> None:
    out = compute_fre_mm([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    assert out == (25.0 / 2.0) ** 0.5


def test_compute_geometry_diagnostics_reports_rank_spacing_and_conditioning() -> None:
    diagnostics = compute_geometry_diagnostics(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [0.0, 0.0, 30.0],
        ]
    )

    assert diagnostics["point_count"] == 4
    assert diagnostics["geometry_rank"] == 3
    assert diagnostics["min_pairwise_distance_mm"] == pytest.approx(10.0)
    assert diagnostics["condition_number"] is not None


def test_build_registration_history_summary_summarizes_recent_runs() -> None:
    current = {
        "timestamp_utc": "2026-04-09T12:00:00Z",
        "T_robot_aurora": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "validation_metrics": {
            "overall_fre_mm": 0.2,
            "residual_norms_mm_by_label": {
                "L1": 0.1,
                "L2": 0.3,
            },
        },
    }
    previous = {
        "timestamp_utc": "2026-04-09T11:45:00Z",
        "T_robot_aurora": [
            [1.0, 0.0, 0.0, -0.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "validation_metrics": {
            "overall_fre_mm": 0.4,
            "residual_norms_mm_by_label": {
                "L1": 0.2,
                "L2": 0.5,
            },
        },
    }

    summary = build_registration_history_summary(
        current_payload=current,
        current_path="/tmp/current_registration.json",
        previous_runs=[("/tmp/previous_registration.json", previous)],
    )

    assert summary["history_run_count"] == 2
    assert summary["comparison_count"] == 1
    assert summary["fre_summary_mm"]["max"] == pytest.approx(0.4)
    assert summary["translation_delta_summary_mm"]["max"] == pytest.approx(0.5)
    assert summary["rotation_delta_summary_deg"]["max"] == pytest.approx(0.0)
    assert summary["worst_landmark_by_mean_residual"] == "L2"
    assert summary["per_landmark_residual_summary_mm_by_label"]["L2"]["mean_mm"] == pytest.approx(0.4)
