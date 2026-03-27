from pathlib import Path

import numpy as np

from continuum_robot.registration.legacy_compat import (
    AuroraPoseSample,
    RegistrationAssetPaths,
    RegistrationAssets,
    expand_points_by_repetition,
    load_registration_assets,
    parse_aurora_csv,
    solve_registration_from_tool_samples,
)
from continuum_robot.registration.live_registration_service import LiveRegistrationService
from continuum_robot.registration.repository import RegistrationRepository
from continuum_robot.registration.rigid_solver import RigidRegistrationSolver
from tests.test_live_registration_service import _FakeTrackerManager


def test_expand_points_by_repetition_matches_legacy_order() -> None:
    points = np.array(
        [
            [1.0, 2.0],
            [10.0, 20.0],
            [100.0, 200.0],
        ]
    )

    expanded = expand_points_by_repetition(points, 3)

    expected = np.array(
        [
            [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            [10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
            [100.0, 100.0, 100.0, 200.0, 200.0, 200.0],
        ]
    )
    assert np.allclose(expanded, expected)


def test_load_registration_assets_reads_protected_tool_files() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assets = load_registration_assets(
        RegistrationAssetPaths(
            model_points_file=repo_root / "tools" / "12_model_registration_points_in_sw",
            tip_points_file=repo_root / "tools" / "all_tip_registration_points_in_sw",
            T_sw_2_model_file=repo_root / "tools" / "T_sw_2_model",
            T_sw_2_tip_file=repo_root / "tools" / "T_sw_2_tip",
            penprobe_file=repo_root / "tools" / "penprobe_08_09_24c",
        )
    )

    assert assets.model_truth_in_sw.shape == (3, 12)
    assert assets.tip_truth_in_sw.shape == (3, 12)
    assert assets.T_sw_2_model.shape == (4, 4)
    assert assets.T_sw_2_tip.shape == (4, 4)
    assert assets.T_measurement_point.shape == (4, 4)
    assert len(assets.model_labels) == 12
    assert len(assets.tip_labels) == 12


def test_parse_aurora_csv_parses_legacy_registration_points_file() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parsed = parse_aurora_csv(repo_root / "references" / "RegistrationPoints.csv")

    assert set(parsed.keys()) == {"0A", "0B"}
    assert len(parsed["0A"]) == 120
    assert len(parsed["0B"]) == 120
    assert parsed["0A"][0].quaternion_wxyz[0] == parsed["0A"][0].quaternion_wxyz[0]
    assert parsed["0A"][0].translation_mm[0] == parsed["0A"][0].translation_mm[0]


def test_solve_registration_from_tool_samples_recovers_known_outputs() -> None:
    model_truth_in_sw = np.array(
        [
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 0.0],
        ]
    )
    tip_truth_in_sw = np.array(
        [
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
            [2.0, 2.0, 2.0],
        ]
    )
    assets = RegistrationAssets(
        paths=RegistrationAssetPaths(
            model_points_file=Path("model"),
            tip_points_file=Path("tip"),
            T_sw_2_model_file=Path("T_sw_2_model"),
            T_sw_2_tip_file=Path("T_sw_2_tip"),
            penprobe_file=Path("penprobe"),
        ),
        model_truth_in_sw=model_truth_in_sw,
        tip_truth_in_sw=tip_truth_in_sw,
        T_sw_2_model=np.eye(4),
        T_sw_2_tip=np.eye(4),
        T_measurement_point=np.eye(4),
        model_labels=["M01", "M02", "M03"],
        tip_labels=["T01", "T02", "T03"],
    )

    T_aurora_2_model = np.eye(4)
    T_aurora_2_model[0:3, 3] = np.array([10.0, -2.0, 5.0])
    T_aurora_2_tip = np.eye(4)
    T_aurora_2_tip[0:3, 3] = np.array([-3.0, 4.0, 6.0])
    T_coil_2_aurora = np.eye(4)
    T_coil_2_aurora[0:3, 3] = np.array([1.0, 2.0, 3.0])
    expected_T_tip_2_coil = np.linalg.inv(T_coil_2_aurora) @ np.linalg.inv(T_aurora_2_tip)

    repetitions = 2
    model_truth_expanded = expand_points_by_repetition(model_truth_in_sw, repetitions)
    tip_truth_expanded = expand_points_by_repetition(tip_truth_in_sw, repetitions)

    model_meas = np.linalg.inv(T_aurora_2_model)
    model_points_in_aurora = (model_meas[0:3, 0:3] @ model_truth_expanded).T + model_meas[0:3, 3]
    tip_meas = np.linalg.inv(T_aurora_2_tip)
    tip_points_in_aurora = (tip_meas[0:3, 0:3] @ tip_truth_expanded).T + tip_meas[0:3, 3]

    measurement_tool_samples = [
        AuroraPoseSample("0B", (1.0, 0.0, 0.0, 0.0), tuple(point), 0.1)
        for point in np.vstack([model_points_in_aurora, tip_points_in_aurora])
    ]
    coil_tool_samples = [
        AuroraPoseSample("0A", (1.0, 0.0, 0.0, 0.0), tuple(T_coil_2_aurora[0:3, 3]), 0.1)
        for _ in range(len(measurement_tool_samples))
    ]

    result = solve_registration_from_tool_samples(
        assets=assets,
        measurement_tool_samples=measurement_tool_samples,
        coil_tool_samples=coil_tool_samples,
        repetitions=repetitions,
        measurement_tool_id="0B",
        coil_tool_id="0A",
        solver=RigidRegistrationSolver(),
    )

    assert np.allclose(result.T_aurora_2_model, T_aurora_2_model, atol=1e-9)
    assert np.allclose(result.T_aurora_2_tip, T_aurora_2_tip, atol=1e-9)
    assert np.allclose(result.T_tip_2_coil, expected_T_tip_2_coil, atol=1e-9)


def test_solve_registration_from_tool_samples_rejects_count_mismatch() -> None:
    assets = RegistrationAssets(
        paths=RegistrationAssetPaths(
            model_points_file=Path("model"),
            tip_points_file=Path("tip"),
            T_sw_2_model_file=Path("T_sw_2_model"),
            T_sw_2_tip_file=Path("T_sw_2_tip"),
            penprobe_file=Path("penprobe"),
        ),
        model_truth_in_sw=np.zeros((3, 1)),
        tip_truth_in_sw=np.zeros((3, 1)),
        T_sw_2_model=np.eye(4),
        T_sw_2_tip=np.eye(4),
        T_measurement_point=np.eye(4),
        model_labels=["M01"],
        tip_labels=["T01"],
    )

    measurement_tool_samples = [AuroraPoseSample("0B", (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.1)]
    coil_tool_samples = [AuroraPoseSample("0A", (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.1)]

    try:
        solve_registration_from_tool_samples(
            assets=assets,
            measurement_tool_samples=measurement_tool_samples,
            coil_tool_samples=coil_tool_samples,
            repetitions=2,
            measurement_tool_id="0B",
            coil_tool_id="0A",
            solver=RigidRegistrationSolver(),
        )
    except ValueError as exc:
        assert "sample count" in str(exc)
    else:
        raise AssertionError("Expected count mismatch to raise ValueError")


def test_live_registration_service_accepts_legacy_csv_and_protected_assets(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    service = LiveRegistrationService(
        tracker_manager=_FakeTrackerManager([]),
        repository=RegistrationRepository(root_dir=tmp_path),
        solver=RigidRegistrationSolver(),
        asset_paths=RegistrationAssetPaths(
            model_points_file=repo_root / "tools" / "12_model_registration_points_in_sw",
            tip_points_file=repo_root / "tools" / "all_tip_registration_points_in_sw",
            T_sw_2_model_file=repo_root / "tools" / "T_sw_2_model",
            T_sw_2_tip_file=repo_root / "tools" / "T_sw_2_tip",
            penprobe_file=repo_root / "tools" / "penprobe_08_09_24c",
        ),
        measurement_tool_id="0A",
        coil_tool_id="0B",
    )

    result = service.complete_registration_from_csv(
        repo_root / "references" / "RegistrationPoints.csv",
        config_used={"test": True},
    )

    assert result.output_path.exists()
    assert result.record.validation_metrics["overall_fre_mm"] >= 0.0
    assert result.record.measurement_tool_id == "0A"
    assert result.record.coil_tool_id == "0B"
