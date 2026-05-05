from pathlib import Path

from continuum_robot.config.config_loader import ConfigLoader
from continuum_robot.config.schemas import RobotConfig


def test_robot_config_defaults_to_segment_a_pair_mapping() -> None:
    robot = RobotConfig()

    assert robot.active_segment_key() == "segment_a"
    assert robot.active_segment_servo_ids() == [1, 2, 3, 4]
    assert robot.active_segment_pairs() == {"axis_a": [1, 3], "axis_b": [2, 4]}


def test_config_loader_defaults_to_full_platform_profile_when_available(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "system.yaml").write_text("mock_mode: true\n", encoding="utf-8")
    (config_dir / "robot_8servo.yaml").write_text(
        "\n".join(
            [
                "mode: single_segment",
                "servo_ids: [1, 2, 3, 4, 5, 6, 7, 8]",
                "tendon_to_servo: [1, 2, 3, 4, 5, 6, 7, 8]",
                "segments:",
                "  segment_a: {label: Spine 1, servo_ids: [1, 2, 3, 4], pairs: {axis_a: [1, 3], axis_b: [2, 4]}}",
                "  segment_b: {label: Spine 2, servo_ids: [5, 6, 7, 8], pairs: {axis_a: [5, 7], axis_b: [6, 8]}}",
            ]
        ),
        encoding="utf-8",
    )

    settings = ConfigLoader(base_dir=config_dir).load_settings()

    assert settings.runtime.robot_config == "robot_8servo.yaml"
    assert settings.robot.all_segment_servo_ids() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert sorted(settings.robot.segment_map()) == ["segment_a", "segment_b"]


def test_config_loader_reads_runtime_registration_and_experiment_settings(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "system.yaml").write_text(
        "\n".join(
            [
                'robot_config: "robot_8servo.yaml"',
                'aurora_port: "/dev/ttyUSB0"',
                'openrb_port: "/dev/ttyUSB1"',
                'tracker_backend: "ndi"',
                'tracker_type: "aurora"',
                "mock_mode: true",
                "poll_rate_hz: 7",
                "figure_output_quality: medium",
                "tracker_freshness_timeout_s: 0.75",
                "tracker_ports_to_probe: [0A, 0B]",
                "tracker_tool_id_aliases: {PORT1: 0A, PORT2: 0B}",
                "tracker_min_effective_fps: 18.0",
                "openrb_settings: {connect_timeout_s: 0.25}",
                "dynamixel_settings: {protocol_version: 2.0, control_table: {present_position: 132}}",
                'latest_registration_path: "data/registrations/latest_registration.json"',
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "robot_8servo.yaml").write_text(
        "\n".join(
            [
                'mode: "8-servo"',
                "spool_diameter_cm: 1.3",
                "ticks_per_revolution: 4096",
                "servo_ids: [1, 2, 3, 4, 5, 6, 7, 8]",
                "tendon_to_servo: [1, 2, 3, 4, 5, 6, 7, 8]",
                "active_segment: segment_b",
                "segments:",
                "  segment_a:",
                "    label: Spine 1",
                "    servo_ids: [1, 2, 3, 4]",
                "    pairs: {axis_a: [1, 3], axis_b: [2, 4]}",
                "  segment_b:",
                "    label: Spine 2",
                "    servo_ids: [5, 6, 7, 8]",
                "    pairs: {axis_a: [5, 7], axis_b: [6, 8]}",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "safety.yaml").write_text(
        "\n".join(
            [
                "position_min_offset_ticks: -500",
                "position_max_offset_ticks: 500",
                "max_current_ma: 900",
                "pretension_current_balance_tolerance_ma: 75",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "registration.yaml").write_text(
        "\n".join(
            [
                "landmark_labels: [A, C, D, B]",
                "captures_per_landmark: 3",
                'capture_tool_id: "0B"',
                "candidate_landmarks:",
                "  - id: A",
                "    display_label: Front Left",
                "    xyz_mm: [0.0, 0.0, 0.0]",
                "    enabled: true",
                "  - id: B",
                "    xyz_mm: [10.0, 0.0, 0.0]",
                "    enabled: false",
                "  - id: C",
                "    xyz_mm: [0.0, 10.0, 0.0]",
                "    enabled: true",
                "  - id: D",
                "    xyz_mm: [10.0, 10.0, 5.0]",
                "    enabled: true",
                "validation:",
                "  max_fre_mm: 1.5",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "experiment.yaml").write_text(
        "\n".join(
            [
                "default_settle_time_s: 1.5",
                "sample_count_per_point: 2",
                'output_dir: "data/custom_runs"',
            ]
        ),
        encoding="utf-8",
    )

    settings = ConfigLoader(base_dir=config_dir).load_settings()

    assert settings.runtime.mock_mode is True
    assert settings.runtime.figure_output_quality == "medium"
    assert settings.runtime.robot_config == "robot_8servo.yaml"
    assert settings.robot.mode == "8-servo"
    assert settings.robot.servo_ids == [1, 2, 3, 4, 5, 6, 7, 8]
    assert settings.robot.active_segment_key() == "segment_b"
    assert settings.robot.active_segment_label() == "Spine 2"
    assert settings.robot.active_segment_servo_ids() == [5, 6, 7, 8]
    assert settings.robot.active_segment_pairs() == {"axis_a": [5, 7], "axis_b": [6, 8]}
    assert settings.serial.tracker_backend == "ndi"
    assert settings.serial.tracker_type == "aurora"
    assert settings.serial.tracker_freshness_timeout_s == 0.75
    assert settings.serial.tracker_ports_to_probe == ["0A", "0B"]
    assert settings.serial.tracker_tool_id_aliases == {"PORT1": "0A", "PORT2": "0B"}
    assert settings.serial.tracker_min_effective_fps == 18.0
    assert settings.serial.openrb_settings == {"connect_timeout_s": 0.25}
    assert settings.serial.dynamixel_settings["protocol_version"] == 2.0
    assert settings.serial.dynamixel_settings["control_table"]["present_position"] == 132
    assert settings.safety.pretension_current_balance_tolerance_ma == 75
    assert settings.registration.capture_tool_id == "0B"
    assert settings.registration.coil_tool_id == "0A"
    assert settings.registration.captures_per_landmark == 3
    assert settings.registration.landmark_labels == ["A", "C", "D", "B"]
    assert settings.registration.nominal_landmarks_robot_xyz_mm["D"] == [10.0, 10.0, 5.0]
    assert [landmark.id for landmark in settings.registration.candidate_landmarks] == ["A", "B", "C", "D"]
    assert settings.registration.candidate_landmarks[0].display_label == "Front Left"
    assert settings.registration.candidate_landmarks[1].enabled is False
    assert settings.registration.model_points_file is None
    assert settings.experiment.output_dir == "data/custom_runs"
    assert settings.calibration.latest_registration_path == "data/registrations/latest_registration.json"


def test_robot_overrides_do_not_stale_servo_membership(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "system.yaml").write_text(
        "\n".join(
            [
                'robot_config: "robot_8servo.yaml"',
                "robot_overrides:",
                "  mode: dual_segment",
                "  servo_ids: [1]",
                "  tendon_to_servo: [1]",
                "  active_segment: segment_b",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "robot_8servo.yaml").write_text(
        "\n".join(
            [
                "mode: single_segment",
                "servo_ids: [1, 2, 3, 4, 5, 6, 7, 8]",
                "tendon_to_servo: [1, 2, 3, 4, 5, 6, 7, 8]",
                "segments:",
                "  segment_a: {label: Spine 1, servo_ids: [1, 2, 3, 4], pairs: {axis_a: [1, 3], axis_b: [2, 4]}}",
                "  segment_b: {label: Spine 2, servo_ids: [5, 6, 7, 8], pairs: {axis_a: [5, 7], axis_b: [6, 8]}}",
            ]
        ),
        encoding="utf-8",
    )

    settings = ConfigLoader(base_dir=config_dir).load_settings()

    assert settings.robot.servo_ids == [1, 2, 3, 4, 5, 6, 7, 8]
    assert settings.robot.operating_context().operating_mode == "dual_segment"
    assert settings.robot.operating_context().expected_servo_ids == [1, 2, 3, 4, 5, 6, 7, 8]


def test_config_loader_can_save_local_overrides_and_load_robot_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "system.yaml").write_text('robot_config: "robot_1servo.yaml"\n', encoding="utf-8")
    (config_dir / "robot_1servo.yaml").write_text(
        "\n".join(
            [
                'mode: "1-servo"',
                "servo_ids: [1]",
                "tendon_to_servo: [1]",
                "tightening_rotation_by_servo: {1: cw}",
            ]
        ),
        encoding="utf-8",
    )
    loader = ConfigLoader(base_dir=config_dir)

    robot = loader.load_robot_config("robot_1servo.yaml")
    path = loader.save_system_local_overrides(
        {
            "mock_mode": False,
            "safety_overrides": {"fine_jog_step_ticks": 7},
            "robot_overrides": {"tightening_rotation_by_servo": {"1": "ccw"}},
        }
    )
    overrides = loader.load_system_local_overrides()

    assert robot.mode == "1-servo"
    assert robot.servo_ids == [1]
    assert path.name == "system.local.yaml"
    assert overrides["mock_mode"] is False
    assert overrides["safety_overrides"]["fine_jog_step_ticks"] == 7
    assert overrides["robot_overrides"]["tightening_rotation_by_servo"]["1"] == "ccw"


def test_config_loader_defaults_tracker_min_effective_fps_to_15_hz(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "system.yaml").write_text('robot_config: "robot_1servo.yaml"\n', encoding="utf-8")
    (config_dir / "robot_1servo.yaml").write_text(
        "\n".join(
            [
                'mode: "1-servo"',
                "servo_ids: [1]",
                "tendon_to_servo: [1]",
            ]
        ),
        encoding="utf-8",
    )

    settings = ConfigLoader(base_dir=config_dir).load_settings()

    assert settings.serial.tracker_min_effective_fps == 15.0
    assert settings.serial.tracker_fallback_enabled is False
    assert settings.serial.tracker_fallback_backend is None


def test_config_loader_defaults_use_canonical_runtime_artifact_paths(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "system.yaml").write_text('robot_config: "robot_1servo.yaml"\n', encoding="utf-8")
    (config_dir / "robot_1servo.yaml").write_text(
        "\n".join(
            [
                'mode: "1-servo"',
                "servo_ids: [1]",
                "tendon_to_servo: [1]",
            ]
        ),
        encoding="utf-8",
    )

    settings = ConfigLoader(base_dir=config_dir).load_settings()

    assert settings.serial.packet_capture_dir == ""
    assert settings.calibration.neutral_setpoints_path == "config/neutral_setpoints.json"
    assert (
        settings.calibration.latest_runtime_tip_calibration_path
        == "data/runtime_tip_calibration/latest_runtime_tip_calibration.json"
    )


def test_repo_pivot_calibration_sample_lives_with_config_examples() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    assert (repo_root / "config" / "pivot_calibration_sample.csv").exists()
    assert not (repo_root / "examples").exists()
