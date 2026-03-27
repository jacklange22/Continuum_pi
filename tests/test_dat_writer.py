from pathlib import Path

from continuum_robot.experiments.dat_writer import DatRunWriter


def test_dat_writer_writes_expected_headers(tmp_path: Path) -> None:
    writer = DatRunWriter(output_dir=tmp_path)
    path = writer.write_run(
        num_cables=4,
        rows=[
            {
                "timestamp_utc": "2026-01-01T00:00:00Z",
                "index": 0,
                "repeat_index": 0,
                "commanded_displacement_cm": [0.0, 0.0, 0.0, 0.0],
                "commanded_goal_ticks": [2048, 2048, 2048, 2048],
                "servo_position_ticks": [2048, 2048, 2048, 2048],
                "servo_current_ma": [120, 121, 122, 123],
                "servo_voltage_mv": [12000, 12000, 12000, 12000],
                "tool_0A_translation_mm": [10.0, 11.0, 12.0],
                "tool_0B_translation_mm": [20.0, 21.0, 22.0],
                "tip_position_xyz": [1.0, 2.0, 3.0],
                "tip_tangent_xyz": [0.1, 0.2, 0.3],
            }
        ],
        filename_stem="sample",
    )

    text = path.read_text(encoding="utf-8")
    assert "NUM_CABLES: 4" in text
    assert "NUM_MEASUREMENTS: 1" in text
    assert "---" in text
    assert "2026-01-01T00:00:00Z,0,0,0.0,0.0,0.0,0.0" in text
    assert "10.0,11.0,12.0,20.0,21.0,22.0,1.0,2.0,3.0,0.1,0.2,0.3" in text
