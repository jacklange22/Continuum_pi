from pathlib import Path

from continuum_robot.experiments.dat_writer import DatRunWriter


def test_dat_writer_writes_expected_headers(tmp_path: Path) -> None:
    writer = DatRunWriter(output_dir=tmp_path)
    path = writer.write_run(
        num_cables=4,
        rows=[
            {
                "index": 0,
                "commanded_displacement_cm": [0.0, 0.0, 0.0, 0.0],
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
    assert "0,0.0,0.0,0.0,0.0,1.0,2.0,3.0,0.1,0.2,0.3" in text
