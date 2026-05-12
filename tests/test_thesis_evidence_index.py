from __future__ import annotations

import json
from pathlib import Path

from continuum_robot.data.build_thesis_evidence_index import build_thesis_evidence_index


def _write_run(root: Path, name: str, *, review_status: str, trust_mode: str = "thesis_trusted", mock: bool = False) -> Path:
    run_dir = root / "data" / "experiments" / "single_segment_repeatability" / name
    run_dir.mkdir(parents=True)
    provenance = {"operating_mode": "single_segment", "mock_mode": bool(mock)}
    trust = {
        "run_trust_mode": trust_mode,
        "valid_for_model_training": False,
        "valid_for_thesis_repeatability": trust_mode == "thesis_trusted" and not mock,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "experiment_name": "single_segment_repeatability",
                "run_id": name,
                "provenance_info": provenance,
                "trust_info": trust,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "experiment_name": "single_segment_repeatability",
                "run_id": name,
                "success": True,
                "status": "success",
                "experiment_metrics": {
                    "run_provenance": provenance,
                    "run_trust": trust,
                    "run_trust_mode": trust_mode,
                    "valid_for_model_training": trust["valid_for_model_training"],
                    "valid_for_thesis_repeatability": trust["valid_for_thesis_repeatability"],
                    "mock_mode": bool(mock),
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_review.json").write_text(
        json.dumps({"review_status": review_status, "include_in_evidence_index": False}),
        encoding="utf-8",
    )
    return run_dir


def _included_run_ids(index_dir: Path) -> set[str]:
    payload = json.loads((index_dir / "thesis_evidence_index.json").read_text(encoding="utf-8"))
    runs = payload["experiments"].get("single_segment_repeatability", [])
    return {str(run["run_id"]) for run in runs}


def test_evidence_index_includes_thesis_and_advisor_reviews_only_by_default(tmp_path: Path) -> None:
    _write_run(tmp_path, "debug-run", review_status="debug")
    _write_run(tmp_path, "thesis-run", review_status="thesis_candidate")
    _write_run(tmp_path, "advisor-run", review_status="advisor_share")

    output_dir = build_thesis_evidence_index(project_root=tmp_path)

    assert _included_run_ids(output_dir) == {"thesis-run", "advisor-run"}


def test_evidence_index_excludes_mock_even_when_reviewed_by_default(tmp_path: Path) -> None:
    _write_run(tmp_path, "mock-run", review_status="thesis_candidate", trust_mode="mock", mock=True)

    output_dir = build_thesis_evidence_index(project_root=tmp_path)

    assert _included_run_ids(output_dir) == set()


def test_evidence_index_debug_and_mock_flags_are_explicit(tmp_path: Path) -> None:
    _write_run(tmp_path, "debug-run", review_status="debug", trust_mode="debug")
    _write_run(tmp_path, "mock-run", review_status="thesis_candidate", trust_mode="mock", mock=True)

    output_dir = build_thesis_evidence_index(
        project_root=tmp_path,
        include_debug=True,
        include_mock=True,
    )

    assert _included_run_ids(output_dir) == {"debug-run", "mock-run"}
