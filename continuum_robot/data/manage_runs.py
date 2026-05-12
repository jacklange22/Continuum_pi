"""CLI for local run review, archive, and trash actions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from continuum_robot.data.run_management import (
    REVIEW_STATUSES,
    archive_run,
    trash_run,
    write_run_review,
)
from continuum_robot.data.curation import (
    build_curation_report,
    preview_cleanup,
    write_curation_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()
    try:
        if args.mark:
            review = write_run_review(
                Path(args.mark),
                status=str(args.status),
                notes=str(args.notes or ""),
                reviewed_by=str(args.reviewed_by or ""),
                include_in_evidence_index=_evidence_flag(args),
            )
            print(f"Marked run: {Path(args.mark).resolve()}")
            print(f"Review status: {review.review_status}")
            print(f"Include in evidence index: {review.include_in_evidence_index}")
            return 0
        if args.archive:
            result = archive_run(Path(args.archive), project_root=project_root, force=bool(args.force))
            print(f"Archived run: {result.source_path} -> {result.destination_path}")
            return 0
        if args.trash:
            result = trash_run(Path(args.trash), project_root=project_root, force=bool(args.force))
            print(f"Moved run to trash: {result.source_path} -> {result.destination_path}")
            return 0
        if args.curation_report:
            report = write_curation_report(
                build_curation_report(project_root),
                output_dir=Path(args.output_dir) if args.output_dir else None,
                also_write_root_aliases=bool(args.write_root_report_aliases),
            )
            print(f"Data curation JSON: {report.json_path}")
            print(f"Data curation report: {report.markdown_path}")
            print(f"Candidate CSV: {report.csv_path}")
            return 0
        if args.preview_cleanup_plan or args.apply_trash_candidates or args.apply_archive_candidates or args.permanently_delete_trash:
            dry_run = bool(args.dry_run or args.preview_cleanup_plan)
            manifest = preview_cleanup(
                project_root,
                apply_trash_candidates=bool(args.apply_trash_candidates),
                apply_archive_candidates=bool(args.apply_archive_candidates),
                permanently_delete_trash=bool(args.permanently_delete_trash),
                force=bool(args.force),
                dry_run=dry_run,
                filters=_cleanup_filters(args),
            )
            print(f"Cleanup manifest: {manifest.manifest_path}")
            print(f"Cleanup summary: {manifest.summary_path}")
            print(
                f"Mode={manifest.mode} actions={manifest.action_count} "
                f"applied={manifest.applied_count} skipped={manifest.skipped_count} errors={manifest.error_count}"
            )
            return 1 if manifest.error_count else 0
    except Exception as exc:
        print(f"Run management failed: {exc}", file=sys.stderr)
        return 1
    parser.error("Choose --mark, --archive, --trash, --curation-report, or a cleanup action.")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mark, archive, trash, inventory, or safely clean local run folders.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--mark", metavar="RUN_DIR", help="Write/update run_review.json for this run.")
    action.add_argument("--archive", metavar="RUN_DIR", help="Move this run to data/experiments_archived/<experiment>.")
    action.add_argument("--trash", metavar="RUN_DIR", help="Move this run to data/trash/<experiment>.")
    action.add_argument("--curation-report", action="store_true", help="Write a non-mutating data curation report.")
    action.add_argument("--preview-cleanup-plan", action="store_true", help="Write a dry-run cleanup manifest.")
    action.add_argument("--apply-trash-candidates", action="store_true", help="Move selected trash candidates to data/trash.")
    action.add_argument("--apply-archive-candidates", action="store_true", help="Move selected archive candidates to data/experiments_archived.")
    action.add_argument("--permanently-delete-trash", action="store_true", help="Delete selected items only from data/trash; requires --force.")
    parser.add_argument("--project-root", default=".", help="Project root. Defaults to current directory.")
    parser.add_argument("--output-dir", help="Output directory for --curation-report.")
    parser.add_argument(
        "--write-root-report-aliases",
        action="store_true",
        help="Also write data_curation_report.md/json at the project root for review.",
    )
    parser.add_argument("--status", choices=sorted(REVIEW_STATUSES), default="debug", help="Review status for --mark.")
    parser.add_argument("--notes", default="", help="Review notes for --mark.")
    parser.add_argument("--reviewed-by", default="", help="Optional reviewer name for --mark.")
    evidence = parser.add_mutually_exclusive_group()
    evidence.add_argument("--include-in-evidence-index", action="store_true", help="Mark this run for evidence-index inclusion.")
    evidence.add_argument("--exclude-from-evidence-index", action="store_true", help="Exclude this run from evidence-index inclusion.")
    parser.add_argument("--dry-run", action="store_true", help="Preview cleanup actions without moving or deleting anything.")
    parser.add_argument("--older-than", metavar="YYYY-MM-DD", help="Limit cleanup candidates to items older than this date.")
    parser.add_argument("--experiment", help="Limit cleanup candidates to this experiment name.")
    parser.add_argument("--root", help="Limit cleanup candidates to a data root such as data/mock_experiments.")
    parser.add_argument("--review-status", choices=sorted(REVIEW_STATUSES), help="Limit cleanup candidates by review status.")
    parser.add_argument("--mock-only", action="store_true", help="Limit cleanup candidates to mock runs.")
    parser.add_argument("--generated-only", action="store_true", help="Limit cleanup candidates to generated-ignore items.")
    parser.add_argument("--max-size", type=int, help="Limit cleanup candidates to items at or below this many bytes.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow archive/trash of protected statuses after confirmation, and required for permanent data/trash deletion.",
    )
    return parser


def _evidence_flag(args: argparse.Namespace) -> bool | None:
    if bool(args.include_in_evidence_index):
        return True
    if bool(args.exclude_from_evidence_index):
        return False
    return None


def _cleanup_filters(args: argparse.Namespace) -> dict[str, object]:
    return {
        "older_than": args.older_than,
        "experiment": args.experiment,
        "root": args.root,
        "review_status": args.review_status,
        "mock_only": bool(args.mock_only),
        "generated_only": bool(args.generated_only),
        "max_size": args.max_size,
    }


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
