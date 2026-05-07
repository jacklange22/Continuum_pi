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
    except Exception as exc:
        print(f"Run management failed: {exc}", file=sys.stderr)
        return 1
    parser.error("Choose --mark, --archive, or --trash.")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mark, archive, or trash one local experiment run folder.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--mark", metavar="RUN_DIR", help="Write/update run_review.json for this run.")
    action.add_argument("--archive", metavar="RUN_DIR", help="Move this run to data/experiments_archived/<experiment>.")
    action.add_argument("--trash", metavar="RUN_DIR", help="Move this run to data/trash/<experiment>.")
    parser.add_argument("--project-root", default=".", help="Project root. Defaults to current directory.")
    parser.add_argument("--status", choices=sorted(REVIEW_STATUSES), default="debug", help="Review status for --mark.")
    parser.add_argument("--notes", default="", help="Review notes for --mark.")
    parser.add_argument("--reviewed-by", default="", help="Optional reviewer name for --mark.")
    evidence = parser.add_mutually_exclusive_group()
    evidence.add_argument("--include-in-evidence-index", action="store_true", help="Mark this run for evidence-index inclusion.")
    evidence.add_argument("--exclude-from-evidence-index", action="store_true", help="Exclude this run from evidence-index inclusion.")
    parser.add_argument("--force", action="store_true", help="Allow archive/trash of protected review statuses after explicit confirmation.")
    return parser


def _evidence_flag(args: argparse.Namespace) -> bool | None:
    if bool(args.include_in_evidence_index):
        return True
    if bool(args.exclude_from_evidence_index):
        return False
    return None


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

