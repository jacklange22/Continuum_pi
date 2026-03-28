"""Deprecated alias for the canonical tracker doctor command."""

from continuum_robot.tracking.doctor_cli import main


if __name__ == "__main__":
    print("NOTE: scripts/run_diagnostics.py is deprecated. Use scripts/run_tracker_doctor.py instead.")
    raise SystemExit(main())
