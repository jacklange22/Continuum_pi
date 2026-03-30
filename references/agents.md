# Reference Folder Guidance

This folder is read-only reference material.

Use this directory for:
- proven legacy math
- packet parsing logic
- transform conventions
- hardware contract details
- prior validation and artifact conventions

Do not treat this folder as the target architecture.

Important interpretation rules:
- Reuse logic, not structure.
- Reuse math, not folder layout.
- Reuse validation ideas, not old workflow sprawl.
- Do not port Arduino-era transport code directly into the current OpenRB/XC330 runtime unless explicitly asked.

Highest-value files:
- `continuum_aurora.py`: legacy Aurora request/read/parse/transform path
- `pivot_cal_lsq.m`: trusted pivot calibration reference
- `new_pivot_cal.py`: Python pivot-calibration reference
- `rigid_registration.py`: legacy registration math/path
- `openrb-150.md`: OpenRB hardware behavior and setup
- `xc330-m288.md`: DYNAMIXEL control table and servo behavior
- `repeatability.py`: old experiment behavior reference
- `RegistrationPoints.csv`, `grid_line_reg.txt`: landmark/registration reference data