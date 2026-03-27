# Repo Instructions

## Overview

- `continuum_robot/` contains the Python application code for tracking, registration, servo control, experiments, config loading, and GUI/controller scaffolding.
- `tracker_bridge/` contains the C++ Aurora bridge that owns the NDI `CombinedApi` lifecycle and publishes tracker data over a Unix socket.
- `config/` contains editable runtime YAML configuration.
- `tests/` contains unit coverage for transform math, tracker parsing, registration persistence, and related services.
- `references/` contains legacy reference scripts, captured data, and vendor/reference documents used for comparison only.
- `tools/` contains legacy tool geometry, registration-point assets, and other lab reference inputs.

## Protected Paths

- Treat everything under `references/` as read-only reference material. Do not modify, rename, delete, or reformat files in this directory unless the user explicitly asks for it.
- Treat everything under `tools/` as read-only reference input. Do not modify, rename, delete, or reformat files in this directory unless the user explicitly asks for it.
- This protection includes SolidWorks-derived registration-point files, transform exports, pen-probe geometry files, and any other lab artifacts stored in `tools/`.

## Working Rules

- When implementing features, read from `references/` and `tools/` if needed, but write new outputs to normal runtime locations such as `data/`, `config/`, or source files under `continuum_robot/`.
- If current application behavior conflicts with a legacy reference, preserve the protected files and adapt the application code or config instead.
