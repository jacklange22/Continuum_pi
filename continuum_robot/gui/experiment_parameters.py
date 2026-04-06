"""Helpers for generic experiment parameter editing in the GUI."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import yaml

from continuum_robot.experiments.schedules import CommandScheduleConfig


@dataclass
class ExperimentParameterField:
    """One editable experiment parameter."""

    key: str
    group: str
    label: str
    raw_value: str
    value_kind: str
    multiline: bool = False
    error: str | None = None


def build_parameter_fields(
    payload: dict[str, Any],
    *,
    drafts: dict[str, str] | None = None,
    errors: dict[str, str] | None = None,
) -> list[ExperimentParameterField]:
    """Flatten one experiment payload into generic editable fields."""
    payload = _expand_known_defaults(payload)
    raw_drafts = dict(drafts or {})
    raw_errors = dict(errors or {})
    fields: list[ExperimentParameterField] = []
    for group, key_path, value in _iter_parameter_items(payload):
        key = ".".join(key_path)
        field = ExperimentParameterField(
            key=key,
            group=group,
            label=_label_for_key(key_path[-1]),
            raw_value=_raw_value_for(value),
            value_kind=_value_kind_for(value),
            multiline=_multiline_for(value),
            error=raw_errors.get(key),
        )
        if key in raw_drafts:
            field.raw_value = raw_drafts[key]
        fields.append(field)
    return fields


def parse_field_value(*, value_kind: str, raw_value: str) -> Any:
    """Parse one field value from the UI draft string."""
    text = str(raw_value)
    stripped = text.strip()
    if value_kind == "bool":
        lowered = stripped.lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
        raise ValueError("Use true or false.")
    if stripped == "":
        return None
    try:
        return yaml.safe_load(text)
    except Exception as exc:  # pragma: no cover - yaml already returns precise parse errors
        raise ValueError(str(exc)) from exc


def apply_field_value(payload: dict[str, Any], *, key: str, value: Any) -> dict[str, Any]:
    """Return a payload copy with one nested field updated."""
    updated = deepcopy(payload)
    current: dict[str, Any] = updated
    segments = [segment for segment in str(key).split(".") if segment]
    if not segments:
        raise ValueError("Parameter key must not be empty.")
    for segment in segments[:-1]:
        nested = current.get(segment)
        if not isinstance(nested, dict):
            nested = {}
            current[segment] = nested
        current = nested
    current[segments[-1]] = value
    return updated


def dump_payload(payload: dict[str, Any]) -> str:
    """Render one payload as canonical YAML for snapshots and tests."""
    return yaml.safe_dump(dict(payload or {}), sort_keys=False) or "{}\n"


def _iter_parameter_items(
    payload: dict[str, Any],
    *,
    prefix: tuple[str, ...] = (),
    group: str = "General",
):
    for raw_key, value in payload.items():
        key = str(raw_key)
        key_path = (*prefix, key)
        next_group = group if len(key_path) == 1 else _group_label_for(prefix[0])
        if isinstance(value, dict) and value:
            nested_group = _group_label_for(key) if len(key_path) == 1 else next_group
            yield from _iter_parameter_items(value, prefix=key_path, group=nested_group)
            continue
        current_group = "General" if len(key_path) == 1 else next_group
        yield current_group, key_path, value


def _expand_known_defaults(payload: dict[str, Any]) -> dict[str, Any]:
    expanded = deepcopy(dict(payload or {}))
    for key in ("schedule", "command_schedule"):
        value = expanded.get(key)
        if not isinstance(value, dict):
            continue
        default_schedule = CommandScheduleConfig().to_dict()
        default_schedule.update(value)
        expanded[key] = default_schedule
    return expanded


def _value_kind_for(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (list, dict)):
        return "yaml"
    return "scalar"


def _multiline_for(value: Any) -> bool:
    return isinstance(value, (list, dict)) or (isinstance(value, str) and "\n" in value)


def _raw_value_for(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return (yaml.safe_dump(value, sort_keys=False) or "").strip()
    return str(value)


def _group_label_for(key: str) -> str:
    return _label_for_key(key)


def _label_for_key(key: str) -> str:
    parts = [segment for segment in str(key).replace("-", "_").split("_") if segment]
    if not parts:
        return str(key)
    return " ".join(part.upper() if part in {"id", "rms", "csv"} else part.capitalize() for part in parts)
