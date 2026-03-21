"""Tendon channel to servo channel mapping helpers."""


def map_tendon_displacement_to_servo_order(
    tendon_displacements: list[float], tendon_to_servo_index: list[int]
) -> list[float]:
    """Reorder tendon displacement vector into servo index order."""
    if len(tendon_displacements) != len(tendon_to_servo_index):
        raise ValueError("Length mismatch between tendon vector and mapping")
    out = [0.0] * len(tendon_displacements)
    for tendon_idx, servo_idx in enumerate(tendon_to_servo_index):
        out[servo_idx - 1] = tendon_displacements[tendon_idx]
    return out
