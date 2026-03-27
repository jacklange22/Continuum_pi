"""Registration validation metrics scaffold."""


def compute_fre_mm(residuals_xyz_mm: list[list[float]]) -> float:
    """Compute fiducial registration error from residual vectors."""
    if not residuals_xyz_mm:
        raise ValueError("residuals_xyz_mm must not be empty")
    total = 0.0
    count = 0
    for x, y, z in residuals_xyz_mm:
        total += x * x + y * y + z * z
        count += 1
    return (total / count) ** 0.5
