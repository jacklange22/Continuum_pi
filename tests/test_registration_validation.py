import pytest

from continuum_robot.registration.validation import compute_fre_mm


def test_compute_fre_mm_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        compute_fre_mm([])


def test_compute_fre_mm_returns_rms_norm() -> None:
    out = compute_fre_mm([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    assert out == (25.0 / 2.0) ** 0.5
