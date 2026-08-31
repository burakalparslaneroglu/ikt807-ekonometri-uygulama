"""Ortak öğrenci-facing sayısal biçimlendirme kuralları."""

from __future__ import annotations


def format_p_value(value: float) -> str:
    if not 0 <= value <= 1:
        raise ValueError("p-değeri 0 ile 1 arasında olmalıdır.")
    if value < 0.001:
        return "p < 0.001"
    return f"p = {value:.3f}"


def format_estimate(value: float, digits: int = 3) -> str:
    if digits < 0:
        raise ValueError("digits negatif olamaz.")
    return f"{value:.{digits}f}".replace(".", ",")
