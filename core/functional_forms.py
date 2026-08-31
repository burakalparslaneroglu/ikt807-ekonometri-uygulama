"""Fonksiyonel biçim yorumları için küçük, test edilebilir dönüşümler."""

from __future__ import annotations

from math import exp


def exact_percent_change(log_coefficient: float, change: float = 1.0) -> float:
    """Log-sonuç modelinde sonlu X değişiminin tam yüzde etkisi."""

    return 100 * (exp(log_coefficient * change) - 1)


def quadratic_marginal_effect(
    linear_coefficient: float,
    quadratic_coefficient: float,
    evaluation_point: float,
) -> float:
    return linear_coefficient + 2 * quadratic_coefficient * evaluation_point


def interaction_slope(
    base_slope: float,
    interaction_coefficient: float,
    group_value: float,
) -> float:
    return base_slope + interaction_coefficient * group_value
