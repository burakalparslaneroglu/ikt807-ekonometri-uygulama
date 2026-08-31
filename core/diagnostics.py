"""OLS kaldıraç, studentized artık ve Cook uzaklığı tanıları."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.ols import OLSFit


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class InfluenceDiagnostics:
    leverage: FloatArray
    internally_studentized_residuals: FloatArray
    cooks_distance: FloatArray


def influence_diagnostics(fit: OLSFit) -> InfluenceDiagnostics:
    if fit.degrees_of_freedom <= 0:
        raise ValueError("Etki tanıları için pozitif serbestlik derecesi gerekir.")
    bread = np.linalg.inv(fit.design_matrix.T @ fit.design_matrix)
    leverage = np.einsum("ij,jk,ik->i", fit.design_matrix, bread, fit.design_matrix)
    mse = float(fit.residuals @ fit.residuals) / fit.degrees_of_freedom
    residual_scale = np.sqrt(np.maximum(mse * (1 - leverage), np.finfo(float).eps))
    studentized = fit.residuals / residual_scale
    cooks = (
        studentized**2
        * leverage
        / (fit.nparams * np.maximum(1 - leverage, np.finfo(float).eps))
    )
    return InfluenceDiagnostics(
        leverage=leverage,
        internally_studentized_residuals=studentized,
        cooks_distance=cooks,
    )
