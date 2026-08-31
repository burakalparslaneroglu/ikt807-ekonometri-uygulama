"""Tanımlama, seçim ayrıştırması ve içsellik için kontrollü DGP'ler."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.ols import fit_ols


@dataclass(frozen=True)
class SelectionDecomposition:
    average_treatment_effect: float
    treatment_on_treated: float
    observed_difference: float
    selection_bias: float


@dataclass(frozen=True)
class EndogeneityConfig:
    nobs: int = 1000
    seed: int = 803
    structural_effect: float = 2.0
    confounding_strength: float = 0.8
    confounder_effect: float = 1.0

    def __post_init__(self) -> None:
        if self.nobs < 100:
            raise ValueError("İçsellik DGP'si en az 100 gözlem içermelidir.")
        if self.seed < 0:
            raise ValueError("seed negatif olamaz.")
        if not 0 <= self.confounding_strength <= 2:
            raise ValueError("confounding_strength 0 ile 2 arasında olmalıdır.")


@dataclass(frozen=True)
class TrialConfig:
    students: int = 1200
    schools: int = 60
    seed: int = 833
    treatment_effect: float = 1.4

    def __post_init__(self) -> None:
        if self.students < 200:
            raise ValueError("Deney DGP'si en az 200 öğrenci içermelidir.")
        if not 10 <= self.schools <= self.students:
            raise ValueError("Okul sayısı 10 ile öğrenci sayısı arasında olmalıdır.")


def simulate_selection_data(
    *,
    nobs: int = 1200,
    seed: int = 813,
    treatment_effect: float = 1.5,
    selection_strength: float = 1.0,
    randomized: bool = False,
) -> pd.DataFrame:
    if nobs < 100:
        raise ValueError("Seçim DGP'si en az 100 gözlem içermelidir.")
    rng = np.random.default_rng(seed)
    baseline = rng.normal(size=nobs)
    y0 = 5 + 1.2 * baseline + rng.normal(scale=0.7, size=nobs)
    y1 = y0 + treatment_effect + 0.25 * baseline
    if randomized:
        treatment = rng.binomial(1, 0.5, size=nobs)
    else:
        propensity = 1 / (1 + np.exp(-selection_strength * baseline))
        treatment = rng.binomial(1, propensity)
    outcome = treatment * y1 + (1 - treatment) * y0
    return pd.DataFrame(
        {
            "baseline": baseline,
            "y0": y0,
            "y1": y1,
            "treatment": treatment,
            "outcome": outcome,
        }
    )


def decompose_observed_difference(frame: pd.DataFrame) -> SelectionDecomposition:
    required = {"y0", "y1", "treatment", "outcome"}
    if not required.issubset(frame.columns):
        raise ValueError("Seçim ayrıştırması için potansiyel sonuç sütunları eksik.")
    treated = frame["treatment"].eq(1)
    control = ~treated
    if not treated.any() or not control.any():
        raise ValueError("Tedavi ve kontrol gruplarının ikisi de gözlenmelidir.")
    ate = float((frame["y1"] - frame["y0"]).mean())
    att = float((frame.loc[treated, "y1"] - frame.loc[treated, "y0"]).mean())
    observed = float(
        frame.loc[treated, "outcome"].mean()
        - frame.loc[control, "outcome"].mean()
    )
    selection = float(
        frame.loc[treated, "y0"].mean() - frame.loc[control, "y0"].mean()
    )
    return SelectionDecomposition(
        average_treatment_effect=ate,
        treatment_on_treated=att,
        observed_difference=observed,
        selection_bias=selection,
    )


def simulate_endogeneity_data(config: EndogeneityConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    confounder = rng.normal(size=config.nobs)
    independent_component = rng.normal(size=config.nobs)
    regressor = (
        config.confounding_strength * confounder + independent_component
    )
    outcome = (
        config.structural_effect * regressor
        + config.confounder_effect * confounder
        + rng.normal(size=config.nobs)
    )
    return pd.DataFrame(
        {"x": regressor, "u": confounder, "y": outcome}
    )


def endogeneity_probability_limit(config: EndogeneityConfig) -> float:
    strength = config.confounding_strength
    omitted_variable_term = (
        config.confounder_effect * strength / (strength**2 + 1)
    )
    return config.structural_effect + omitted_variable_term


def endogeneity_estimates(config: EndogeneityConfig) -> tuple[float, float]:
    frame = simulate_endogeneity_data(config)
    omitted = fit_ols(frame["y"], frame[["x"]], ("x",)).coefficient("x")
    controlled = fit_ols(
        frame["y"], frame[["x", "u"]], ("x", "u")
    ).coefficient("x")
    return omitted, controlled


def endogeneity_monte_carlo(
    sample_sizes: tuple[int, ...],
    *,
    repetitions: int,
    seed: int,
    confounding_strength: float,
) -> pd.DataFrame:
    if repetitions < 20:
        raise ValueError("Monte Carlo en az 20 tekrar içermelidir.")
    rows: list[dict[str, float | int | str]] = []
    for nobs in sample_sizes:
        for repetition in range(repetitions):
            config = EndogeneityConfig(
                nobs=nobs,
                seed=seed + repetition + 10_000 * nobs,
                confounding_strength=confounding_strength,
            )
            omitted, controlled = endogeneity_estimates(config)
            rows.extend(
                (
                    {"n": nobs, "Tekrar": repetition, "Model": "U dışarıda", "Tahmin": omitted},
                    {"n": nobs, "Tekrar": repetition, "Model": "U kontrol", "Tahmin": controlled},
                )
            )
    return pd.DataFrame(rows)


def simulate_cluster_trial(config: TrialConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    school = np.arange(config.students) % config.schools
    rng.shuffle(school)
    school_treatment = rng.binomial(1, 0.5, size=config.schools)
    tracking = school_treatment[school]
    baseline = rng.normal(size=config.students)
    girl = rng.binomial(1, 0.5, size=config.students)
    age = rng.normal(10.5, 0.8, size=config.students)
    school_shock = rng.normal(scale=1.4, size=config.schools)[school]
    outcome = (
        50
        + config.treatment_effect * tracking
        + 5.5 * baseline
        - 0.6 * girl
        + 0.3 * (age - 10.5)
        + school_shock
        + rng.normal(scale=5.5, size=config.students)
    )
    lowstream = ((tracking == 1) & (baseline < np.median(baseline))).astype(int)
    return pd.DataFrame(
        {
            "schoolid": school,
            "tracking": tracking,
            "std_mark": baseline,
            "girl": girl,
            "agetest": age,
            "lowstream": lowstream,
            "totalscore": outcome,
        }
    )
