"""Konu 01-02 için kontrollü ücret ve çıkarım veri üretim süreçleri."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WageDGPConfig:
    nobs: int = 800
    seed: int = 807
    nonlinear_strength: float = 0.0
    heteroskedasticity: float = 0.0
    ability_confounding: float = 0.0
    cluster_correlation: float = 0.0
    clusters: int = 40

    def __post_init__(self) -> None:
        if self.nobs < 80:
            raise ValueError("DGP en az 80 gözlem içermelidir.")
        if self.seed < 0:
            raise ValueError("seed negatif olamaz.")
        if not 0 <= self.nonlinear_strength <= 2:
            raise ValueError("nonlinear_strength 0 ile 2 arasında olmalıdır.")
        if not 0 <= self.heteroskedasticity <= 2:
            raise ValueError("heteroskedasticity 0 ile 2 arasında olmalıdır.")
        if not 0 <= self.ability_confounding <= 2:
            raise ValueError("ability_confounding 0 ile 2 arasında olmalıdır.")
        if not 0 <= self.cluster_correlation <= 1:
            raise ValueError("cluster_correlation 0 ile 1 arasında olmalıdır.")
        if not 2 <= self.clusters <= self.nobs:
            raise ValueError("Küme sayısı 2 ile gözlem sayısı arasında olmalıdır.")


def simulate_wage_data(config: WageDGPConfig) -> pd.DataFrame:
    """Bilinen doğrusal bileşenleri olan deterministik bir ücret DGP'si üretir."""

    rng = np.random.default_rng(config.seed)
    cluster = np.arange(config.nobs) % config.clusters
    rng.shuffle(cluster)
    ability = rng.normal(size=config.nobs)
    education = np.clip(
        np.rint(13.5 + 1.15 * ability + rng.normal(scale=1.7, size=config.nobs)),
        8,
        20,
    )
    experience = rng.uniform(0, 36, size=config.nobs)
    female = rng.binomial(1, 0.5, size=config.nobs)
    cluster_shock = rng.normal(size=config.clusters)[cluster]
    individual_shock = rng.normal(size=config.nobs)
    composite_shock = (
        np.sqrt(config.cluster_correlation) * cluster_shock
        + np.sqrt(1 - config.cluster_correlation) * individual_shock
    )
    noise_scale = 0.20 * (
        1 + config.heteroskedasticity * (education - education.min()) / 12
    )
    centered_education = education - 14
    log_wage = (
        1.25
        + 0.08 * education
        + 0.032 * experience
        - 0.045 * (experience**2 / 100)
        - 0.16 * female
        + 0.012 * config.nonlinear_strength * centered_education**2
        + 0.16 * config.ability_confounding * ability
        + noise_scale * composite_shock
    )
    frame = pd.DataFrame(
        {
            "education": education,
            "experience": experience,
            "experience2_100": experience**2 / 100,
            "female": female,
            "cluster": cluster,
            "lwage": log_wage,
        }
    )
    frame["hrwage"] = np.exp(frame["lwage"])
    return frame


def add_influential_observation(frame: pd.DataFrame) -> pd.DataFrame:
    """Kaldıraç ve etki ayrımını göstermek için tek bir kontrollü gözlem ekler."""

    required = {"education", "experience", "experience2_100", "female", "cluster", "lwage", "hrwage"}
    if not required.issubset(frame.columns):
        raise ValueError("Etkili gözlem yalnız ücret DGP çerçevesine eklenebilir.")
    row = {
        "education": 25.0,
        "experience": 2.0,
        "experience2_100": 0.04,
        "female": 0,
        "cluster": int(frame["cluster"].max()) + 1,
        "lwage": float(frame["lwage"].mean() - 1.1),
    }
    row["hrwage"] = float(np.exp(row["lwage"]))
    return pd.concat((frame, pd.DataFrame([row])), ignore_index=True)
