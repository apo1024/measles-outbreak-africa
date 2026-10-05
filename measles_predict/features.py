"""Feature engineering shared by training, evaluation and prediction.

Input: the `model_inputs` table (one row = country x month) produced by the
study database (see data/model_inputs.csv).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SUBREGIONS = ["Afrique australe", "Afrique centrale", "Afrique de l'Est",
              "Afrique de l'Ouest", "Afrique du Nord"]
REFERENCE_SUBREGION = "Afrique du Nord"

# Targets
TARGETS = {
    "outbreak_next1": "Flambée au mois suivant (t+1)",
    "outbreak_next3": "Flambée dans les 3 mois suivants (t+1 à t+3)",
}
DEFAULT_TARGET = "outbreak_next3"

# Human-readable labels (FR) for reports
LABELS = {
    "log_cases": "log(1+cas mois t)",
    "log_lag1": "log(1+cas t-1)",
    "log_lag2": "log(1+cas t-2)",
    "log_sum_3m": "log(1+cas 3 mois précédents)",
    "log_sum_12m": "log(1+cas 12 mois précédents)",
    "log_baseline": "log(1+moyenne de référence 24 mois)",
    "ratio_to_threshold": "Cas t / seuil épidémique",
    "outbreak_now": "Flambée en cours (mois t)",
    "months_since_outbreak": "Mois depuis la dernière flambée",
    "month_sin": "Saisonnalité (sinus)",
    "month_cos": "Saisonnalité (cosinus)",
    "mcv1": "Couverture MCV1 (%) année N-1",
    "mcv2": "Couverture MCV2 (%) année N-1",
    "ivr_score": "Indice de vulnérabilité (IVR) N-1",
    "children_pct": "Population 0-14 ans (%)",
    "log_density": "log densité de population",
    "urban_pct": "Population urbaine (%)",
    "log_population": "log population",
}
LABELS.update({f"sr_{s}": f"Sous-région : {s}" for s in SUBREGIONS})

FEATURES = [
    "log_cases", "log_lag1", "log_lag2", "log_sum_3m", "log_sum_12m", "log_baseline",
    "ratio_to_threshold", "outbreak_now", "months_since_outbreak", "month_sin", "month_cos",
    "mcv1", "mcv2", "ivr_score", "children_pct", "log_density", "urban_pct", "log_population",
] + [f"sr_{s}" for s in SUBREGIONS if s != REFERENCE_SUBREGION]

# Compact feature set for the interpretable statistical model (Model A)
FEATURES_A = [
    "log_cases", "log_sum_12m", "ratio_to_threshold", "outbreak_now",
    "months_since_outbreak", "month_sin", "month_cos", "mcv1", "mcv2", "ivr_score",
    "log_population",
]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with model features added."""
    d = df.copy()
    d["log_cases"] = np.log1p(d["cases"])
    d["log_lag1"] = np.log1p(d["cases_lag1"])
    d["log_lag2"] = np.log1p(d["cases_lag2"])
    d["log_sum_3m"] = np.log1p(d["cases_sum_3m"])
    d["log_sum_12m"] = np.log1p(d["cases_sum_12m"])
    d["log_baseline"] = np.log1p(d["baseline_mean_24m"])
    d["ratio_to_threshold"] = np.clip(d["cases"] / d["epidemic_threshold"], 0, 20)
    d["outbreak_now"] = d["outbreak"]
    d["log_density"] = np.log1p(d["pop_density"])
    for s in SUBREGIONS:
        if s != REFERENCE_SUBREGION:
            d[f"sr_{s}"] = (d["subregion"] == s).astype(float)
    return d


def complete_rows(d: pd.DataFrame, features=FEATURES) -> pd.Series:
    """Boolean mask of rows where every feature is available."""
    return d[features].notna().all(axis=1)
