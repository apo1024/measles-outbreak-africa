"""Model A (statistical) and Model B (machine learning) for measles outbreak prediction.

Model A  - Binomial GLM (logistic regression) for P(outbreak), plus a
           Negative Binomial GLM for next-month case counts (offset = log population).
Model B  - Gradient boosted trees (XGBoost) classifier; a Random Forest is kept
           as a secondary learner for comparison.
Ensemble - Simple mean of Model A and Model B probabilities.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             roc_auc_score)
from xgboost import XGBClassifier

from .features import FEATURES, FEATURES_A, LABELS


# --------------------------------------------------------------------- Model A
@dataclass
class ModelA:
    """Logistic regression on standardised features (statsmodels GLM Binomial)."""
    features: list = field(default_factory=lambda: list(FEATURES_A))
    mean_: dict = field(default_factory=dict)
    sd_: dict = field(default_factory=dict)
    params_: dict = field(default_factory=dict)
    bse_: dict = field(default_factory=dict)
    result_: object = None

    def _z(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = pd.DataFrame({f: (X[f] - self.mean_[f]) / self.sd_[f] for f in self.features}, index=X.index)
        return sm.add_constant(Z, has_constant="add")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.mean_ = X[self.features].mean().to_dict()
        self.sd_ = X[self.features].std().replace(0, 1).to_dict()
        res = sm.GLM(y.astype(float), self._z(X), family=sm.families.Binomial()).fit()
        self.result_ = res
        self.params_ = res.params.to_dict()
        self.bse_ = res.bse.to_dict()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Z = self._z(X)
        eta = sum(Z[k] * v for k, v in self.params_.items())
        return (1 / (1 + np.exp(-eta))).to_numpy()

    def odds_ratios(self) -> pd.DataFrame:
        rows = []
        for k, b in self.params_.items():
            se = self.bse_[k]
            rows.append({"variable": k, "label": LABELS.get(k, k), "coef_std": b,
                         "OR_per_SD": np.exp(b), "OR_low": np.exp(b - 1.96 * se),
                         "OR_high": np.exp(b + 1.96 * se),
                         "p_value": float(self.result_.pvalues[k]) if self.result_ is not None else None,
                         "sd": self.sd_.get(k)})
        return pd.DataFrame(rows)

    def to_json(self) -> dict:
        return {"type": "logistic_glm_standardised", "features": self.features,
                "mean": self.mean_, "sd": self.sd_, "coef": self.params_}

    @classmethod
    def from_json(cls, js: dict) -> "ModelA":
        m = cls(features=js["features"])
        m.mean_, m.sd_, m.params_ = js["mean"], js["sd"], js["coef"]
        return m


@dataclass
class ModelACounts:
    """Negative binomial GLM (alpha = 0.5) for next-month cases.

    Population enters as a covariate (log scale) rather than as an offset because
    the autoregressive terms are log raw counts, not rates.
    """
    features: list = field(default_factory=lambda: [
        "log_cases", "log_lag1", "log_sum_12m", "month_sin", "month_cos",
        "mcv1", "mcv2", "ivr_score", "log_population"])
    alpha: float = 0.5
    result_: object = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        Xc = sm.add_constant(X[self.features], has_constant="add")
        self.result_ = sm.GLM(y, Xc, family=sm.families.NegativeBinomial(alpha=self.alpha)).fit()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xc = sm.add_constant(X[self.features], has_constant="add")
        return np.asarray(self.result_.predict(Xc))

    def irr(self) -> pd.DataFrame:
        r = self.result_
        ci = r.conf_int()
        return pd.DataFrame({"variable": r.params.index,
                             "label": [LABELS.get(k, k) for k in r.params.index],
                             "IRR": np.exp(r.params.values), "IRR_low": np.exp(ci[0].values),
                             "IRR_high": np.exp(ci[1].values), "p_value": r.pvalues.values})


# --------------------------------------------------------------------- Model B
XGB_PARAMS = dict(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                  eval_metric="logloss", n_jobs=4, random_state=42)
RF_PARAMS = dict(n_estimators=500, min_samples_leaf=5, max_features="sqrt",
                 n_jobs=4, random_state=42)


class ModelB:
    """XGBoost classifier (primary ML model)."""

    def __init__(self, features=FEATURES, **params):
        self.features = list(features)
        self.model = XGBClassifier(**{**XGB_PARAMS, **params})

    def fit(self, X, y):
        self.model.fit(X[self.features], y.astype(int))
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X[self.features])[:, 1]

    def importance(self) -> pd.DataFrame:
        gain = self.model.get_booster().get_score(importance_type="gain")
        d = pd.DataFrame({"variable": self.features,
                          "gain": [gain.get(f, 0.0) for f in self.features]})
        d["label"] = d.variable.map(lambda k: LABELS.get(k, k))
        d["gain_pct"] = 100 * d.gain / d.gain.sum()
        return d.sort_values("gain", ascending=False)


class ModelRF:
    def __init__(self, features=FEATURES):
        self.features = list(features)
        self.model = RandomForestClassifier(**RF_PARAMS)

    def fit(self, X, y):
        self.model.fit(X[self.features], y.astype(int))
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X[self.features])[:, 1]


# --------------------------------------------------------------------- metrics
def youden_threshold(y, p) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def evaluate(y, p, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "prevalence": float(y.mean()),
        "auc_roc": float(roc_auc_score(y, p)), "auc_pr": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)), "threshold": float(threshold),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "ppv": float(tp / (tp + fp)) if tp + fp else np.nan,
        "npv": float(tn / (tn + fn)) if tn + fn else np.nan,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


# --------------------------------------------------------------------- persistence
def save_bundle(path: Path, model_a: ModelA, model_b: ModelB, thresholds: dict, meta: dict):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "model_a.json").write_text(json.dumps(model_a.to_json(), indent=2), encoding="utf-8")
    model_b.model.save_model(path / "model_b_xgb.json")
    (path / "metadata.json").write_text(json.dumps(
        {**meta, "features_b": model_b.features, "thresholds": thresholds}, indent=2,
        ensure_ascii=False), encoding="utf-8")


def load_bundle(path: Path):
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    a = ModelA.from_json(json.loads((path / "model_a.json").read_text(encoding="utf-8")))
    b = ModelB(features=meta["features_b"])
    b.model.load_model(path / "model_b_xgb.json")
    return a, b, meta


__all__ = ["ModelA", "ModelACounts", "ModelB", "ModelRF", "evaluate", "youden_threshold",
           "save_bundle", "load_bundle", "joblib"]
