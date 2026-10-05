"""Training, temporal validation and forecasting pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import DEFAULT_TARGET, FEATURES, FEATURES_A, complete_rows, make_features
from .models import ModelA, ModelACounts, ModelB, ModelRF, evaluate, youden_threshold

RISK_BINS = [0, 0.15, 0.30, 0.50, 1.0001]
RISK_LABELS = ["Faible", "Modéré", "Élevé", "Très élevé"]


def load_inputs(path) -> pd.DataFrame:
    d = pd.read_csv(path)
    return make_features(d)


def risk_category(p) -> pd.Series:
    return pd.cut(pd.Series(p), RISK_BINS, labels=RISK_LABELS, right=False).astype(str)


def temporal_split(d: pd.DataFrame, target: str, test_start: int = 2023):
    m = complete_rows(d) & d[target].notna()
    dd = d[m]
    return dd[dd.year < test_start], dd[dd.year >= test_start]


def train_and_evaluate(d: pd.DataFrame, target: str = DEFAULT_TARGET, test_start: int = 2023):
    """Fit A, B, RF on years < test_start and evaluate on years >= test_start."""
    train, test = temporal_split(d, target, test_start)
    ytr, yte = train[target], test[target]
    a = ModelA().fit(train, ytr)
    b = ModelB().fit(train, ytr)
    rf = ModelRF().fit(train, ytr)
    p_tr = {"A": a.predict_proba(train), "B": b.predict_proba(train), "RF": rf.predict_proba(train)}
    p_te = {"A": a.predict_proba(test), "B": b.predict_proba(test), "RF": rf.predict_proba(test)}
    p_tr["Ensemble"] = (p_tr["A"] + p_tr["B"]) / 2
    p_te["Ensemble"] = (p_te["A"] + p_te["B"]) / 2
    # naive baseline: persistence (outbreak now -> outbreak next)
    p_te["Naïf"] = test["outbreak_now"].to_numpy().astype(float)
    p_tr["Naïf"] = train["outbreak_now"].to_numpy().astype(float)
    thresholds = {k: (0.5 if k == "Naïf" else youden_threshold(ytr, v)) for k, v in p_tr.items()}
    metrics = {k: evaluate(yte, v, thresholds[k]) for k, v in p_te.items()}
    preds = test[["iso3", "year", "month", "period", target]].copy()
    for k, v in p_te.items():
        preds[f"p_{k}"] = v
    return {"model_a": a, "model_b": b, "model_rf": rf, "metrics": metrics,
            "thresholds": thresholds, "predictions": preds, "n_train": len(train),
            "n_test": len(test), "train": train, "test": test}


def rolling_origin(d: pd.DataFrame, target: str = DEFAULT_TARGET, years=range(2018, 2026)):
    """Expanding-window validation: train on < Y, test on year Y."""
    from sklearn.metrics import roc_auc_score
    rows = []
    m = complete_rows(d) & d[target].notna()
    dd = d[m]
    for y in years:
        tr, te = dd[dd.year < y], dd[dd.year == y]
        if te[target].nunique() < 2:
            continue
        a = ModelA().fit(tr, tr[target])
        b = ModelB().fit(tr, tr[target])
        pa, pb = a.predict_proba(te), b.predict_proba(te)
        rows.append({"test_year": y, "n_test": len(te), "prevalence": te[target].mean(),
                     "auc_A": roc_auc_score(te[target], pa), "auc_B": roc_auc_score(te[target], pb),
                     "auc_Ensemble": roc_auc_score(te[target], (pa + pb) / 2)})
    return pd.DataFrame(rows)


def count_model(d: pd.DataFrame, test_start: int = 2023):
    feats = ModelACounts().features
    m = d[feats + ["cases", "cases_next1"]].notna().all(axis=1)
    dd = d[m]
    tr, te = dd[dd.year < test_start], dd[dd.year >= test_start]
    nb = ModelACounts().fit(tr, tr["cases_next1"])
    pred = nb.predict(te)
    mae = float(np.mean(np.abs(pred - te.cases_next1)))
    mae_naive = float(np.mean(np.abs(te.cases - te.cases_next1)))
    med_ae = float(np.median(np.abs(pred - te.cases_next1)))
    return nb, {"mae_nb": mae, "mae_persistence": mae_naive, "median_ae_nb": med_ae,
                "n_test": int(len(te))}


def fit_final(d: pd.DataFrame, target: str = DEFAULT_TARGET):
    m = complete_rows(d) & d[target].notna()
    dd = d[m]
    a = ModelA().fit(dd, dd[target])
    b = ModelB().fit(dd, dd[target])
    pa, pb = a.predict_proba(dd), b.predict_proba(dd)
    thr = {"A": youden_threshold(dd[target], pa), "B": youden_threshold(dd[target], pb),
           "Ensemble": youden_threshold(dd[target], (pa + pb) / 2)}
    return a, b, thr, len(dd)


CONSOLIDATION_LAG = 1   # months dropped at the end of the provisional series
INCOMPLETE_RATIO = 0.25  # last month < 25 % of the usual level -> flagged


def data_cutoff(d: pd.DataFrame) -> pd.Period:
    return pd.Period(d.loc[d["cases"].notna(), "period"].max(), freq="M")


def latest_rows(d: pd.DataFrame, consolidation_lag: int = CONSOLIDATION_LAG) -> pd.DataFrame:
    """Most recent consolidated month per country where the features are complete.

    WHO monthly data are provisional: the last month(s) are reported by few
    countries and are often incomplete, so the final `consolidation_lag`
    months of the series are not used as forecast origin.
    """
    origin_max = str(data_cutoff(d) - consolidation_lag)
    m = complete_rows(d) & (d["period"] <= origin_max)
    return d[m].sort_values(["iso3", "year", "month"]).groupby("iso3").tail(1)


def same_month_reference(d: pd.DataFrame) -> pd.Series:
    """Mean cases for the same calendar month over the 3 previous years."""
    ref = d[["iso3", "year", "month", "cases"]].copy()
    vals = []
    for k in (1, 2, 3):
        r = ref.copy()
        r["year"] = r["year"] + k
        vals.append(r.rename(columns={"cases": f"c{k}"}))
    m = d[["iso3", "year", "month"]]
    for v in vals:
        m = m.merge(v, on=["iso3", "year", "month"], how="left")
    return pd.Series(m[["c1", "c2", "c3"]].mean(axis=1).to_numpy(), index=d.index)


def forecast(d: pd.DataFrame, model_a, model_b, thresholds: dict, rows: pd.DataFrame | None = None):
    rows = latest_rows(d) if rows is None else rows
    out = rows[["iso3", "subregion", "period", "cases", "epidemic_threshold", "outbreak_now",
                "mcv1", "mcv2", "ivr_score"]].copy()
    usual = same_month_reference(d).reindex(rows.index)
    out["usual_same_month"] = usual.round(1)
    out["possibly_incomplete"] = ((usual >= 20) & (rows["cases"] < INCOMPLETE_RATIO * usual)).astype(int)
    out["p_A"] = model_a.predict_proba(rows)
    out["p_B"] = model_b.predict_proba(rows)
    out["p_ensemble"] = (out.p_A + out.p_B) / 2
    out["risk"] = risk_category(out.p_ensemble).values
    out["alert"] = (out.p_ensemble >= thresholds["Ensemble"]).astype(int)
    ref = pd.Period(d.loc[d["cases"].notna(), "period"].max(), freq="M")
    out["data_age_months"] = [(ref - pd.Period(p, freq="M")).n for p in out.period]
    out["stale"] = (out.data_age_months > 3).astype(int)
    return out.sort_values("p_ensemble", ascending=False).reset_index(drop=True)


__all__ = ["load_inputs", "train_and_evaluate", "rolling_origin", "count_model", "fit_final",
           "forecast", "latest_rows", "risk_category", "FEATURES", "FEATURES_A"]
