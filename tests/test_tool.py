"""Basic integrity tests for the prediction tool."""
from pathlib import Path

import numpy as np
import pandas as pd

from measles_predict.features import FEATURES, FEATURES_A, make_features
from measles_predict.models import load_bundle
from measles_predict.pipeline import forecast, latest_rows

REPO = Path(__file__).resolve().parents[1]


def _data():
    return make_features(pd.read_csv(REPO / "data" / "model_inputs.csv"))


def test_inputs_shape_and_keys():
    d = _data()
    assert not d.duplicated(["iso3", "year", "month"]).any()
    assert d.iso3.nunique() >= 50
    assert set(FEATURES_A) <= set(FEATURES) | {"log_population"}


def test_outbreak_definition():
    d = _data()
    m = d.outbreak.notna()
    expected = (d.loc[m, "cases"] > d.loc[m, "epidemic_threshold"]).astype(float)
    assert (expected.values == d.loc[m, "outbreak"].values).all()
    assert (d.epidemic_threshold.dropna() >= 10).all()


def test_models_predict_probabilities():
    d = _data()
    a, b, meta = load_bundle(REPO / "models")
    rows = latest_rows(d)
    for p in (a.predict_proba(rows), b.predict_proba(rows)):
        assert np.all((p >= 0) & (p <= 1))
    fc = forecast(d, a, b, meta["thresholds"])
    assert {"p_A", "p_B", "p_ensemble", "risk", "alert", "stale", "possibly_incomplete"} <= set(fc.columns)
    assert fc.p_ensemble.is_monotonic_decreasing


def test_more_cases_increase_model_a_risk():
    d = _data()
    a, _, _ = load_bundle(REPO / "models")
    row = latest_rows(d).head(1).copy()
    low = a.predict_proba(row)[0]
    row["cases"] = row["epidemic_threshold"] * 3
    row["outbreak"] = 1.0
    high = a.predict_proba(make_features(row))[0]
    assert high > low
