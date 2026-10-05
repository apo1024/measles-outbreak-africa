"""Tests of the cross-analysis and subnational modules (offline)."""
import numpy as np
import pandas as pd

from measles_predict import cross, subnational as sn


def test_panel_and_period_view():
    p = cross.build_panel()
    assert {"incidence_pm", "mcv1", "ivr_score", "ivr_class"} <= set(p.columns)
    v = cross.period_view(p, [2023, 2024, 2025])
    assert v.iso3.is_unique and len(v) == 54


def test_correlation_and_crosstab():
    v = cross.period_view(cross.build_panel(), range(2023, 2026))
    r = cross.correlate(v, "mcv1", "incidence_pm", logy=True)
    assert r["n"] > 40 and r["spearman_rho"] < 0          # more coverage, less measles
    t, test = cross.crosstab(v, "ivr_class", "incidence_class")
    assert int(t.values.sum()) == test["n"] and 0 <= test["cramers_v"] <= 1
    assert cross.bivariate_classes(v, "ivr_score", "incidence_pm").dropna().str.match(r"[123]-[123]").all()


def test_name_matching():
    m = sn.match_names(["DS Tillabéri", "NIAMEY", "Zone de santé de Dosso", "Xyzabc"], ["Tillabéri", "Niamey", "Dosso", "Maradi"])
    got = dict(zip(m.source, m.match))
    assert got["DS Tillabéri"] == "Tillabéri" and got["NIAMEY"] == "Niamey" and got["Zone de santé de Dosso"] == "Dosso"
    assert pd.isna(got["Xyzabc"])


def test_district_surveillance_flags_spike():
    periods = pd.period_range("2024-01", "2026-06", freq="M").astype(str)
    rng = np.random.default_rng(0)
    rows = [{"d": u, "p": p, "c": int(rng.poisson(4))} for u in ("A", "B") for p in periods]
    rows[-1]["c"] = 60                                     # spike in district B, last month
    allp, last = sn.district_surveillance(pd.DataFrame(rows), "d", "p", "c")
    st = last.set_index("unit").status
    assert st["B"] == "Flambée" and st["A"] != "Flambée"


def test_localities_and_inventory():
    loc = sn.load_localities("NER")
    assert len(loc) > 20 and loc.crs.to_epsg() == 4326
    inv = sn.inventory()
    assert inv[inv.level == "ADM2"].iso3.nunique() >= 50
