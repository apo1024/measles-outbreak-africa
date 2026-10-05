"""Export the data used by the static GitHub Pages dashboard (docs/).

    python scripts/export_pages.py
"""
from __future__ import annotations

import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from measles_predict.features import LABELS, make_features  # noqa: E402
from measles_predict.models import load_bundle  # noqa: E402
from measles_predict.pipeline import forecast  # noqa: E402

OUT = REPO / "docs" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def clean(o):
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    return o


def main():
    raw = pd.read_csv(REPO / "data" / "model_inputs.csv")
    d = make_features(raw)
    countries = pd.read_csv(REPO / "data" / "countries.csv")
    ivr = pd.read_csv(REPO / "data" / "vulnerability_index.csv")
    names = countries.set_index("iso3")["country_fr"].to_dict()
    ma, mb, meta = load_bundle(REPO / "models")
    fc = forecast(d, ma, mb, meta["thresholds"])
    fc["country"] = fc.iso3.map(names)
    fc.to_csv(REPO / "outputs" / "forecast_latest.csv", index=False)

    last_ivr = ivr[ivr.year == ivr.year.max()].set_index("iso3")
    recs = []
    for iso in countries.iso3:
        r = fc[fc.iso3 == iso]
        rec = {"iso3": iso, "country": names[iso],
               "subregion": countries.set_index("iso3").loc[iso, "subregion"],
               "ivr": clean(round(float(last_ivr.loc[iso, "ivr_score"]), 1)) if iso in last_ivr.index else None,
               "ivr_class": last_ivr.loc[iso, "ivr_class"] if iso in last_ivr.index else None}
        if len(r):
            r = r.iloc[0]
            rec.update({k: clean(float(r[k])) for k in ("p_A", "p_B", "p_ensemble", "cases",
                                                         "epidemic_threshold", "mcv1", "mcv2")})
            rec.update({"period": r.period, "risk": r.risk, "alert": int(r.alert), "stale": int(r.stale),
                        "possibly_incomplete": int(r.possibly_incomplete)})
        recs.append(rec)

    hist = {}
    for iso, g in raw.groupby("iso3"):
        hist[iso] = {"period": g.period.tolist(),
                     "cases": [clean(x) for x in g.cases.astype(float).tolist()],
                     "threshold": [clean(round(x, 1)) for x in g.epidemic_threshold.astype(float).tolist()],
                     "outbreak": [clean(x) for x in g.outbreak.astype(float).tolist()]}

    # latest raw row per country for the in-browser Model A calculator
    calc = {}
    for iso, g in d[d.cases.notna()].groupby("iso3"):
        r = g.iloc[-1]
        calc[iso] = {f: clean(float(r[f])) for f in ma.features}
        calc[iso].update({"epidemic_threshold": clean(float(r.epidemic_threshold)), "period": r.period,
                          "cases": clean(float(r.cases))})

    payload = {"meta": {**meta, "n_countries": len(countries), "n_alerts": int(fc.alert.sum())},
               "forecast": recs}
    (OUT / "forecast.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (OUT / "history.json").write_text(json.dumps(hist), encoding="utf-8")
    model_a = json.loads((REPO / "models" / "model_a.json").read_text(encoding="utf-8"))
    model_a["labels"] = {k: LABELS.get(k, k) for k in model_a["features"]}
    (OUT / "model_a.json").write_text(json.dumps(model_a, ensure_ascii=False), encoding="utf-8")
    (OUT / "calculator_inputs.json").write_text(json.dumps(calc), encoding="utf-8")
    shutil.copy(REPO / "data" / "africa_adm0_simplified.geojson", OUT / "africa_adm0.geojson")
    print(f"Pages data exported to {OUT}")


if __name__ == "__main__":
    main()
