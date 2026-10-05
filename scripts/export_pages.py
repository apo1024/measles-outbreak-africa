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
from measles_predict.pipeline import forecast, latest_rows  # noqa: E402

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

    ok = d[mb.features].notna().all(axis=1)
    p_hist = pd.Series(np.nan, index=d.index)
    p_hist[ok] = (ma.predict_proba(d[ok]) + mb.predict_proba(d[ok])) / 2
    hist = {}
    for iso, g in raw.groupby("iso3"):
        hist[iso] = {"period": g.period.tolist(),
                     "cases": [clean(x) for x in g.cases.astype(float).tolist()],
                     "threshold": [clean(round(x, 1)) for x in g.epidemic_threshold.astype(float).tolist()],
                     "outbreak": [clean(x) for x in g.outbreak.astype(float).tolist()],
                     "p": [clean(round(float(x), 4)) for x in p_hist.loc[g.index].tolist()]}

    # latest raw row per country for the in-browser Model A calculator
    calc = {}
    feats = list(dict.fromkeys(list(ma.features) + list(mb.features)))
    for _, r in latest_rows(d).iterrows():
        calc[r.iso3] = {f: clean(float(r[f])) for f in feats}
        calc[r.iso3].update({"epidemic_threshold": clean(float(r.epidemic_threshold)), "period": r.period,
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
    export_cross(fc)
    export_subnational()
    export_model_b(mb)
    export_performance(ma, mb)
    print(f"Pages data exported to {OUT}")


def export_cross(fc):
    """Country x year panel for the in-browser cross-analysis."""
    from measles_predict import cross
    panel = cross.build_panel(forecast=fc[["iso3", "p_ensemble", "risk"]])
    keep = ["iso3", "country_fr", "subregion", "year"] + [v for v in cross.VARIABLES if v in panel and v not in ("subregion",)]
    rows = panel[keep].copy()
    recs = [{k: clean(v) if isinstance(v, float) else v for k, v in r.items()} for r in rows.to_dict("records")]
    meta = {k: {"label": lab, "kind": kind} for k, (lab, kind) in cross.VARIABLES.items() if k in keep or k == "subregion"}
    (OUT / "panel.json").write_text(json.dumps({"variables": meta, "order": cross.CAT_ORDER, "rows": recs},
                                               ensure_ascii=False), encoding="utf-8")


def export_subnational():
    """Districts (ADM2, simplified) and localities per country, plus the inventory of ADM1-ADM4 sources."""
    from measles_predict import subnational as sn
    inv = sn.inventory()
    geo_dir = REPO / "docs" / "geo"
    (geo_dir / "ADM2").mkdir(parents=True, exist_ok=True)
    (geo_dir / "localites").mkdir(parents=True, exist_ok=True)
    for iso in inv.loc[inv.level == "ADM2", "iso3"].unique():
        f = geo_dir / "ADM2" / f"{iso}.geojson"
        if f.exists():
            continue
        try:
            g = sn.load_boundaries(iso, "ADM2")
            g["geometry"] = g.geometry.simplify(0.002, preserve_topology=True)
            g.to_file(f, driver="GeoJSON", COORDINATE_PRECISION=4)
        except Exception as e:  # noqa: BLE001
            print("ADM2", iso, e)
    loc = pd.read_csv(sn.LOCALITIES)
    for iso, g in loc.groupby("iso3"):
        (geo_dir / "localites" / f"{iso}.json").write_text(
            g[["name", "population", "lat", "lon"]].to_json(orient="values"), encoding="utf-8")
    inv.to_json(OUT / "boundaries_inventory.json", orient="records", force_ascii=False)


def export_model_b(mb):
    """Compact XGBoost model evaluated in the browser (same trees, same predictions)."""
    booster = mb.model.get_booster()
    js = json.loads(booster.save_raw("json").decode("utf-8"))
    learner = js["learner"]
    base = float(str(learner["learner_model_param"]["base_score"]).strip("[]"))
    trees = []
    for t in learner["gradient_booster"]["model"]["trees"]:
        trees.append([t["left_children"], t["right_children"], t["split_indices"],
                      [float(f"{v:.9g}") for v in t["split_conditions"]], t["default_left"]])
    out = {"features": mb.features, "base_margin": float(np.log(base / (1 - base))), "trees": trees}
    (OUT / "model_b.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")


def export_performance(ma, mb):
    perf = {}
    f = REPO / "outputs" / "performance_outbreak_next3.csv"
    if f.exists():
        perf["metrics"] = pd.read_csv(f).rename(columns={"Unnamed: 0": "modele"}).round(4).to_dict("records")
    f = REPO / "outputs" / "rolling_origin_outbreak_next3.csv"
    if f.exists():
        perf["rolling"] = pd.read_csv(f).round(4).to_dict("records")
    perf["odds_ratios"] = [{"variable": k, "label": LABELS.get(k, k), "or_per_sd": round(float(np.exp(v)), 3)}
                           for k, v in ma.params_.items() if k != "const"]
    imp = mb.importance().head(12)
    perf["importance"] = [{"label": r.label, "gain_pct": round(float(r.gain_pct), 2)} for r in imp.itertuples()]
    (OUT / "performance.json").write_text(json.dumps(perf, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
