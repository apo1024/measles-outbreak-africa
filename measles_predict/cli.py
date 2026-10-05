"""Command-line interface.

    python -m measles_predict forecast            # latest outbreak-risk forecast
    python -m measles_predict country NGA         # history + forecast for one country
    python -m measles_predict scenario NGA --cases 800 --mcv1 60
    python -m measles_predict evaluate            # temporal validation of models A and B
    python -m measles_predict train               # refit on all data and save to models/
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

from .models import load_bundle, save_bundle
from .pipeline import (fit_final, forecast, latest_rows, load_inputs, rolling_origin,
                       train_and_evaluate)

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "model_inputs.csv"
MODELS = REPO / "models"
OUTPUTS = REPO / "outputs"


def _names():
    c = pd.read_csv(REPO / "data" / "countries.csv")
    return c.set_index("iso3")["country_fr"].to_dict()


def cmd_forecast(a):
    d = load_inputs(a.data)
    ma, mb, meta = load_bundle(a.models)
    fc = forecast(d, ma, mb, meta["thresholds"])
    fc.insert(1, "country", fc.iso3.map(_names()))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fc.to_csv(out, index=False)
    cols = ["iso3", "country", "period", "p_A", "p_B", "p_ensemble", "risk", "alert", "stale", "possibly_incomplete"]
    print(f"Probabilité de flambée dans les 3 mois suivant le dernier mois notifié "
          f"(seuil d'alerte ensemble = {meta['thresholds']['Ensemble']:.3f})\n")
    print(fc[cols].head(a.top).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n{int(fc.alert.sum())} pays en alerte. Résultats complets : {out}")


def cmd_country(a):
    d = load_inputs(a.data)
    ma, mb, meta = load_bundle(a.models)
    iso = a.iso3.upper()
    sub = d[d.iso3 == iso]
    if sub.empty:
        sys.exit(f"Pays inconnu ou sans données : {iso}")
    print(f"{_names().get(iso, iso)} ({iso}) – 12 derniers mois")
    cols = ["period", "cases", "epidemic_threshold", "outbreak", "mcv1", "mcv2", "ivr_score"]
    print(sub[cols].tail(12).to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    fc = forecast(d, ma, mb, meta["thresholds"], rows=latest_rows(sub))
    print("\nPrévision :", fc[["period", "p_A", "p_B", "p_ensemble", "risk", "alert"]].to_dict("records")[0])


def cmd_scenario(a):
    """What-if analysis: modify the latest month of a country and re-predict."""
    d = load_inputs(a.data)
    ma, mb, meta = load_bundle(a.models)
    iso = a.iso3.upper()
    raw = pd.read_csv(a.data)
    row = raw[(raw.iso3 == iso) & raw.cases.notna()].tail(1).copy()
    if row.empty:
        sys.exit(f"Pays inconnu ou sans données : {iso}")
    base = forecast(d, ma, mb, meta["thresholds"], rows=load_inputs_from(row))
    if a.cases is not None:
        row["cases"] = a.cases
        row["outbreak"] = float(a.cases > row["epidemic_threshold"].iloc[0])
    for k in ("mcv1", "mcv2", "ivr_score"):
        if getattr(a, k) is not None:
            row[k] = getattr(a, k)
    new = forecast(d, ma, mb, meta["thresholds"], rows=load_inputs_from(row))
    show = ["period", "cases", "mcv1", "mcv2", "ivr_score", "p_A", "p_B", "p_ensemble", "risk"]
    print("Situation observée :\n", base[show].to_string(index=False))
    print("Scénario :\n", new[show].to_string(index=False))


def load_inputs_from(rows: pd.DataFrame) -> pd.DataFrame:
    from .features import make_features
    return make_features(rows)


def cmd_evaluate(a):
    d = load_inputs(a.data)
    r = train_and_evaluate(d, a.target, test_start=a.test_start)
    met = pd.DataFrame(r["metrics"]).T
    print(f"Cible : {a.target} | apprentissage < {a.test_start} (n={r['n_train']}) | test (n={r['n_test']})")
    print(met[["auc_roc", "auc_pr", "brier", "sensitivity", "specificity", "ppv"]].round(3).to_string())
    OUTPUTS.mkdir(exist_ok=True)
    met.to_csv(OUTPUTS / f"performance_{a.target}.csv")
    if a.rolling:
        ro = rolling_origin(d, a.target)
        print("\nValidation à origine glissante :\n", ro.round(3).to_string(index=False))
        ro.to_csv(OUTPUTS / f"rolling_origin_{a.target}.csv", index=False)


def cmd_train(a):
    d = load_inputs(a.data)
    ma, mb, thr, n = fit_final(d, "outbreak_next3")
    last = d.loc[d.cases.notna(), "period"].max()
    save_bundle(a.models, ma, mb, thr, {"target": "outbreak_next3", "trained_on_rows": n,
                                        "data_until": last,
                                        "outbreak_definition": "cases(t) > max(10, mean + 2 SD of previous 24 months)",
                                        "created": pd.Timestamp.now().strftime("%Y-%m-%d")})
    print(f"Modèles réentraînés sur {n} lignes (données jusqu'à {last}). Seuils : {json.dumps(thr)}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="measles_predict",
                                description="Surveillance et prédiction des flambées de rougeole en Afrique")
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--models", default=str(MODELS))
    sp = p.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("forecast", help="prévision pour tous les pays")
    f.add_argument("--out", default=str(OUTPUTS / "forecast_latest.csv"))
    f.add_argument("--top", type=int, default=20)
    f.set_defaults(func=cmd_forecast)
    c = sp.add_parser("country", help="historique et prévision d'un pays")
    c.add_argument("iso3")
    c.set_defaults(func=cmd_country)
    s = sp.add_parser("scenario", help="simulation « et si ? »")
    s.add_argument("iso3")
    s.add_argument("--cases", type=float)
    s.add_argument("--mcv1", type=float)
    s.add_argument("--mcv2", type=float)
    s.add_argument("--ivr_score", type=float)
    s.set_defaults(func=cmd_scenario)
    e = sp.add_parser("evaluate", help="validation temporelle")
    e.add_argument("--target", default="outbreak_next3", choices=["outbreak_next1", "outbreak_next3"])
    e.add_argument("--test-start", type=int, default=2023)
    e.add_argument("--rolling", action="store_true")
    e.set_defaults(func=cmd_evaluate)
    t = sp.add_parser("train", help="réentraîner et sauvegarder les modèles")
    t.set_defaults(func=cmd_train)
    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
