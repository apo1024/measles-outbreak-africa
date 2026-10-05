"""Command-line interface.

    python -m measles_predict forecast            # latest outbreak-risk forecast
    python -m measles_predict country NGA         # history + forecast for one country
    python -m measles_predict scenario NGA --cases 800 --mcv1 60
    python -m measles_predict evaluate            # temporal validation of models A and B
    python -m measles_predict train               # refit on all data and save to models/
    python -m measles_predict cross --x ivr_score --y incidence_pm --years 2023-2025
    python -m measles_predict boundaries NER --level ADM2 --out niger_districts.kml
    python -m measles_predict districts data/exemples/EXEMPLE_SIMULE_districts_NER.csv --iso3 NER
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


def _years(txt: str):
    if "-" in txt:
        a, b = txt.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(txt)]


def cmd_cross(a):
    from . import cross
    panel = cross.build_panel()
    view = cross.period_view(panel, _years(a.years))
    if a.x:
        r = cross.correlate(view, a.x, a.y, a.logx, a.logy)
        print(f"{cross.label(a.x)} × {cross.label(a.y)} – années {a.years} – n = {r.get('n')} pays")
        if r.get("n", 0) >= 4:
            print(f"  Spearman rho = {r['spearman_rho']:.3f} (p = {r['spearman_p']:.4f})")
            print(f"  Pearson  r   = {r['pearson_r']:.3f} (p = {r['pearson_p']:.4f}) ; R² = {r['r2']:.3f}")
    if a.row:
        t, test = cross.crosstab(view, a.row, a.col, a.value)
        print()
        print(f"Tableau croisé : {cross.label(a.row)} × {cross.label(a.col)}")
        print(t.round(2).to_string())
        if test:
            print(f"Chi² = {test['chi2']:.2f}, ddl = {test['dof']}, p = {test['p']:.4f}, V de Cramér = {test['cramers_v']:.3f}")
    if a.list:
        for k, (lab, kind) in cross.VARIABLES.items():
            print(f"  {k:28s} {kind:4s} {lab}")
    if a.out:
        view.to_csv(a.out, index=False)
        print(f"Tableau des pays enregistré : {a.out}")


def cmd_boundaries(a):
    from . import subnational as sn
    iso = a.iso3.upper()
    if a.level is None:
        inv = sn.inventory()
        print(inv[inv.iso3 == iso][["level", "release", "n_units", "source", "license"]].to_string(index=False))
        return
    if a.level == "LOCALITES":
        g = sn.load_localities(iso).rename(columns={"name": "nom"})
        data = sn.to_kml_bytes(g, "nom")
    else:
        g = sn.load_boundaries(iso, a.level)
        data = sn.to_kml_bytes(g)
    out = Path(a.out or f"{iso}_{a.level}.kml")
    if out.suffix.lower() == ".geojson":
        out.write_text(g.to_json(), encoding="utf-8")
    else:
        out.write_bytes(data)
    print(f"{len(g)} unités -> {out}")


def cmd_districts(a):
    from . import subnational as sn
    df = pd.read_csv(a.file, sep=None, engine="python")
    units = sn.load_boundaries(a.iso3.upper(), a.level)
    joined, m = sn.join_data(units.drop(columns="geometry"), df, a.name_col, a.cutoff)
    print(f"Rapprochement des noms : {m.match.notna().sum()}/{len(m)} trouvés")
    if m.match.isna().any():
        print("  Non trouvés :", ", ".join(m.loc[m.match.isna(), "source"].head(20)))
    allp, last = sn.district_surveillance(joined[joined.unit_name_matched.notna()], "unit_name_matched",
                                          a.period_col, a.cases_col, a.pop_col, window=a.window,
                                          k_sd=a.k_sd, min_cases=a.min_cases)
    print(last.status.value_counts().to_string())
    cols = ["unit", "period", "cases", "threshold", "ratio_to_threshold", "status"]
    print(last[cols].head(a.top).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    if a.out:
        last.to_csv(a.out, index=False)
        print(f"Alertes enregistrées : {a.out}")


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
    x = sp.add_parser("cross", help="analyses croisées entre pays")
    x.add_argument("--years", default="2023-2025", help="année (2025) ou période (2023-2025)")
    x.add_argument("--x")
    x.add_argument("--y")
    x.add_argument("--logx", action="store_true")
    x.add_argument("--logy", action="store_true")
    x.add_argument("--row", help="variable catégorielle en lignes (ex. ivr_class)")
    x.add_argument("--col", help="variable catégorielle en colonnes (ex. incidence_class)")
    x.add_argument("--value", help="variable numérique à moyenner dans les cellules")
    x.add_argument("--list", action="store_true", help="liste des variables disponibles")
    x.add_argument("--out", help="CSV du tableau des pays")
    x.set_defaults(func=cmd_cross)
    b = sp.add_parser("boundaries", help="limites infranationales (KML/GeoJSON)")
    b.add_argument("iso3")
    b.add_argument("--level", choices=["ADM1", "ADM2", "ADM3", "ADM4", "LOCALITES"])
    b.add_argument("--out")
    b.set_defaults(func=cmd_boundaries)
    ds = sp.add_parser("districts", help="surveillance par district à partir d'un fichier CSV")
    ds.add_argument("file")
    ds.add_argument("--iso3", required=True)
    ds.add_argument("--level", default="ADM2")
    ds.add_argument("--name-col", dest="name_col", default="district")
    ds.add_argument("--period-col", dest="period_col", default="period")
    ds.add_argument("--cases-col", dest="cases_col", default="cases")
    ds.add_argument("--pop-col", dest="pop_col", default=None)
    ds.add_argument("--window", type=int, default=24)
    ds.add_argument("--k-sd", dest="k_sd", type=float, default=2.0)
    ds.add_argument("--min-cases", dest="min_cases", type=int, default=5)
    ds.add_argument("--cutoff", type=float, default=0.82)
    ds.add_argument("--top", type=int, default=15)
    ds.add_argument("--out")
    ds.set_defaults(func=cmd_districts)
    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
