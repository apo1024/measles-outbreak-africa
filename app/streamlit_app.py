"""Outil de surveillance et de prédiction des flambées de rougeole en Afrique.

Lancer :  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from measles_predict.features import LABELS, make_features  # noqa: E402
from measles_predict.models import load_bundle  # noqa: E402
from measles_predict.pipeline import RISK_LABELS, forecast, latest_rows  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tab_cross import render_cross  # noqa: E402
from tab_districts import render_districts  # noqa: E402

st.set_page_config(page_title="Rougeole Afrique – Prédiction des flambées", page_icon="🩺", layout="wide")
RISK_COLORS = {"Faible": "#7fbf7b", "Modéré": "#fec44f", "Élevé": "#fe9929", "Très élevé": "#cc4c02"}


def files_version(*paths) -> tuple:
    """Cache key that changes whenever a data or model file is updated (e.g. after a git pull
    on Streamlit Cloud or the monthly refresh), so cached objects are never stale."""
    files = [p for path in paths for p in (sorted(path.glob("*")) if path.is_dir() else [path])]
    return tuple((f.name, f.stat().st_mtime_ns, f.stat().st_size) for f in files if f.is_file())


DATA_FILES = [REPO / "data" / f for f in ("model_inputs.csv", "countries.csv", "vulnerability_index.csv",
                                          "africa_adm0_simplified.geojson")]


@st.cache_data
def load_data(version: tuple):
    raw = pd.read_csv(REPO / "data" / "model_inputs.csv")
    countries = pd.read_csv(REPO / "data" / "countries.csv")
    ivr = pd.read_csv(REPO / "data" / "vulnerability_index.csv")
    geo = json.loads((REPO / "data" / "africa_adm0_simplified.geojson").read_text(encoding="utf-8"))
    return raw, make_features(raw), countries, ivr, geo


@st.cache_resource
def load_models(version: tuple):
    return load_bundle(REPO / "models")


raw, d, countries, ivr, geo = load_data(files_version(*DATA_FILES))
model_a, model_b, meta = load_models(files_version(REPO / "models"))
NAMES = countries.set_index("iso3")["country_fr"].to_dict()
fc = forecast(d, model_a, model_b, meta["thresholds"])
fc.insert(1, "Pays", fc.iso3.map(NAMES))

st.title("🩺 Rougeole en Afrique – surveillance et prédiction des flambées")
st.caption(f"Données OMS jusqu'à {meta['data_until']} · modèles entraînés le {meta['created']} · "
           "Auteur : Miracle Destine Apollon (info@idreamlore.com)")

tab1, tab2, tab_x, tab_dist, tab3, tab4, tab5 = st.tabs([
    "🗺️ Tableau de bord", "📈 Profil pays", "🔀 Analyses croisées", "📍 Districts & localités",
    "🧪 Simulateur", "✅ Performance", "📘 Méthodes"])

with tab_x:
    render_cross(fc, geo, NAMES)

with tab_dist:
    render_districts(NAMES)

# ------------------------------------------------------------------ dashboard
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pays suivis", len(fc))
    c2.metric("Pays en alerte", int(fc.alert.sum()))
    c3.metric("Risque élevé / très élevé", int(fc.risk.isin(["Élevé", "Très élevé"]).sum()))
    c4.metric("Seuil d'alerte (ensemble)", f"{100 * meta['thresholds']['Ensemble']:.1f} %")

    layer = st.radio("Carte", ["Probabilité de flambée (3 mois)", "Indice de vulnérabilité (IVR)",
                               "Couverture MCV1", "Incidence 12 derniers mois"], horizontal=True)
    mp = fc[["iso3", "Pays", "p_ensemble", "risk", "period"]].copy()
    last_year = int(ivr.year.max())
    mp = mp.merge(ivr[ivr.year == last_year][["iso3", "ivr_score", "ivr_class"]], on="iso3", how="outer")
    mp = mp.merge(raw.sort_values("period").groupby("iso3").tail(1)[["iso3", "mcv1"]], on="iso3", how="left")
    inc = raw[raw.cases.notna()].groupby("iso3").tail(12).groupby("iso3").agg(
        cases12=("cases", "sum"), lp=("log_population", "last"))
    inc["incid12"] = 1e6 * inc.cases12 / pd.Series(2.718281828 ** inc.lp, index=inc.index)
    mp = mp.merge(inc[["incid12"]].reset_index(), on="iso3", how="left")
    mp["Pays"] = mp.iso3.map(NAMES)
    cfg = {
        "Probabilité de flambée (3 mois)": ("p_ensemble", "OrRd", [0, 0.5]),
        "Indice de vulnérabilité (IVR)": ("ivr_score", "YlOrRd", [0, 90]),
        "Couverture MCV1": ("mcv1", "RdYlGn", [40, 100]),
        "Incidence 12 derniers mois": ("incid12", "Reds", [0, 200]),
    }
    col, scale, rng = cfg[layer]
    fig = px.choropleth(mp, geojson=geo, locations="iso3", featureidkey="properties.iso3", color=col,
                        color_continuous_scale=scale, range_color=rng, hover_name="Pays",
                        hover_data={"iso3": False, "p_ensemble": ":.2f", "ivr_score": ":.1f",
                                    "mcv1": ":.0f", "incid12": ":.1f", "period": True})
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Classement des pays – risque de flambée dans les 3 mois suivants")
    show = fc[["Pays", "iso3", "period", "cases", "epidemic_threshold", "p_A", "p_B", "p_ensemble",
               "risk", "alert", "stale", "possibly_incomplete", "ivr_score", "mcv1", "mcv2"]].rename(columns={
        "period": "Dernier mois", "cases": "Cas", "epidemic_threshold": "Seuil épidémique",
        "p_A": "P(A)", "p_B": "P(B)", "p_ensemble": "P(ensemble)", "risk": "Risque", "alert": "Alerte",
        "stale": "Données > 3 mois", "possibly_incomplete": "Possiblement incomplet", "ivr_score": "IVR"})
    st.dataframe(show.style.format({"P(A)": "{:.1%}", "P(B)": "{:.1%}", "P(ensemble)": "{:.1%}",
                                    "Seuil épidémique": "{:.0f}", "IVR": "{:.1f}", "Cas": "{:.0f}"}),
                 width="stretch", height=480)
    st.download_button("Télécharger la prévision (CSV)", fc.to_csv(index=False).encode("utf-8"),
                       "prevision_flambees_rougeole.csv", "text/csv")
    st.info("Une flambée est définie comme un mois où les cas dépassent le seuil épidémique du pays "
            "(max(10, moyenne + 2 écarts-types des 24 mois précédents)). Les derniers mois de données OMS "
            "sont provisoires : vérifier les colonnes « Données > 3 mois » et « Possiblement incomplet ».")

# ------------------------------------------------------------------ country profile
with tab2:
    opts = sorted(raw.iso3.unique(), key=lambda i: NAMES.get(i, i))
    iso = st.selectbox("Pays", opts, index=opts.index("NGA") if "NGA" in opts else 0,
                       format_func=lambda i: f"{NAMES.get(i, i)} ({i})")
    sub = d[d.iso3 == iso].copy()
    sub["date"] = pd.to_datetime(sub.period)
    fig = go.Figure()
    fig.add_bar(x=sub.date, y=sub.cases, name="Cas mensuels", marker_color="#2a78b5")
    fig.add_scatter(x=sub.date, y=sub.epidemic_threshold, name="Seuil épidémique", line=dict(color="#d9534f", dash="dash"))
    ob = sub[sub.outbreak == 1]
    fig.add_scatter(x=ob.date, y=ob.cases, mode="markers", name="Flambée", marker=dict(color="#d9534f", size=8, symbol="x"))
    fig.update_layout(height=420, title=f"Courbe épidémique – {NAMES.get(iso, iso)}", yaxis_title="Cas",
                      legend=dict(orientation="h"), margin=dict(t=50))
    st.plotly_chart(fig, width="stretch")

    hist = sub[sub[model_b.features].notna().all(axis=1)].copy()
    if len(hist):
        hist["P(A)"] = model_a.predict_proba(hist)
        hist["P(B)"] = model_b.predict_proba(hist)
        hist["P(ensemble)"] = (hist["P(A)"] + hist["P(B)"]) / 2
        f2 = px.line(hist, x="date", y=["P(A)", "P(B)", "P(ensemble)"],
                     title="Probabilité prédite de flambée dans les 3 mois (rétrospectif, données d'apprentissage)")
        f2.add_hline(y=meta["thresholds"]["Ensemble"], line_dash="dot", annotation_text="seuil d'alerte")
        f2.update_layout(height=330, legend_title=None, yaxis_tickformat=".0%")
        st.plotly_chart(f2, width="stretch")

    c1, c2 = st.columns(2)
    iv = ivr[ivr.iso3 == iso].sort_values("year")
    f3 = px.line(iv, x="year", y="ivr_score", markers=True, title="Indice de vulnérabilité (IVR)")
    c1.plotly_chart(f3, width="stretch")
    last = iv.tail(1).melt(id_vars=["iso3", "year"], value_vars=["d_immunite", "d_demographie", "d_socioeco",
                                                                  "d_nutrition", "d_acces_soins"])
    last["variable"] = last.variable.map({"d_immunite": "Déficit vaccinal", "d_demographie": "Démographie",
                                          "d_socioeco": "Pauvreté", "d_nutrition": "Malnutrition",
                                          "d_acces_soins": "Accès aux soins"})
    f4 = px.bar(last, x="value", y="variable", orientation="h", range_x=[0, 1],
                title=f"Domaines de vulnérabilité {int(iv.year.max())} (0 = favorable, 1 = défavorable)")
    c2.plotly_chart(f4, width="stretch")

# ------------------------------------------------------------------ simulator
with tab3:
    st.markdown("Modifiez la situation du **dernier mois consolidé** d'un pays et observez l'effet sur la "
                "probabilité de flambée prédite par les modèles A et B.")
    iso_s = st.selectbox("Pays ", opts, index=opts.index("NGA") if "NGA" in opts else 0,
                         format_func=lambda i: f"{NAMES.get(i, i)} ({i})", key="sim")
    base = latest_rows(d[d.iso3 == iso_s])
    if base.empty:
        st.warning("Données insuffisantes pour ce pays.")
    else:
        b = raw.loc[base.index].copy()
        thr = float(b.epidemic_threshold.iloc[0])
        st.write(f"Dernier mois consolidé : **{b.period.iloc[0]}** · cas = {b.cases.iloc[0]:.0f} · "
                 f"seuil épidémique = {thr:.0f}")
        c1, c2, c3, c4 = st.columns(4)
        cases = c1.number_input("Cas ce mois", 0, 200000, int(b.cases.iloc[0]), step=10)
        mcv1 = c2.slider("MCV1 (%)", 0, 100, int(b.mcv1.iloc[0]))
        mcv2 = c3.slider("MCV2 (%)", 0, 100, int(b.mcv2.iloc[0]))
        ivr_s = c4.slider("IVR", 0.0, 100.0, float(round(b.ivr_score.iloc[0], 1)))
        s = b.copy()
        s["cases"], s["mcv1"], s["mcv2"], s["ivr_score"] = cases, mcv1, mcv2, ivr_s
        s["outbreak"] = float(cases > thr)
        before = forecast(d, model_a, model_b, meta["thresholds"], rows=make_features(b))
        after = forecast(d, model_a, model_b, meta["thresholds"], rows=make_features(s))
        r1, r2, r3 = st.columns(3)
        r1.metric("Modèle A (logistique)", f"{after.p_A[0]:.1%}", f"{100 * (after.p_A[0] - before.p_A[0]):+.1f} pts")
        r2.metric("Modèle B (XGBoost)", f"{after.p_B[0]:.1%}", f"{100 * (after.p_B[0] - before.p_B[0]):+.1f} pts")
        r3.metric("Ensemble A+B", f"{after.p_ensemble[0]:.1%}",
                  f"{100 * (after.p_ensemble[0] - before.p_ensemble[0]):+.1f} pts")
        st.markdown(f"**Niveau de risque : {after.risk[0]}** · "
                    f"{'🚨 ALERTE' if after.alert[0] else 'pas d’alerte'}")

# ------------------------------------------------------------------ performance
with tab4:
    st.markdown("Validation temporelle : apprentissage sur 2013-2022, test sur 2023-2026 (données jamais vues).")
    perf = REPO / "outputs" / "performance_outbreak_next3.csv"
    if perf.exists():
        p = pd.read_csv(perf, index_col=0)
        st.dataframe(p[["auc_roc", "auc_pr", "brier", "sensitivity", "specificity", "ppv", "npv", "n"]]
                     .rename(columns={"auc_roc": "AUC-ROC", "auc_pr": "AUC-PR", "brier": "Brier",
                                      "sensitivity": "Sensibilité", "specificity": "Spécificité",
                                      "ppv": "VPP", "npv": "VPN"}).round(3), width="stretch")
    ro = REPO / "outputs" / "rolling_origin_outbreak_next3.csv"
    if ro.exists():
        r = pd.read_csv(ro)
        st.plotly_chart(px.line(r, x="test_year", y=["auc_A", "auc_B", "auc_Ensemble"], markers=True,
                                title="AUC par année – validation à origine glissante", range_y=[0.5, 1]),
                        width="stretch")
    a = pd.DataFrame({"variable": list(model_a.params_), "coef": list(model_a.params_.values())})
    a = a[a.variable != "const"]
    a["OR par écart-type"] = (2.718281828 ** a.coef).round(3)
    a["Variable"] = a.variable.map(lambda k: LABELS.get(k, k))
    st.subheader("Modèle A – odds ratios (par écart-type)")
    st.dataframe(a[["Variable", "OR par écart-type"]], width="stretch")
    imp = model_b.importance().head(12)
    st.plotly_chart(px.bar(imp.iloc[::-1], x="gain_pct", y="label", orientation="h",
                           title="Modèle B – importance des variables (% du gain)"), width="stretch")

# ------------------------------------------------------------------ methods
with tab5:
    st.markdown((REPO / "docs" / "METHODES.md").read_text(encoding="utf-8")
                if (REPO / "docs" / "METHODES.md").exists() else "Voir README.md")
