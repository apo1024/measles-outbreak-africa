"""Onglet « Analyses croisées »."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from measles_predict import cross

SR_COLORS = {"Afrique de l'Ouest": "#2a78b5", "Afrique centrale": "#d9534f", "Afrique de l'Est": "#e8a33d",
             "Afrique australe": "#3a9e6f", "Afrique du Nord": "#8064a2"}


@st.cache_data
def _panel(fc_small: pd.DataFrame, version: tuple) -> pd.DataFrame:
    return cross.build_panel(forecast=fc_small)


def _data_version() -> tuple:
    return tuple((f.name, f.stat().st_mtime_ns) for f in sorted(cross.DATA.glob("*.csv")))


def _fmt_p(p):
    return "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


def render_cross(fc: pd.DataFrame, geo: dict, names: dict):
    panel = _panel(fc[["iso3", "p_ensemble", "risk"]].copy(), _data_version())
    st.markdown("Croisez n'importe quelle variable épidémiologique, vaccinale, de vulnérabilité, "
                "socio-économique ou de prédiction. Unité d'analyse : le pays (une année ou une moyenne sur plusieurs années).")

    years = sorted(panel.year.unique())
    c1, c2, c3 = st.columns([2, 2, 3])
    mode = c1.radio("Période", ["Une année", "Moyenne sur plusieurs années"], horizontal=False)
    if mode == "Une année":
        y = c2.selectbox("Année", years[::-1], index=1 if len(years) > 1 else 0)
        sel_years = [y]
    else:
        y0, y1 = c2.select_slider("Années", options=years, value=(2023, 2025) if 2025 in years else (years[0], years[-1]))
        sel_years = list(range(y0, y1 + 1))
    regions = c3.multiselect("Sous-régions", sorted(panel.subregion.unique()), default=sorted(panel.subregion.unique()))
    view = cross.period_view(panel, sel_years)
    view = view[view.subregion.isin(regions)]
    if "p_ensemble" in view and view.p_ensemble.isna().all():
        view = view.drop(columns=["p_ensemble", "risk"], errors="ignore")
    nums = [v for v in cross.numeric_vars() if v in view and view[v].notna().sum() >= 4]
    cats = [v for v in cross.categorical_vars() if v in view and view[v].notna().any()]
    st.caption(f"{len(view)} pays · années {sel_years[0]}–{sel_years[-1]}"
               + (" · la probabilité prédite n'existe que pour la dernière année" if "p_ensemble" not in view else ""))

    sub1, sub2, sub3, sub4, sub5 = st.tabs(["Nuage de points X × Y", "Tableau croisé", "Matrice de corrélation",
                                           "Carte bivariée", "Comparaison de groupes"])

    # ------------------------------------------------------------ scatter
    with sub1:
        a, b, c, d_ = st.columns(4)
        x = a.selectbox("Variable X", nums, index=nums.index("ivr_score") if "ivr_score" in nums else 0, format_func=cross.label)
        yv = b.selectbox("Variable Y", nums, index=nums.index("incidence_pm") if "incidence_pm" in nums else 1, format_func=cross.label)
        color = c.selectbox("Couleur", ["subregion"] + [k for k in cats if k != "subregion"], format_func=cross.label)
        logs = d_.multiselect("Échelle log", ["X", "Y"], default=["Y"] if yv in ("incidence_pm", "cases") else [])
        res = cross.correlate(view, x, yv, "X" in logs, "Y" in logs)
        plot = view.dropna(subset=[x, yv]).copy()
        if "X" in logs:
            plot = plot[plot[x] > 0]
        if "Y" in logs:
            plot = plot[plot[yv] > 0]
        fig = px.scatter(plot, x=x, y=yv, color=color, hover_name="country_fr", text="iso3",
                         size="population_total" if "population_total" in plot else None, size_max=38,
                         log_x="X" in logs, log_y="Y" in logs,
                         color_discrete_map=SR_COLORS if color == "subregion" else None,
                         labels={x: cross.label(x), yv: cross.label(yv), color: cross.label(color)})
        fig.update_traces(textposition="top center", textfont_size=9)
        if res.get("n", 0) >= 4:
            xs = np.linspace(plot[x].min(), plot[x].max(), 50)
            xx = np.log10(xs) if res["logx"] else xs
            yy = res["intercept"] + res["slope"] * xx
            fig.add_trace(go.Scatter(x=xs, y=10 ** yy if res["logy"] else yy, mode="lines", name="Régression linéaire",
                                     line=dict(color="#333", dash="dash")))
        fig.update_layout(height=560, legend_title=None)
        st.plotly_chart(fig, width="stretch")
        if res.get("n", 0) >= 4:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Spearman ρ", f"{res['spearman_rho']:.2f}", f"p {_fmt_p(res['spearman_p'])}", delta_color="off")
            m2.metric("Pearson r", f"{res['pearson_r']:.2f}", f"p {_fmt_p(res['pearson_p'])}", delta_color="off")
            m3.metric("R² (régression)", f"{res['r2']:.2f}")
            m4.metric("Pays", res["n"])
            st.caption("Corrélation écologique entre pays : elle n'implique pas de causalité individuelle. "
                       "La régression est calculée sur l'échelle affichée (log10 si cochée).")
        else:
            st.warning("Pas assez de pays avec les deux variables.")

    # ------------------------------------------------------------ crosstab
    with sub2:
        a, b, c, d_ = st.columns(4)
        row = a.selectbox("Lignes", cats, index=cats.index("ivr_class") if "ivr_class" in cats else 0, format_func=cross.label)
        col_opts = [k for k in cats if k != row]
        col = b.selectbox("Colonnes", col_opts, index=col_opts.index("incidence_class") if "incidence_class" in col_opts else 0,
                          format_func=cross.label)
        val = c.selectbox("Contenu des cellules", ["(nombre de pays)"] + nums,
                          format_func=lambda k: k if k.startswith("(") else "Moyenne de : " + cross.label(k))
        norm = d_.radio("Pourcentages", ["Aucun", "Par ligne", "Par colonne"], horizontal=False,
                        disabled=val != "(nombre de pays)")
        t, test = cross.crosstab(view, row, col, None if val.startswith("(") else val,
                                 normalize={"Par ligne": "row", "Par colonne": "col"}.get(norm))
        fig = px.imshow(t, text_auto=".1f" if (norm != "Aucun" or not val.startswith("(")) else True,
                        color_continuous_scale="OrRd", aspect="auto",
                        labels=dict(x=cross.label(col), y=cross.label(row), color=""))
        fig.update_layout(height=420)
        st.plotly_chart(fig, width="stretch")
        if test:
            st.markdown(f"**Test du χ² d'indépendance** : χ² = {test['chi2']:.1f} ; ddl = {test['dof']} ; "
                        f"p = {_fmt_p(test['p'])} ; V de Cramér = {test['cramers_v']:.2f} (n = {test['n']} pays). "
                        "Effectifs faibles : interpréter avec prudence.")
        with st.expander("Liste des pays par cellule"):
            st.dataframe(view.dropna(subset=[row, col])[["country_fr", "subregion", row, col]
                         + ([val] if not val.startswith("(") else [])].sort_values([row, col]), width="stretch")

    # ------------------------------------------------------------ correlation matrix
    with sub3:
        default = [v for v in ["incidence_pm", "outbreak_months", "mcv1", "mcv2", "ivr_score", "under5_mortality",
                               "stunting_pct", "gdp_per_capita_usd", "urban_pct", "children_pct"] if v in nums]
        chosen = st.multiselect("Variables", nums, default=default, format_func=cross.label)
        method = st.radio("Méthode", ["spearman", "pearson"], horizontal=True,
                          format_func=lambda m: "Spearman (rangs)" if m == "spearman" else "Pearson (linéaire)")
        if len(chosen) >= 2:
            cm = cross.corr_matrix(view, chosen, method)
            cm.index = cm.columns = [cross.label(v) for v in chosen]
            fig = px.imshow(cm, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
            fig.update_layout(height=620)
            st.plotly_chart(fig, width="stretch")

    # ------------------------------------------------------------ bivariate map
    with sub4:
        a, b = st.columns(2)
        bx = a.selectbox("Variable 1 (rouge)", nums, index=nums.index("ivr_score") if "ivr_score" in nums else 0,
                         format_func=cross.label, key="bx")
        by = b.selectbox("Variable 2 (bleu)", nums, index=nums.index("incidence_pm") if "incidence_pm" in nums else 1,
                         format_func=cross.label, key="by")
        bv = view.copy()
        bv["classe"] = cross.bivariate_classes(bv, bx, by)
        fig = px.choropleth(bv.dropna(subset=["classe"]), geojson=geo, locations="iso3", featureidkey="properties.iso3",
                            color="classe", color_discrete_map=cross.BIVARIATE_PALETTE, hover_name="country_fr",
                            hover_data={bx: ":.1f", by: ":.1f", "classe": True, "iso3": False},
                            category_orders={"classe": list(cross.BIVARIATE_PALETTE)})
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(height=600, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        c1, c2 = st.columns([4, 1])
        c1.plotly_chart(fig, width="stretch")
        leg = pd.DataFrame([[cross.BIVARIATE_PALETTE[f"{i}-{j}"] for i in (1, 2, 3)] for j in (3, 2, 1)])
        c2.markdown("**Légende** (tertiles)")
        c2.markdown("".join(
            "<div style='display:flex'>" + "".join(
                f"<div style='width:34px;height:34px;background:{leg.iloc[r, k]}'></div>" for k in range(3)) + "</div>"
            for r in range(3)), unsafe_allow_html=True)
        c2.caption(f"→ {cross.label(bx)} croissant\n\n↑ {cross.label(by)} croissant\n\n"
                   "Coin foncé en haut à droite : les deux valeurs sont élevées.")
        hot = bv[bv.classe == "3-3"].country_fr.tolist()
        if hot:
            st.info(f"Pays dans le tertile supérieur pour les deux variables : {', '.join(hot)}")

    # ------------------------------------------------------------ group comparison
    with sub5:
        a, b = st.columns(2)
        var = a.selectbox("Variable", nums, index=nums.index("incidence_pm") if "incidence_pm" in nums else 0,
                          format_func=cross.label, key="gv")
        grp = b.selectbox("Groupes", cats, index=cats.index("mcv1_class") if "mcv1_class" in cats else 0,
                          format_func=cross.label, key="gg")
        gd = view.dropna(subset=[var, grp])
        order = [o for o in cross.CAT_ORDER.get(grp, sorted(gd[grp].unique())) if o in gd[grp].unique()]
        fig = px.box(gd, x=grp, y=var, points="all", hover_name="country_fr", category_orders={grp: order},
                     log_y=var in ("incidence_pm", "cases", "population_total", "gdp_per_capita_usd"),
                     labels={grp: cross.label(grp), var: cross.label(var)}, color_discrete_sequence=["#b3261e"])
        fig.update_layout(height=480)
        st.plotly_chart(fig, width="stretch")
        gt = cross.group_test(gd, var, grp)
        if gt:
            st.markdown(f"**Test de Kruskal-Wallis** : H = {gt['kruskal_h']:.1f} ; p = {_fmt_p(gt['p'])} "
                        f"({gt['k']} groupes, {gt['n']} pays).")
        summ = gd.groupby(grp)[var].agg(["count", "median", "mean", "min", "max"]).reindex(order).round(2)
        st.dataframe(summ, width="stretch")

    st.download_button("Télécharger le tableau croisé des pays (CSV)", view.to_csv(index=False).encode("utf-8"),
                       f"analyse_croisee_{sel_years[0]}_{sel_years[-1]}.csv", "text/csv")
