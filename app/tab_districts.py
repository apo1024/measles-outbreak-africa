"""Onglet « Districts & localités » : cartographie infranationale et surveillance par district."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from measles_predict import cross, subnational as sn

STATUS_COLORS = {"Flambée": "#c0392b", "Vigilance": "#f39c12", "Normal": "#7fbf7b", "Historique insuffisant": "#bdbdbd"}


@st.cache_data(show_spinner="Téléchargement des limites (une seule fois, ensuite en cache)…")
def _units(iso3: str, level: str):
    g = sn.load_boundaries(iso3, level)
    return g.to_json(), g.drop(columns="geometry")


@st.cache_data
def _localities(iso3: str):
    return sn.load_localities(iso3)


def _read_table(f) -> pd.DataFrame:
    name = f.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(f)
    raw = f.read()
    for sep in (",", ";", "\t"):
        try:
            d = pd.read_csv(pd.io.common.BytesIO(raw), sep=sep)
            if d.shape[1] > 1:
                return d
        except Exception:  # noqa: BLE001
            continue
    return pd.read_csv(pd.io.common.BytesIO(raw))


def render_districts(names: dict):
    inv = sn.inventory()
    st.markdown("Limites infranationales recherchées pour les 54 pays : **districts (ADM2)**, "
                "**sous-districts / communes (ADM3)**, **localités administratives (ADM4)** et **villes et villages** "
                "(GeoNames). Source principale : geoBoundaries – limites humanitaires OCHA (COD-AB).")
    with st.expander("Couverture des limites disponibles par pays"):
        cov = inv[inv.level != "LOCALITES"].pivot_table(index="country_fr", columns="level", values="n_units", aggfunc="first")
        loc = inv[inv.level == "LOCALITES"].set_index("country_fr")["n_units"].rename("Localités (points)")
        st.dataframe(cov.join(loc, how="outer").fillna(0).astype(int).astype(str).replace("0", "–"), width="stretch", height=360)

    isos = sorted(names, key=lambda i: names[i])
    c1, c2, c3 = st.columns([2, 2, 2])
    iso = c1.selectbox("Pays", isos, index=isos.index("NER") if "NER" in isos else 0,
                       format_func=lambda i: f"{names[i]} ({i})", key="dist_iso")
    source = c3.radio("Limites", ["geoBoundaries (téléchargées)", "Mon fichier (KML, GeoJSON, ZIP shapefile)"], key="dist_src")
    levels = sn.available_levels(iso)
    level = c2.selectbox("Niveau", levels, index=levels.index("ADM2") if "ADM2" in levels else 0,
                         format_func=lambda l: sn.LEVEL_LABELS[l], key="dist_lvl")

    if source.startswith("Mon fichier"):
        up = st.file_uploader("Fichier de limites (zones de santé, aires de santé…)", type=["kml", "geojson", "json", "zip", "gpkg"])
        if not up:
            st.info("Importez un fichier de limites, ou choisissez geoBoundaries.")
            return
        g = sn.load_user_boundaries(up)
        gj, units = g.to_json(), g.drop(columns="geometry")
    else:
        try:
            gj, units = _units(iso, level)
        except Exception as e:  # noqa: BLE001
            st.error(f"Limites indisponibles : {e}")
            return
    geo = json.loads(gj)
    for i, f in enumerate(geo["features"]):
        f["id"] = str(units.unit_id.iloc[i])
    st.caption(f"{len(units)} unités · {sn.LEVEL_LABELS.get(level, 'fichier importé')}")

    # ------------------------------------------------ données de surveillance
    st.subheader("1. Vos données par district (facultatif)")
    st.markdown("Importez un fichier CSV/Excel avec une colonne de **nom de district**, une colonne **cas** et, "
                "facultativement, **période** (AAAA-MM ou date), **population** et toute autre variable "
                "(couverture vaccinale, accès aux soins…). Les noms sont rapprochés automatiquement des limites.")
    t1, t2 = st.columns(2)
    t1.download_button("Modèle de fichier à remplir (CSV)", sn.template_csv(units), f"modele_{iso}_{level}.csv", "text/csv")
    data = st.file_uploader("Fichier de surveillance", type=["csv", "xlsx", "xls"], key="dist_data")
    merged, surv_last, surv_all, user = None, None, None, None
    if data:
        user = _read_table(data)
        cols = list(user.columns)
        guess = lambda keys: next((c for c in cols if any(k in c.lower() for k in keys)), None)  # noqa: E731
        a, b, c, d_ = st.columns(4)
        name_col = a.selectbox("Colonne district", cols, index=cols.index(guess(["district", "zone", "nom", "name", "unit"]) or cols[0]))
        cases_col = b.selectbox("Colonne cas", cols, index=cols.index(guess(["cas", "case"]) or cols[min(1, len(cols) - 1)]))
        per_col = c.selectbox("Colonne période", ["(aucune)"] + cols,
                              index=(["(aucune)"] + cols).index(guess(["period", "date", "mois", "month", "semaine", "week"]) or "(aucune)"))
        pop_col = d_.selectbox("Colonne population", ["(aucune)"] + cols,
                               index=(["(aucune)"] + cols).index(guess(["pop"]) or "(aucune)"))
        cutoff = st.slider("Tolérance du rapprochement des noms (1 = identique)", 0.6, 1.0, 0.82, 0.02)
        joined, matches = sn.join_data(pd.DataFrame({"unit_name": units.unit_name}), user, name_col, cutoff)
        ok = matches.match.notna().sum()
        st.write(f"Rapprochement : **{ok}/{len(matches)}** noms trouvés "
                 f"({(matches.method == 'approché').sum()} par similarité).")
        with st.expander("Voir / vérifier le rapprochement des noms"):
            st.dataframe(matches.sort_values("score"), width="stretch")
        joined = joined[joined.unit_name_matched.notna()]
        if per_col != "(aucune)":
            k1, k2, k3 = st.columns(3)
            window = k1.number_input("Fenêtre de référence (périodes)", 6, 104, 24)
            ksd = k2.number_input("Nombre d'écarts-types", 1.0, 3.0, 2.0, 0.5)
            mincase = k3.number_input("Seuil minimal (cas)", 1, 50, 5)
            surv_all, surv_last = sn.district_surveillance(joined, "unit_name_matched", per_col, cases_col,
                                                           None if pop_col == "(aucune)" else pop_col,
                                                           window=int(window), k_sd=ksd, min_cases=int(mincase))
            merged = units.merge(surv_last, left_on="unit_name", right_on="unit", how="left")
        else:
            agg = joined.groupby("unit_name_matched").agg(
                {cases_col: "sum", **({pop_col: "max"} if pop_col != "(aucune)" else {})}).reset_index()
            agg = agg.rename(columns={cases_col: "cases", pop_col: "population"} if pop_col != "(aucune)" else {cases_col: "cases"})
            if "population" in agg:
                agg["incidence_100k"] = 1e5 * agg.cases / agg.population
            extra = [x for x in user.select_dtypes("number").columns if x not in (cases_col, pop_col)]
            if extra:
                agg = agg.merge(joined.groupby("unit_name_matched")[extra].mean().reset_index(), on="unit_name_matched")
            merged = units.merge(agg, left_on="unit_name", right_on="unit_name_matched", how="left")

    # ------------------------------------------------ carte
    st.subheader("2. Carte")
    loc = _localities(iso)
    m1, m2, m3 = st.columns(3)
    show_loc = m1.checkbox("Afficher les localités (villes et villages)", value=True)
    minpop = m2.select_slider("Population minimale des localités", [1000, 5000, 10000, 50000, 100000], value=5000)
    indicator = None
    if merged is not None:
        opts = [c for c in ("status", "ratio_to_threshold", "cases", "incidence_100k") if c in merged and merged[c].notna().any()]
        opts += [c for c in merged.select_dtypes("number").columns if c not in opts and c not in ("baseline_mean", "baseline_sd", "outbreak", "threshold")]
        indicator = m3.selectbox("Indicateur cartographié", opts)
    fig = go.Figure()
    if indicator is None:
        fig.add_trace(go.Choroplethmap(geojson=geo, locations=units.unit_id.astype(str), z=np.zeros(len(units)),
                                       colorscale=[[0, "#f3e1dc"], [1, "#f3e1dc"]], showscale=False,
                                       marker_line_color="#7a2e1f", marker_line_width=0.8, marker_opacity=0.45,
                                       text=units.unit_name, hovertemplate="%{text}<extra></extra>"))
    elif indicator == "status":
        for stt, col in STATUS_COLORS.items():
            sub = merged[merged.status == stt]
            if len(sub):
                fig.add_trace(go.Choroplethmap(geojson=geo, locations=sub.unit_id.astype(str), z=np.ones(len(sub)),
                                               colorscale=[[0, col], [1, col]], showscale=False, name=stt, showlegend=True,
                                               marker_opacity=0.75, marker_line_width=0.6, marker_line_color="white",
                                               text=sub.unit_name + " – " + stt + " (cas " + sub.cases.fillna(0).astype(int).astype(str) + ")",
                                               hovertemplate="%{text}<extra></extra>"))
    else:
        fig.add_trace(go.Choroplethmap(geojson=geo, locations=merged.unit_id.astype(str), z=merged[indicator],
                                       colorscale="OrRd", marker_opacity=0.75, marker_line_width=0.5,
                                       marker_line_color="white", colorbar_title=indicator,
                                       text=merged.unit_name, hovertemplate="%{text}: %{z:.1f}<extra></extra>"))
    if show_loc and len(loc):
        lp = loc[loc.population >= minpop]
        fig.add_trace(go.Scattermap(lat=lp.lat, lon=lp.lon, mode="markers", name="Localités",
                                    marker=dict(size=np.clip(np.log10(lp.population.clip(lower=1000)) * 3 - 5, 4, 16),
                                                color="#1d2433", opacity=0.7),
                                    text=lp.name + " (" + lp.population.map("{:,}".format).str.replace(",", " ") + " hab.)",
                                    hovertemplate="%{text}<extra></extra>"))
    b = json.loads(json.dumps(geo))  # bounds
    xs, ys = [], []
    for f in b["features"]:
        def walk(c):
            if isinstance(c[0], (int, float)):
                xs.append(c[0]); ys.append(c[1])
            else:
                for cc in c:
                    walk(cc)
        walk(f["geometry"]["coordinates"])
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    zoom = float(np.clip(8.5 - np.log2(span + 1e-6) * 1.0, 2, 11))
    fig.update_layout(map=dict(style="open-street-map", center=dict(lon=(min(xs) + max(xs)) / 2, lat=(min(ys) + max(ys)) / 2), zoom=zoom),
                      height=650, margin=dict(l=0, r=0, t=0, b=0), legend=dict(bgcolor="rgba(255,255,255,0.8)"))
    st.plotly_chart(fig, width="stretch")

    # ------------------------------------------------ téléchargements
    d1, d2, d3, d4 = st.columns(4)
    import geopandas as gpd
    gdf = gpd.GeoDataFrame.from_features(geo["features"], crs=4326)
    if merged is not None:
        gdf = gdf.merge(merged.drop(columns=[c for c in merged.columns if c in gdf.columns and c != "unit_id"]),
                        on="unit_id", how="left") if "unit_id" in gdf else gdf
    for c in gdf.columns:
        if pd.api.types.is_datetime64_any_dtype(gdf[c]):
            gdf[c] = gdf[c].dt.strftime("%Y-%m-%d")
    d1.download_button("Limites (KML)", sn.to_kml_bytes(gdf), f"{iso}_{level}.kml", "application/vnd.google-earth.kml+xml")
    d2.download_button("Limites (GeoJSON)", gdf.to_json().encode("utf-8"), f"{iso}_{level}.geojson", "application/geo+json")
    if len(loc):
        lk = loc.rename(columns={"name": "nom"})
        d3.download_button("Localités (KML)", sn.to_kml_bytes(lk, "nom"), f"{iso}_localites.kml", "application/vnd.google-earth.kml+xml")
        units_g = gdf[["unit_id", "unit_name", "geometry"]] if "unit_name" in gdf else None
        if units_g is not None:
            lj = sn.localities_in_units(loc, units_g)
            d4.download_button("Localités par district (CSV)", lj.to_csv(index=False).encode("utf-8"),
                               f"{iso}_{level}_localites_par_district.csv", "text/csv")

    # ------------------------------------------------ résultats surveillance
    if surv_last is not None:
        st.subheader("3. Surveillance par district")
        counts = surv_last.status.value_counts()
        k = st.columns(4)
        for i, s in enumerate(["Flambée", "Vigilance", "Normal", "Historique insuffisant"]):
            k[i].metric(s, int(counts.get(s, 0)))
        show = surv_last[["unit", "period", "cases", "threshold", "ratio_to_threshold", "status"]
                         + (["incidence_100k"] if "incidence_100k" in surv_last else [])].rename(columns={
            "unit": "District", "period": "Dernière période", "cases": "Cas", "threshold": "Seuil",
            "ratio_to_threshold": "Cas / seuil", "status": "Statut"})
        st.dataframe(show.style.format({"Seuil": "{:.0f}", "Cas / seuil": "{:.2f}"}), width="stretch", height=360)
        sel = st.selectbox("Courbe épidémique du district", surv_last.unit.tolist())
        s = surv_all[surv_all.unit == sel]
        f2 = go.Figure()
        f2.add_bar(x=s.period, y=s.cases, name="Cas", marker_color="#2a78b5")
        f2.add_scatter(x=s.period, y=s.threshold, name="Seuil épidémique", line=dict(color="#d9534f", dash="dash"))
        ob = s[s.outbreak == 1]
        f2.add_scatter(x=ob.period, y=ob.cases, mode="markers", name="Flambée", marker=dict(color="#d9534f", size=9, symbol="x"))
        f2.update_layout(height=360, legend=dict(orientation="h"))
        st.plotly_chart(f2, width="stretch")
        st.download_button("Télécharger les alertes par district (CSV)", surv_last.to_csv(index=False).encode("utf-8"),
                           f"alertes_districts_{iso}.csv", "text/csv")

    # ------------------------------------------------ analyses croisées district
    if merged is not None:
        numc = [c for c in merged.select_dtypes("number").columns
                if merged[c].notna().sum() >= 4 and c not in ("baseline_sd", "outbreak")]
        if len(numc) >= 2:
            st.subheader("4. Analyse croisée entre districts")
            a, b = st.columns(2)
            x = a.selectbox("X", numc, key="dx")
            y = b.selectbox("Y", [c for c in numc if c != x], key="dy")
            res = cross.correlate(merged, x, y)
            fx = px.scatter(merged.dropna(subset=[x, y]), x=x, y=y, hover_name="unit_name", trendline=None)
            st.plotly_chart(fx, width="stretch")
            if res.get("n", 0) >= 4:
                st.markdown(f"Spearman ρ = **{res['spearman_rho']:.2f}** (p = {res['spearman_p']:.3g}) · "
                            f"Pearson r = {res['pearson_r']:.2f} · n = {res['n']} districts")

    st.caption("Limites : geoBoundaries (gbHumanitarian / OCHA COD-AB, gbOpen ; licences ODbL / CC BY / CC BY-IGO). "
               "Localités : GeoNames (CC BY 4.0), villes et villages de plus de 1 000 habitants. "
               "Les frontières affichées n'impliquent aucune prise de position officielle.")
