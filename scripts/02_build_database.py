"""
02_construction_base.py
Nettoie les données brutes, calcule l'Indice de Vulnérabilité Rougeole (IVR),
construit la table spatio-temporelle pays x mois et charge le tout dans SQLite.

Sorties :
  02_Base_de_donnees/rougeole_afrique.sqlite
  02_Base_de_donnees/processed/*.csv  (une copie CSV de chaque table)
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]   # repository root
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data"
GEO = ROOT / "data"
DB = ROOT / "data" / "rougeole_afrique.sqlite"
PROC.mkdir(exist_ok=True)

START_YEAR = 2012

# ---------------------------------------------------------------- IVR
# Domaines, poids et variables (signe +1 = plus la valeur est élevée, plus la
# vulnérabilité est forte ; -1 = l'inverse).
IVR_DOMAINS = {
    "d_immunite":    (0.35, [("mcv1", -1), ("mcv2", -1)]),
    "d_demographie": (0.15, [("children_pct", +1), ("birth_rate", +1), ("log_density", +1)]),
    "d_socioeco":    (0.15, [("log_gdp", -1), ("poverty_215_pct", +1)]),
    "d_nutrition":   (0.15, [("stunting_pct", +1)]),
    "d_acces_soins": (0.20, [("under5_mortality", +1), ("skilled_birth_attendance", -1),
                             ("log_health_exp", -1), ("physicians_per_1000", -1)]),
}
IVR_CLASSES = ["Faible", "Modérée", "Élevée", "Très élevée"]


def load_countries():
    c = pd.read_csv(RAW / "countries.csv")
    m = pd.read_csv(RAW / "who_measles_monthly_africa.csv")
    names = m.drop_duplicates("iso3").set_index("iso3")
    c["country_en"] = c["iso3"].map(names["country"])
    c["who_region"] = c["iso3"].map(names["region"])
    c.loc[c.iso3 == "MUS", ["country_en", "who_region"]] = ["Mauritius", "AFR"]
    lon, lat, area = [], [], []
    africa = gpd.read_file(GEO / "africa_adm0_simplified.geojson")
    for iso in c.iso3:
        g = africa[africa.iso3 == iso]
        ga = g.to_crs("ESRI:102022")  # Africa Albers Equal Area
        cen = ga.geometry.union_all().centroid
        cen_ll = gpd.GeoSeries([cen], crs="ESRI:102022").to_crs(4326).iloc[0]
        lon.append(round(cen_ll.x, 4)); lat.append(round(cen_ll.y, 4))
        area.append(round(ga.area.sum() / 1e6, 0))
    c["centroid_lon"], c["centroid_lat"], c["area_km2"] = lon, lat, area
    return c


def load_boundaries(countries):
    meta = pd.read_csv(GEO / "boundaries_metadata.csv") if (GEO / "boundaries_metadata.csv").exists() else None
    rows = []
    for iso in countries.iso3:
        for adm in ("ADM0", "ADM1"):
            g = gpd.read_file(GEO / f"{iso}_{adm}.geojson")
            src = lic = None
            if meta is not None:
                r = meta[(meta.iso3 == iso) & (meta.level == adm)]
                if len(r):
                    src, lic = r.iloc[0]["source"], r.iloc[0]["license"]
            rows.append({"iso3": iso, "adm_level": adm, "n_units": len(g),
                         "kml_path": f"03_SIG_Cartographie/kml/{adm}/{iso}_{adm}.kml",
                         "geojson_path": f"03_SIG_Cartographie/geojson/{iso}_{adm}.geojson",
                         "source": src, "license": lic})
    return pd.DataFrame(rows)


def load_monthly():
    m = pd.read_csv(RAW / "who_measles_monthly_africa.csv")
    m = m.rename(columns={"measles_suspect": "suspect", "measles_clinical": "clinical",
                          "measles_epi-linked": "epi_linked", "measles_lab-confirmed": "lab_confirmed",
                          "measles_total": "total_cases", "rubellatotal": "rubella_total"})
    cols = ["iso3", "year", "month", "suspect", "clinical", "epi_linked", "lab_confirmed",
            "total_cases", "discarded", "rubella_total"]
    m = m[cols].groupby(["iso3", "year", "month"], as_index=False).sum(min_count=1)
    return m


def load_annual():
    g = pd.read_csv(RAW / "who_gho_annual.csv")
    annual = g[["iso3", "year", "measles_cases_reported"]].dropna().rename(
        columns={"measles_cases_reported": "reported_cases"})
    vacc = g[["iso3", "year", "mcv1", "mcv2"]].dropna(how="all", subset=["mcv1", "mcv2"])
    return annual, vacc


def build_socioeconomic(countries, vacc):
    w = pd.read_csv(RAW / "worldbank_annual.csv")
    years = range(2005, 2027)
    grid = pd.MultiIndex.from_product([countries.iso3, years], names=["iso3", "year"]).to_frame(index=False)
    s = grid.merge(w, on=["iso3", "year"], how="left")
    s = s.merge(countries[["iso3", "subregion"]], on="iso3")
    s = s.drop(columns=[c for c in ["refugee_population"] if c in s])
    vars_ = ["population_total", "population_0_14", "urban_pct", "pop_density", "birth_rate",
             "gdp_per_capita_usd", "poverty_215_pct", "under5_mortality", "stunting_pct",
             "health_exp_per_capita", "physicians_per_1000", "skilled_birth_attendance"]
    flags = pd.DataFrame(index=s.index, columns=vars_, data=False)
    s = s.sort_values(["iso3", "year"]).reset_index(drop=True)
    for v in vars_:
        miss = s[v].isna()
        # 1) interpolation linéaire intra-pays + prolongation (ffill / bfill)
        s[v] = s.groupby("iso3")[v].transform(lambda x: x.interpolate(limit_direction="both"))
        # 2) population : extrapolation par le taux de croissance récent
        # 3) reste : médiane sous-région x année, puis médiane globale
        s[v] = s[v].fillna(s.groupby(["subregion", "year"])[v].transform("median"))
        s[v] = s[v].fillna(s[v].median())
        flags[v] = miss
    # Population 2025-2026 : croissance observée 2019-2024
    for v in ["population_total", "population_0_14"]:
        for iso, d in s.groupby("iso3"):
            obs = w[(w.iso3 == iso) & w[v].notna()].sort_values("year")
            if obs.empty:
                continue
            last_y = int(obs.year.max())
            last = obs[v].iloc[-1]
            prev = obs[obs.year == last_y - 5][v]
            g = (last / prev.iloc[0]) ** (1 / 5) if len(prev) else 1.02
            for y in range(last_y + 1, 2027):
                s.loc[(s.iso3 == iso) & (s.year == y), v] = last * g ** (y - last_y)
    s["children_pct"] = 100 * s["population_0_14"] / s["population_total"]
    s["imputed_flag"] = flags.apply(lambda r: ";".join([v for v in vars_ if r[v]]), axis=1)
    return s.drop(columns=["subregion"])


def minmax(x: pd.Series) -> pd.Series:
    lo, hi = x.quantile(0.025), x.quantile(0.975)
    return ((x.clip(lo, hi) - lo) / (hi - lo)).fillna(0.5)


def build_ivr(socio, vacc, countries):
    v = vacc.copy()
    # MCV2 manquant = vaccin non introduit -> 0 %
    v["mcv2"] = v["mcv2"].fillna(0)
    d = socio.merge(v, on=["iso3", "year"], how="left").sort_values(["iso3", "year"])
    for c in ["mcv1", "mcv2"]:
        d[c] = d.groupby("iso3")[c].transform(lambda x: x.ffill().bfill())
    d["log_density"] = np.log1p(d["pop_density"])
    d["log_gdp"] = np.log(d["gdp_per_capita_usd"])
    d["log_health_exp"] = np.log1p(d["health_exp_per_capita"])
    d = d[d.year >= 2010].copy()
    for dom, (_, items) in IVR_DOMAINS.items():
        parts = [minmax(d[var]) if sign > 0 else 1 - minmax(d[var]) for var, sign in items]
        d[dom] = np.mean(parts, axis=0)
    d["ivr_score"] = 100 * sum(w * d[dom] for dom, (w, _) in IVR_DOMAINS.items())
    cuts = d["ivr_score"].quantile([0.25, 0.5, 0.75]).values
    d["ivr_class"] = pd.cut(d["ivr_score"], [-np.inf, *cuts, np.inf], labels=IVR_CLASSES).astype(str)
    d["ivr_rank"] = d.groupby("year")["ivr_score"].rank(ascending=False, method="min").astype(int)
    cols = ["iso3", "year", *IVR_DOMAINS, "ivr_score", "ivr_class", "ivr_rank"]
    out = d[cols].round(4)
    pd.Series(cuts, index=["q25", "q50", "q75"]).to_csv(PROC / "ivr_class_cutpoints.csv")
    return out, d[["iso3", "year", "mcv1", "mcv2"]]


def build_model_inputs(monthly, socio, ivr, vacc_filled, countries):
    last = monthly.assign(p=monthly.year * 100 + monthly.month).p.max()
    periods = pd.period_range(f"{START_YEAR}-01", f"{last // 100}-{last % 100:02d}", freq="M")
    iso_with_data = sorted(monthly.iso3.unique())
    grid = pd.MultiIndex.from_product([iso_with_data, periods], names=["iso3", "per"]).to_frame(index=False)
    grid["year"], grid["month"] = grid.per.dt.year, grid.per.dt.month
    d = grid.merge(monthly[["iso3", "year", "month", "total_cases"]], on=["iso3", "year", "month"], how="left")
    d = d.rename(columns={"total_cases": "cases"}).sort_values(["iso3", "per"]).reset_index(drop=True)
    d["reported"] = d["cases"].notna().astype(int)
    g = d.groupby("iso3")["cases"]
    for k in (1, 2, 3):
        d[f"cases_lag{k}"] = g.shift(k)
    d["cases_sum_3m"] = g.transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())
    d["cases_sum_12m"] = g.transform(lambda x: x.shift(1).rolling(12, min_periods=6).sum())
    d["baseline_mean_24m"] = g.transform(lambda x: x.shift(1).rolling(24, min_periods=12).mean())
    d["baseline_sd_24m"] = g.transform(lambda x: x.shift(1).rolling(24, min_periods=12).std())
    d["epidemic_threshold"] = np.maximum(10, d.baseline_mean_24m + 2 * d.baseline_sd_24m)
    ok = d.cases.notna() & d.baseline_mean_24m.notna()
    d["outbreak"] = np.where(ok, (d.cases > d.epidemic_threshold).astype(float), np.nan)
    go = d.groupby("iso3")["outbreak"]
    d["outbreak_next1"] = go.shift(-1)
    nxt = pd.concat([go.shift(-k) for k in (1, 2, 3)], axis=1)
    d["outbreak_next3"] = np.where(nxt.notna().any(axis=1), nxt.max(axis=1), np.nan)
    d["cases_next1"] = g.shift(-1)

    def since(x):
        out, c = [], np.nan
        for v in x:
            out.append(c)
            if v == 1:
                c = 0
            elif not np.isnan(c):
                c += 1
        return pd.Series(out, index=x.index)
    d["months_since_outbreak"] = go.transform(since).fillna(60).clip(upper=60)
    d["month_sin"] = np.sin(2 * np.pi * d.month / 12).round(5)
    d["month_cos"] = np.cos(2 * np.pi * d.month / 12).round(5)

    # Covariables annuelles de l'année précédente (pas de fuite d'information :
    # la couverture de l'année N n'est publiée qu'en juillet N+1)
    cov = (socio[["iso3", "year", "population_total", "children_pct", "pop_density", "urban_pct"]]
           .merge(ivr[["iso3", "year", "ivr_score"]], on=["iso3", "year"], how="left")
           .merge(vacc_filled, on=["iso3", "year"], how="left"))
    cov_lag = cov.copy()
    cov_lag["year"] = cov_lag["year"] + 1
    d = d.merge(cov_lag.drop(columns=["population_total"]), on=["iso3", "year"], how="left")
    d = d.merge(socio[["iso3", "year", "population_total"]], on=["iso3", "year"], how="left")
    d["incidence_pm"] = (1e6 * d.cases / d.population_total).round(4)
    d["log_population"] = np.log(d.population_total).round(5)
    d = d.merge(countries[["iso3", "subregion"]], on="iso3")
    d["period"] = d.per.astype(str)
    cols = ["iso3", "year", "month", "period", "reported", "cases", "incidence_pm",
            "cases_lag1", "cases_lag2", "cases_lag3", "cases_sum_3m", "cases_sum_12m",
            "baseline_mean_24m", "baseline_sd_24m", "epidemic_threshold", "outbreak",
            "outbreak_next1", "outbreak_next3", "cases_next1", "months_since_outbreak",
            "month_sin", "month_cos", "mcv1", "mcv2", "ivr_score", "children_pct",
            "pop_density", "urban_pct", "log_population", "subregion"]
    return d[cols]


SOURCES = [
    ("WHO_MONTHLY", "OMS – données provisoires mensuelles rougeole/rubéole par pays",
     "https://immunizationdata.who.int/docs/librariesprovider21/measles-and-rubella/404-table-web-epi-curve-data.xlsx",
     "OMS – usage avec citation"),
    ("WHO_GHO", "OMS Global Health Observatory : WHS3_62, WHS8_110 (MCV1), MCV2 (WUENIC)",
     "https://ghoapi.azureedge.net/api/", "CC BY-NC-SA 3.0 IGO"),
    ("WORLDBANK", "Banque mondiale – World Development Indicators", "https://api.worldbank.org/v2/", "CC BY 4.0"),
    ("GEOBOUNDARIES", "geoBoundaries gbOpen – limites ADM0/ADM1", "https://www.geoboundaries.org/", "ODbL / CC BY"),
]


def main():
    countries = load_countries()
    boundaries = pd.DataFrame(columns=["iso3", "adm_level", "n_units", "kml_path", "geojson_path", "source", "license"])
    monthly = load_monthly()
    annual, vacc = load_annual()
    socio = build_socioeconomic(countries, vacc)
    ivr, vacc_filled = build_ivr(socio, vacc, countries)
    mi = build_model_inputs(monthly, socio, ivr, vacc_filled, countries)
    accessed = pd.Timestamp.now().strftime("%Y-%m-%d")
    sources = pd.DataFrame(SOURCES, columns=["source_id", "description", "url", "licence"]).assign(accessed=accessed)

    socio_cols = ["iso3", "year", "population_total", "population_0_14", "children_pct", "urban_pct",
                  "pop_density", "birth_rate", "gdp_per_capita_usd", "poverty_215_pct", "under5_mortality",
                  "stunting_pct", "health_exp_per_capita", "physicians_per_1000",
                  "skilled_birth_attendance", "imputed_flag"]
    tables = {
        "countries": countries, "boundaries": boundaries, "measles_monthly": monthly,
        "measles_annual": annual, "vaccination": vacc, "socioeconomic": socio[socio_cols],
        "vulnerability_index": ivr, "model_inputs": mi, "data_sources": sources,
    }
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    con.executescript((ROOT / "data" / "schema.sql").read_text(encoding="utf-8"))
    for name, df in tables.items():
        df.to_sql(name, con, if_exists="append", index=False)
        if name in ("model_inputs", "countries", "vulnerability_index", "measles_monthly", "vaccination", "socioeconomic"):
            df.to_csv(PROC / f"{name}.csv", index=False)
        print(f"{name:22s} {len(df):>7,} lignes")
    con.commit()
    con.close()
    print(f"Base créée : {DB}")


if __name__ == "__main__":
    main()
