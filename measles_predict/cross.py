"""Cross-analyses (analyses croisées) between epidemiological, immunisation,
vulnerability, socio-economic and prediction variables.

Main entry points
-----------------
build_panel()            country x year table with every variable
period_view()            one row per country for a year or a multi-year average
correlate()              Pearson / Spearman + OLS slope between two variables
crosstab()               contingency table (counts or mean of a 3rd variable) + chi² / Cramér's V
corr_matrix()            correlation matrix
group_test()             Kruskal-Wallis comparison of a variable across groups
bivariate_classes()      3 x 3 classes for a bivariate map
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

# code -> (label FR, kind)   kind: num | cat
VARIABLES = {
    "cases": ("Cas notifiés", "num"),
    "incidence_pm": ("Incidence (cas / million)", "num"),
    "outbreak_months": ("Mois de flambée", "num"),
    "reported_months": ("Mois notifiés", "num"),
    "mcv1": ("Couverture MCV1 (%)", "num"),
    "mcv2": ("Couverture MCV2 (%)", "num"),
    "ivr_score": ("Indice de vulnérabilité (IVR)", "num"),
    "d_immunite": ("IVR – déficit vaccinal (0-1)", "num"),
    "d_acces_soins": ("IVR – accès aux soins (0-1)", "num"),
    "d_demographie": ("IVR – démographie (0-1)", "num"),
    "d_socioeco": ("IVR – pauvreté (0-1)", "num"),
    "d_nutrition": ("IVR – malnutrition (0-1)", "num"),
    "population_total": ("Population totale", "num"),
    "children_pct": ("Population 0-14 ans (%)", "num"),
    "urban_pct": ("Population urbaine (%)", "num"),
    "pop_density": ("Densité (hab./km²)", "num"),
    "birth_rate": ("Natalité (‰)", "num"),
    "gdp_per_capita_usd": ("PIB par habitant (USD)", "num"),
    "poverty_215_pct": ("Pauvreté à 2,15 $/jour (%)", "num"),
    "under5_mortality": ("Mortalité < 5 ans (‰)", "num"),
    "stunting_pct": ("Retard de croissance (%)", "num"),
    "health_exp_per_capita": ("Dépenses de santé / hab. (USD)", "num"),
    "physicians_per_1000": ("Médecins / 1 000 hab.", "num"),
    "skilled_birth_attendance": ("Accouchements assistés (%)", "num"),
    "p_ensemble": ("Probabilité de flambée prédite (3 mois)", "num"),
    "subregion": ("Sous-région", "cat"),
    "ivr_class": ("Classe de vulnérabilité", "cat"),
    "risk": ("Niveau de risque prédit", "cat"),
    "mcv1_class": ("Classe de couverture MCV1", "cat"),
    "incidence_class": ("Classe d'incidence", "cat"),
    "outbreak_any": ("Au moins une flambée", "cat"),
}
CAT_ORDER = {
    "ivr_class": ["Faible", "Modérée", "Élevée", "Très élevée"],
    "risk": ["Faible", "Modéré", "Élevé", "Très élevé"],
    "mcv1_class": ["< 60 %", "60-79 %", "80-89 %", "≥ 90 %"],
    "incidence_class": ["< 5", "5-20", "20-50", "50-100", "≥ 100"],
    "outbreak_any": ["Non", "Oui"],
}


def label(code: str) -> str:
    return VARIABLES.get(code, (code, ""))[0]


def numeric_vars() -> list[str]:
    return [k for k, (_, t) in VARIABLES.items() if t == "num"]


def categorical_vars() -> list[str]:
    return [k for k, (_, t) in VARIABLES.items() if t == "cat"]


def build_panel(data_dir: Path = DATA, forecast: pd.DataFrame | None = None) -> pd.DataFrame:
    """Country x year panel (2012 onwards) with all variables."""
    c = pd.read_csv(data_dir / "countries.csv")
    m = pd.read_csv(data_dir / "measles_monthly.csv")
    mi = pd.read_csv(data_dir / "model_inputs.csv")
    v = pd.read_csv(data_dir / "vaccination.csv")
    s = pd.read_csv(data_dir / "socioeconomic.csv")
    ivr = pd.read_csv(data_dir / "vulnerability_index.csv")
    ann = m.groupby(["iso3", "year"]).agg(cases=("total_cases", "sum"), reported_months=("month", "nunique")).reset_index()
    ob = mi.groupby(["iso3", "year"])["outbreak"].sum(min_count=1).rename("outbreak_months").reset_index()
    years = range(2012, int(m.year.max()) + 1)
    p = pd.MultiIndex.from_product([c.iso3, years], names=["iso3", "year"]).to_frame(index=False)
    p = (p.merge(c[["iso3", "country_fr", "subregion"]], on="iso3")
          .merge(ann, on=["iso3", "year"], how="left").merge(ob, on=["iso3", "year"], how="left")
          .merge(v, on=["iso3", "year"], how="left")
          .merge(s.drop(columns=["imputed_flag"], errors="ignore"), on=["iso3", "year"], how="left")
          .merge(ivr.drop(columns=["ivr_rank"], errors="ignore"), on=["iso3", "year"], how="left"))
    p["incidence_pm"] = 1e6 * p.cases / p.population_total
    if forecast is not None:
        last = int(p.year.max())
        f = forecast[["iso3", "p_ensemble", "risk"]].assign(year=last)
        p = p.merge(f, on=["iso3", "year"], how="left")
    return add_classes(p)


def add_classes(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["mcv1_class"] = pd.cut(d.mcv1, [-1, 60, 80, 90, 101], right=False, labels=CAT_ORDER["mcv1_class"]).astype(str)
    d["incidence_class"] = pd.cut(d.incidence_pm, [-1, 5, 20, 50, 100, np.inf], right=False,
                                  labels=CAT_ORDER["incidence_class"]).astype(str)
    if "outbreak_months" in d:
        d["outbreak_any"] = d.outbreak_months.map(lambda v: np.nan if pd.isna(v) else ("Oui" if v > 0 else "Non"))
    for col in categorical_vars():
        if col in d:
            d.loc[d[col].isin(["nan", "None"]), col] = np.nan
    return d


def period_view(panel: pd.DataFrame, years) -> pd.DataFrame:
    """One row per country: values of a single year, or mean over a range of years
    (sums for counts are averaged per year; categorical = most frequent)."""
    years = [years] if isinstance(years, int) else list(years)
    sub = panel[panel.year.isin(years)]
    nums = [c for c in numeric_vars() if c in sub]
    agg = sub.groupby(["iso3", "country_fr", "subregion"])[nums].mean().reset_index()
    cats = [c for c in ("ivr_class", "risk") if c in sub]
    for c in cats:
        mode = sub.groupby("iso3")[c].agg(lambda x: x.dropna().mode().iloc[0] if x.notna().any() else np.nan)
        agg[c] = agg.iso3.map(mode)
    return add_classes(agg)


def correlate(d: pd.DataFrame, x: str, y: str, logx=False, logy=False) -> dict:
    sub = d[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if logx:
        sub = sub[sub[x] > 0]
    if logy:
        sub = sub[sub[y] > 0]
    n = len(sub)
    if n < 4:
        return {"n": n}
    xx = np.log10(sub[x]) if logx else sub[x]
    yy = np.log10(sub[y]) if logy else sub[y]
    pr, pp = stats.pearsonr(xx, yy)
    sr, sp = stats.spearmanr(sub[x], sub[y])
    slope, intercept, r, p, se = stats.linregress(xx, yy)
    return {"n": n, "pearson_r": pr, "pearson_p": pp, "spearman_rho": sr, "spearman_p": sp,
            "slope": slope, "intercept": intercept, "r2": r ** 2, "slope_se": se, "logx": logx, "logy": logy}


def crosstab(d: pd.DataFrame, row: str, col: str, value: str | None = None, agg: str = "mean",
             normalize: str | None = None) -> tuple[pd.DataFrame, dict]:
    sub = d.dropna(subset=[row, col])
    if value:
        t = sub.pivot_table(index=row, columns=col, values=value, aggfunc=agg)
    else:
        t = pd.crosstab(sub[row], sub[col])
    t = t.reindex(index=[r for r in CAT_ORDER.get(row, sorted(t.index)) if r in t.index],
                  columns=[c for c in CAT_ORDER.get(col, sorted(t.columns)) if c in t.columns])
    test = {}
    counts = pd.crosstab(sub[row], sub[col])
    if counts.shape[0] > 1 and counts.shape[1] > 1:
        chi2, p, dof, _ = stats.chi2_contingency(counts)
        n = counts.values.sum()
        test = {"chi2": chi2, "p": p, "dof": dof, "n": int(n),
                "cramers_v": float(np.sqrt(chi2 / (n * (min(counts.shape) - 1))))}
    if not value and normalize == "row":
        t = 100 * t.div(t.sum(axis=1), axis=0)
    elif not value and normalize == "col":
        t = 100 * t.div(t.sum(axis=0), axis=1)
    return t, test


def corr_matrix(d: pd.DataFrame, cols: list[str], method: str = "spearman") -> pd.DataFrame:
    return d[cols].apply(pd.to_numeric, errors="coerce").corr(method=method)


def group_test(d: pd.DataFrame, var: str, by: str) -> dict:
    groups = [g[var].dropna().values for _, g in d.groupby(by) if g[var].notna().sum() > 0]
    if len(groups) < 2:
        return {}
    h, p = stats.kruskal(*groups)
    return {"kruskal_h": h, "p": p, "k": len(groups), "n": int(sum(len(g) for g in groups))}


BIVARIATE_PALETTE = {  # x tertile (1-3) then y tertile (1-3)
    "1-1": "#e8e8e8", "2-1": "#e4acac", "3-1": "#c85a5a",
    "1-2": "#b0d5df", "2-2": "#ad9ea5", "3-2": "#985356",
    "1-3": "#64acbe", "2-3": "#627f8c", "3-3": "#574249",
}


def bivariate_classes(d: pd.DataFrame, x: str, y: str) -> pd.Series:
    def tert(s):
        return pd.qcut(s.rank(method="first"), 3, labels=["1", "2", "3"]).astype(str)
    ok = d[[x, y]].notna().all(axis=1)
    out = pd.Series(np.nan, index=d.index, dtype=object)
    out[ok] = tert(d.loc[ok, x]) + "-" + tert(d.loc[ok, y])
    return out
