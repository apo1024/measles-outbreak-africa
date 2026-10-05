"""Subnational (district / locality) tools.

- Boundaries ADM1-ADM4 from geoBoundaries (gbHumanitarian = OCHA COD-AB first, then gbOpen),
  downloaded on demand and cached in data/geo_cache/.
- Localities (towns and villages > 1 000 inhabitants, GeoNames, CC BY 4.0).
- Tolerant matching of district names between a surveillance file and the boundaries.
- District-level surveillance: epidemic threshold and outbreak alerts per unit.
"""
from __future__ import annotations

import difflib
import re
import tempfile
import unicodedata
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
CACHE = DATA / "geo_cache"
INVENTORY = DATA / "boundaries_inventory.csv"
LOCALITIES = DATA / "localities_africa.csv.gz"
LEVEL_LABELS = {"ADM1": "Régions / provinces (ADM1)", "ADM2": "Districts / départements (ADM2)",
                "ADM3": "Sous-districts / communes (ADM3)", "ADM4": "Localités administratives (ADM4)"}
RELEASES = ["gbHumanitarian", "gbOpen"]
STOPWORDS = r"\b(district|districts|zone|zones|de|du|des|la|le|les|sante|health|province|region|regional|" \
            r"prefecture|commune|county|lga|ds|zs|area|sub|state|municipality|municipalite|departement|" \
            r"department|cercle|arrondissement|sanitaire)\b"


# ------------------------------------------------------------------ boundaries
def inventory() -> pd.DataFrame:
    return pd.read_csv(INVENTORY) if INVENTORY.exists() else pd.DataFrame(
        columns=["iso3", "level", "release", "n_units", "source", "license", "url"])


def available_levels(iso3: str) -> list[str]:
    inv = inventory()
    lv = set(inv.loc[(inv.iso3 == iso3) & (inv.level != "LOCALITES"), "level"])
    return [l for l in ("ADM1", "ADM2", "ADM3", "ADM4") if l in lv or l == "ADM1"]


def _api_url(iso3: str, level: str):
    for rel in RELEASES:
        try:
            r = requests.get(f"https://www.geoboundaries.org/api/current/{rel}/{iso3}/{level}/", timeout=60)
            if r.ok:
                js = r.json()
                return js.get("simplifiedGeometryGeoJSON") or js.get("gjDownloadURL"), rel, js.get("boundaryLicense")
        except (requests.RequestException, ValueError):
            continue
    return None, None, None


def load_boundaries(iso3: str, level: str) -> gpd.GeoDataFrame:
    """Return the ADM polygons of a country (downloaded once, then cached)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{iso3}_{level}.geojson"
    if not f.exists():
        inv = inventory()
        row = inv[(inv.iso3 == iso3) & (inv.level == level)]
        url = row.url.iloc[0] if len(row) and isinstance(row.url.iloc[0], str) else _api_url(iso3, level)[0]
        if not url:
            raise ValueError(f"Aucune limite {level} disponible pour {iso3}")
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        f.write_bytes(r.content)
    g = gpd.read_file(f).to_crs(4326)
    g["unit_name"] = g["shapeName"].astype(str) if "shapeName" in g else g.index.astype(str)
    g["unit_id"] = g["shapeID"].astype(str) if "shapeID" in g else g.index.astype(str)
    g["iso3"] = iso3
    return g[["iso3", "unit_id", "unit_name", "geometry"]]


def load_user_boundaries(file, name_field: str | None = None) -> gpd.GeoDataFrame:
    """Read a user-supplied KML / GeoJSON / zipped shapefile (e.g. health zones)."""
    suffix = Path(getattr(file, "name", str(file))).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.read() if hasattr(file, "read") else Path(file).read_bytes())
    g = gpd.read_file(tmp.name).to_crs(4326)
    cand = [name_field] if name_field else ["Name", "name", "NAME", "shapeName", "ADM2_FR", "ADM2_EN", "nom", "NOM"]
    col = next((c for c in cand if c and c in g.columns), g.columns[0])
    g["unit_name"] = g[col].astype(str)
    g["unit_id"] = g.index.astype(str)
    return g


def to_kml_bytes(g: gpd.GeoDataFrame, name_col: str = "unit_name") -> bytes:
    g = g.copy()
    g["Name"] = g[name_col].astype(str)
    for c in g.columns:
        if c != "geometry" and g[c].dtype == object:
            g[c] = g[c].astype(str)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "out.kml"
        g.to_file(p, driver="KML")
        return p.read_bytes()


# ------------------------------------------------------------------ localities
def load_localities(iso3: str | None = None, min_population: int = 0) -> gpd.GeoDataFrame:
    d = pd.read_csv(LOCALITIES)
    if iso3:
        d = d[d.iso3 == iso3]
    d = d[d.population >= min_population]
    return gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.lon, d.lat), crs=4326)


def localities_in_units(loc: gpd.GeoDataFrame, units: gpd.GeoDataFrame) -> pd.DataFrame:
    """Spatial join: which district each locality falls in."""
    j = gpd.sjoin(loc, units[["unit_id", "unit_name", "geometry"]], how="left", predicate="within")
    return pd.DataFrame(j.drop(columns=["geometry", "index_right"], errors="ignore"))


# ------------------------------------------------------------------ name matching
def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(STOPWORDS, " ", s)
    return re.sub(r"\s+", " ", s).strip()


def match_names(source: list[str], target: list[str], cutoff: float = 0.82) -> pd.DataFrame:
    """Match each source name (surveillance file) to a target name (boundaries)."""
    tnorm = {normalize(t): t for t in target}
    keys = list(tnorm)
    rows = []
    for s in source:
        n = normalize(s)
        if n in tnorm:
            rows.append({"source": s, "match": tnorm[n], "score": 1.0, "method": "exact"})
            continue
        best = difflib.get_close_matches(n, keys, n=1, cutoff=cutoff)
        if best:
            sc = difflib.SequenceMatcher(None, n, best[0]).ratio()
            rows.append({"source": s, "match": tnorm[best[0]], "score": round(sc, 3), "method": "approché"})
        else:
            rows.append({"source": s, "match": None, "score": 0.0, "method": "non trouvé"})
    return pd.DataFrame(rows)


def join_data(units: gpd.GeoDataFrame, df: pd.DataFrame, name_col: str, cutoff: float = 0.82):
    """Attach a table (one row per unit, or several periods) to the polygons."""
    m = match_names(df[name_col].dropna().astype(str).unique().tolist(), units.unit_name.tolist(), cutoff)
    d = df.merge(m[["source", "match"]], left_on=df[name_col].astype(str), right_on="source", how="left")
    d = d.drop(columns=["key_0", "source"], errors="ignore").rename(columns={"match": "unit_name_matched"})
    return d, m


# ------------------------------------------------------------------ district surveillance
def district_surveillance(df: pd.DataFrame, unit_col: str, period_col: str, cases_col: str,
                          pop_col: str | None = None, window: int = 24, min_periods: int = 6,
                          k_sd: float = 2.0, min_cases: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Epidemic threshold per unit: max(min_cases, mean + k_sd * SD of the previous `window` periods).

    Returns (all periods with flags, latest-period summary per unit).
    """
    d = df[[unit_col, period_col, cases_col] + ([pop_col] if pop_col else [])].copy()
    d.columns = ["unit", "period", "cases"] + (["population"] if pop_col else [])
    d["period"] = pd.to_datetime(d["period"].astype(str), errors="coerce")
    d = d.dropna(subset=["period"]).groupby(["unit", "period"], as_index=False).sum(numeric_only=True)
    d = d.sort_values(["unit", "period"])
    g = d.groupby("unit")["cases"]
    d["baseline_mean"] = g.transform(lambda x: x.shift(1).rolling(window, min_periods=min_periods).mean())
    d["baseline_sd"] = g.transform(lambda x: x.shift(1).rolling(window, min_periods=min_periods).std())
    d["threshold"] = np.maximum(min_cases, d.baseline_mean + k_sd * d.baseline_sd.fillna(0))
    d["outbreak"] = np.where(d.baseline_mean.notna(), (d.cases > d.threshold).astype(int), np.nan)
    d["ratio_to_threshold"] = d.cases / d.threshold
    if pop_col:
        d["incidence_100k"] = 1e5 * d.cases / d.population
    last = d.sort_values("period").groupby("unit").tail(1).copy()
    last["status"] = np.select([last.outbreak == 1, last.ratio_to_threshold >= 0.75],
                               ["Flambée", "Vigilance"], "Normal")
    last.loc[last.outbreak.isna(), "status"] = "Historique insuffisant"
    return d, last.sort_values("ratio_to_threshold", ascending=False)


def template_csv(units: gpd.GeoDataFrame, periods: list[str] | None = None) -> bytes:
    """Data-entry template: one row per unit (and per period if given)."""
    names = sorted(units.unit_name.unique())
    if periods:
        t = pd.DataFrame([(n, p) for n in names for p in periods], columns=["district", "period"])
    else:
        t = pd.DataFrame({"district": names, "period": ""})
    t["cases"] = ""
    t["population"] = ""
    return t.to_csv(index=False).encode("utf-8")
