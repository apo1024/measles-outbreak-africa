"""
01_collecte_donnees.py
Collecte des données brutes pour l'étude « Rougeole en Afrique ».

Sources (toutes publiques, sans authentification) :
  1. OMS – données provisoires mensuelles rougeole/rubéole par pays (epi-curve)
  2. OMS GHO API – cas annuels notifiés (WHS3_62), couverture MCV1 (WHS8_110), MCV2 (MCV2)
  3. Banque mondiale API – démographie, socio-économie, santé
  4. geoBoundaries – limites administratives ADM0 et ADM1 (GeoJSON simplifié)

Auteur : Miracle Destine Apollon – info@idreamlore.com
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]   # repository root
RAW = ROOT / "data" / "raw"
GEO = ROOT / "data" / "geo"
RAW.mkdir(parents=True, exist_ok=True)
GEO.mkdir(parents=True, exist_ok=True)

# 54 États africains (ISO3, nom FR, sous-région UA)
AFRICA = [
    ("DZA", "Algérie", "Afrique du Nord"), ("EGY", "Égypte", "Afrique du Nord"),
    ("LBY", "Libye", "Afrique du Nord"), ("MAR", "Maroc", "Afrique du Nord"),
    ("TUN", "Tunisie", "Afrique du Nord"), ("MRT", "Mauritanie", "Afrique du Nord"),
    ("BEN", "Bénin", "Afrique de l'Ouest"), ("BFA", "Burkina Faso", "Afrique de l'Ouest"),
    ("CPV", "Cabo Verde", "Afrique de l'Ouest"), ("CIV", "Côte d'Ivoire", "Afrique de l'Ouest"),
    ("GMB", "Gambie", "Afrique de l'Ouest"), ("GHA", "Ghana", "Afrique de l'Ouest"),
    ("GIN", "Guinée", "Afrique de l'Ouest"), ("GNB", "Guinée-Bissau", "Afrique de l'Ouest"),
    ("LBR", "Libéria", "Afrique de l'Ouest"), ("MLI", "Mali", "Afrique de l'Ouest"),
    ("NER", "Niger", "Afrique de l'Ouest"), ("NGA", "Nigéria", "Afrique de l'Ouest"),
    ("SEN", "Sénégal", "Afrique de l'Ouest"), ("SLE", "Sierra Leone", "Afrique de l'Ouest"),
    ("TGO", "Togo", "Afrique de l'Ouest"),
    ("CMR", "Cameroun", "Afrique centrale"), ("CAF", "République centrafricaine", "Afrique centrale"),
    ("TCD", "Tchad", "Afrique centrale"), ("COG", "Congo", "Afrique centrale"),
    ("COD", "RD Congo", "Afrique centrale"), ("GNQ", "Guinée équatoriale", "Afrique centrale"),
    ("GAB", "Gabon", "Afrique centrale"), ("STP", "Sao Tomé-et-Principe", "Afrique centrale"),
    ("BDI", "Burundi", "Afrique centrale"),
    ("COM", "Comores", "Afrique de l'Est"), ("DJI", "Djibouti", "Afrique de l'Est"),
    ("ERI", "Érythrée", "Afrique de l'Est"), ("ETH", "Éthiopie", "Afrique de l'Est"),
    ("KEN", "Kenya", "Afrique de l'Est"), ("MDG", "Madagascar", "Afrique de l'Est"),
    ("MUS", "Maurice", "Afrique de l'Est"), ("RWA", "Rwanda", "Afrique de l'Est"),
    ("SYC", "Seychelles", "Afrique de l'Est"), ("SOM", "Somalie", "Afrique de l'Est"),
    ("SSD", "Soudan du Sud", "Afrique de l'Est"), ("SDN", "Soudan", "Afrique de l'Est"),
    ("TZA", "Tanzanie", "Afrique de l'Est"), ("UGA", "Ouganda", "Afrique de l'Est"),
    ("AGO", "Angola", "Afrique australe"), ("BWA", "Botswana", "Afrique australe"),
    ("SWZ", "Eswatini", "Afrique australe"), ("LSO", "Lesotho", "Afrique australe"),
    ("MWI", "Malawi", "Afrique australe"), ("MOZ", "Mozambique", "Afrique australe"),
    ("NAM", "Namibie", "Afrique australe"), ("ZAF", "Afrique du Sud", "Afrique australe"),
    ("ZMB", "Zambie", "Afrique australe"), ("ZWE", "Zimbabwe", "Afrique australe"),
]
ISO = [a[0] for a in AFRICA]

WHO_MONTHLY_URL = ("https://immunizationdata.who.int/docs/librariesprovider21/"
                   "measles-and-rubella/404-table-web-epi-curve-data.xlsx")
GHO = "https://ghoapi.azureedge.net/api/{code}"
GHO_INDICATORS = {"WHS3_62": "measles_cases_reported", "WHS8_110": "mcv1", "MCV2": "mcv2"}

WB = "https://api.worldbank.org/v2/country/{iso}/indicator/{ind}?format=json&per_page=2000&date=2005:2025"
WB_INDICATORS = {
    "SP.POP.TOTL": "population_total",
    "SP.POP.0014.TO": "population_0_14",
    "SP.URB.TOTL.IN.ZS": "urban_pct",
    "EN.POP.DNST": "pop_density",
    "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
    "SH.DYN.MORT": "under5_mortality",
    "SP.DYN.CBRT.IN": "birth_rate",
    "SH.STA.STNT.ME.ZS": "stunting_pct",
    "SH.XPD.CHEX.PC.CD": "health_exp_per_capita",
    "SI.POV.DDAY": "poverty_215_pct",
    "SM.POP.REFG": "refugee_population",
    "SH.MED.PHYS.ZS": "physicians_per_1000",
    "SH.STA.BRTC.ZS": "skilled_birth_attendance",
}

S = requests.Session()
S.headers["User-Agent"] = "measles-africa-study/1.0 (info@idreamlore.com)"


def get(url, **kw):
    for attempt in range(5):
        try:
            r = S.get(url, timeout=90, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == 4:
                raise
            print(f"  retry {attempt + 1}: {e}")
            time.sleep(2 * (attempt + 1))


def collect_countries():
    df = pd.DataFrame(AFRICA, columns=["iso3", "country_fr", "subregion"])
    df.to_csv(RAW / "countries.csv", index=False)
    print(f"[pays] {len(df)} pays")


def collect_who_monthly():
    out = RAW / "who_measles_monthly_epicurve.xlsx"
    if not out.exists():
        out.write_bytes(get(WHO_MONTHLY_URL).content)
    d = pd.read_excel(out, sheet_name="WEB")
    d.columns = [str(c).replace("\n", "").replace(" ", "_").lower() for c in d.columns]
    d = d[d["iso3"].isin(ISO)].copy()
    d.to_csv(RAW / "who_measles_monthly_africa.csv", index=False)
    print(f"[OMS mensuel] {len(d)} lignes, {d['iso3'].nunique()} pays, "
          f"{d['year'].min()}-{d['year'].max()}")


def collect_gho():
    frames = []
    for code, name in GHO_INDICATORS.items():
        js = get(GHO.format(code=code)).json()["value"]
        d = pd.DataFrame(js)
        d = d[(d["SpatialDimType"] == "COUNTRY") & d["SpatialDim"].isin(ISO)]
        d = d[["SpatialDim", "TimeDim", "NumericValue"]].rename(
            columns={"SpatialDim": "iso3", "TimeDim": "year", "NumericValue": name})
        frames.append(d.set_index(["iso3", "year"]))
        print(f"[GHO] {code}: {len(d)} valeurs")
    g = pd.concat(frames, axis=1).reset_index().sort_values(["iso3", "year"])
    g.to_csv(RAW / "who_gho_annual.csv", index=False)


def collect_worldbank():
    rows = []
    for ind, name in WB_INDICATORS.items():
        # requête groupée par lots de pays (séparés par ;)
        for i in range(0, len(ISO), 20):
            batch = ";".join(ISO[i:i + 20])
            url = WB.format(iso=batch, ind=ind)
            js = get(url).json()
            if len(js) < 2 or js[1] is None:
                continue
            for rec in js[1]:
                if rec["value"] is not None:
                    rows.append({"iso3": rec["countryiso3code"], "year": int(rec["date"]),
                                 "indicator": name, "value": rec["value"]})
        print(f"[BM] {ind} -> {name}")
    d = pd.DataFrame(rows)
    w = d.pivot_table(index=["iso3", "year"], columns="indicator", values="value").reset_index()
    w.to_csv(RAW / "worldbank_annual.csv", index=False)
    print(f"[BM] {len(w)} pays-années")


def collect_boundaries():
    meta = []
    for iso in ISO:
        for adm in ("ADM0", "ADM1"):
            out = GEO / f"{iso}_{adm}.geojson"
            if out.exists():
                continue
            try:
                info = get(f"https://www.geoboundaries.org/api/current/gbOpen/{iso}/{adm}/").json()
                url = info.get("simplifiedGeometryGeoJSON") or info["gjDownloadURL"]
                out.write_bytes(get(url).content)
                meta.append({"iso3": iso, "level": adm, "source": info.get("boundarySource"),
                             "license": info.get("boundaryLicense"), "year": info.get("boundaryYearRepresented")})
                print(f"[geoBoundaries] {iso} {adm}")
            except Exception as e:  # noqa: BLE001
                print(f"[geoBoundaries] ÉCHEC {iso} {adm}: {e}")
    if meta:
        pd.DataFrame(meta).to_csv(GEO / "boundaries_metadata.csv", index=False)


if __name__ == "__main__":
    collect_countries()
    collect_who_monthly()
    collect_gho()
    collect_worldbank()
    if "--boundaries" in __import__("sys").argv:
        collect_boundaries()  # optional: per-country ADM0/ADM1 GeoJSON
    (RAW / "collecte_log.json").write_text(json.dumps(
        {"date_collecte": pd.Timestamp.now().isoformat(), "sources": {
            "who_monthly": WHO_MONTHLY_URL, "gho": list(GHO_INDICATORS),
            "worldbank": list(WB_INDICATORS), "boundaries": "geoboundaries.org gbOpen"}},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print("Collecte terminée.")
