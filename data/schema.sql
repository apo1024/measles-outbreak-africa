-- ============================================================
-- Base de données : rougeole_afrique.sqlite
-- Étude : Modélisation spatio-temporelle et prédiction des
--         flambées de rougeole en Afrique (2012-2026)
-- Auteur : Miracle Destine Apollon (info@idreamlore.com)
-- Compatible SQLite / PostgreSQL (adapter les types si besoin)
-- ============================================================

DROP TABLE IF EXISTS model_inputs;
DROP TABLE IF EXISTS vulnerability_index;
DROP TABLE IF EXISTS socioeconomic;
DROP TABLE IF EXISTS vaccination;
DROP TABLE IF EXISTS measles_annual;
DROP TABLE IF EXISTS measles_monthly;
DROP TABLE IF EXISTS boundaries;
DROP TABLE IF EXISTS countries;
DROP TABLE IF EXISTS data_sources;

-- Table 1 : pays (unité spatiale d'analyse)
CREATE TABLE countries (
    iso3            TEXT PRIMARY KEY,
    country_fr      TEXT NOT NULL,
    country_en      TEXT,
    subregion       TEXT NOT NULL,      -- sous-région Union africaine
    who_region      TEXT,               -- AFR ou EMR
    centroid_lon    REAL,
    centroid_lat    REAL,
    area_km2        REAL
);

-- Table 2 : limites administratives (fichiers KML / GeoJSON)
CREATE TABLE boundaries (
    iso3            TEXT REFERENCES countries(iso3),
    adm_level       TEXT,               -- ADM0 / ADM1
    n_units         INTEGER,
    kml_path        TEXT,
    geojson_path    TEXT,
    source          TEXT,
    license         TEXT,
    PRIMARY KEY (iso3, adm_level)
);

-- Table 3 : cas mensuels de rougeole (surveillance OMS, provisoire)
CREATE TABLE measles_monthly (
    iso3                TEXT REFERENCES countries(iso3),
    year                INTEGER,
    month               INTEGER CHECK (month BETWEEN 1 AND 12),
    suspect             INTEGER,
    clinical            INTEGER,
    epi_linked          INTEGER,
    lab_confirmed       INTEGER,
    total_cases         INTEGER,        -- clinique + épi-lié + labo
    discarded           INTEGER,
    rubella_total       INTEGER,
    PRIMARY KEY (iso3, year, month)
);

-- Table 4 : cas annuels officiels (JRF OMS/UNICEF)
CREATE TABLE measles_annual (
    iso3            TEXT REFERENCES countries(iso3),
    year            INTEGER,
    reported_cases  INTEGER,
    PRIMARY KEY (iso3, year)
);

-- Table 5 : couverture vaccinale (WUENIC, %)
CREATE TABLE vaccination (
    iso3    TEXT REFERENCES countries(iso3),
    year    INTEGER,
    mcv1    REAL,
    mcv2    REAL,
    PRIMARY KEY (iso3, year)
);

-- Table 6 : indicateurs démographiques et socio-économiques (Banque mondiale)
CREATE TABLE socioeconomic (
    iso3                      TEXT REFERENCES countries(iso3),
    year                      INTEGER,
    population_total          REAL,
    population_0_14           REAL,
    children_pct              REAL,
    urban_pct                 REAL,
    pop_density               REAL,
    birth_rate                REAL,
    gdp_per_capita_usd        REAL,
    poverty_215_pct           REAL,
    under5_mortality          REAL,
    stunting_pct              REAL,
    health_exp_per_capita     REAL,
    physicians_per_1000       REAL,
    skilled_birth_attendance  REAL,
    imputed_flag              TEXT,      -- liste des variables imputées
    PRIMARY KEY (iso3, year)
);

-- Table 7 : Indice de Vulnérabilité Rougeole (IVR)
CREATE TABLE vulnerability_index (
    iso3            TEXT REFERENCES countries(iso3),
    year            INTEGER,
    d_immunite      REAL,   -- domaine 1 : déficit vaccinal
    d_demographie   REAL,   -- domaine 2 : pression démographique
    d_socioeco      REAL,   -- domaine 3 : pauvreté
    d_nutrition     REAL,   -- domaine 4 : malnutrition
    d_acces_soins   REAL,   -- domaine 5 : accès aux soins
    ivr_score       REAL,   -- 0 (faible) à 100 (très élevé)
    ivr_class       TEXT,   -- Faible / Modérée / Élevée / Très élevée
    ivr_rank        INTEGER,
    PRIMARY KEY (iso3, year)
);

-- Table 8 : table spatio-temporelle d'entrée des modèles (pays x mois)
CREATE TABLE model_inputs (
    iso3                TEXT REFERENCES countries(iso3),
    year                INTEGER,
    month               INTEGER,
    period              TEXT,           -- AAAA-MM
    reported            INTEGER,        -- 1 si le pays a notifié ce mois
    cases               REAL,
    incidence_pm        REAL,           -- cas par million d'habitants
    cases_lag1          REAL,
    cases_lag2          REAL,
    cases_lag3          REAL,
    cases_sum_3m        REAL,
    cases_sum_12m       REAL,
    baseline_mean_24m   REAL,
    baseline_sd_24m     REAL,
    epidemic_threshold  REAL,           -- max(10, moyenne + 2 ET)
    outbreak            INTEGER,        -- flambée au mois t
    outbreak_next1      INTEGER,        -- flambée au mois t+1
    outbreak_next3      INTEGER,        -- flambée entre t+1 et t+3
    cases_next1         REAL,
    months_since_outbreak REAL,
    month_sin           REAL,
    month_cos           REAL,
    mcv1                REAL,
    mcv2                REAL,
    ivr_score           REAL,
    children_pct        REAL,
    pop_density         REAL,
    urban_pct           REAL,
    log_population      REAL,
    subregion           TEXT,
    PRIMARY KEY (iso3, year, month)
);

-- Table 9 : traçabilité des sources
CREATE TABLE data_sources (
    source_id   TEXT PRIMARY KEY,
    description TEXT,
    url         TEXT,
    licence     TEXT,
    accessed    TEXT
);

CREATE INDEX idx_mi_period ON model_inputs(period);
CREATE INDEX idx_mm_year ON measles_monthly(year);
