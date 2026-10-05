# measles-outbreak-africa

**Surveillance et prédiction des flambées de rougeole en Afrique (54 pays, 2012-2026)**
*Measles outbreak surveillance and prediction tool for Africa — English summary below.*

[![tests](https://github.com/apo1024/measles-outbreak-africa/actions/workflows/tests.yml/badge.svg)](https://github.com/apo1024/measles-outbreak-africa/actions/workflows/tests.yml)

- 🌐 **Tableau de bord en ligne** : https://apo1024.github.io/measles-outbreak-africa/
- 📘 **Guide d'utilisation** : [docs/GUIDE_UTILISATION.md](docs/GUIDE_UTILISATION.md)
- 🔬 **Méthodes** : [docs/METHODES.md](docs/METHODES.md)

Auteur : **Miracle Destine Apollon**, épidémiologiste de terrain et informaticien – Chicago, IL, USA – info@idreamlore.com

---

## Ce que fait l'outil

| Fonction | Détail |
|---|---|
| **Surveillance** | Courbes épidémiques mensuelles de 53 pays, seuil épidémique par pays (moyenne + 2 écarts-types des 24 mois précédents, minimum 10 cas), détection automatique des flambées |
| **Vulnérabilité** | Indice de Vulnérabilité Rougeole (IVR, 0-100) : déficit vaccinal (35 %), accès aux soins (20 %), démographie, pauvreté et malnutrition (15 % chacun) |
| **Modèle A – statistique** | Régression logistique (probabilité de flambée) + binomiale négative (nombre de cas du mois suivant), coefficients interprétables |
| **Modèle B – machine learning** | XGBoost (forêt aléatoire en comparaison), importance des variables |
| **Prévision** | Probabilité de flambée dans les **3 mois suivants** par pays, ensemble A+B, niveau de risque et alerte |
| **Simulation** | Scénarios « et si ? » (cas, couverture MCV1/MCV2, IVR) |
| **Analyses croisées** | Nuage X × Y avec corrélations de Spearman/Pearson et régression, tableaux croisés avec test du χ² et V de Cramér, matrice de corrélation, carte bivariée, comparaison de groupes (Kruskal-Wallis) – 31 variables, une année ou une période |
| **Districts et localités** | Limites infranationales des 54 pays : 6 681 districts (ADM2, 53 pays), 20 912 sous-districts/communes (ADM3, 29 pays), 28 925 localités administratives (ADM4, 6 pays) et 11 966 villes et villages (GeoNames) ; export KML/GeoJSON |
| **Surveillance par district** | Import d'un fichier de cas par district (CSV/Excel), rapprochement automatique des noms, seuil épidémique et alertes par district, cartes, courbes, analyse croisée entre districts |
| **Mise à jour** | GitHub Action mensuelle : téléchargement des nouvelles données OMS, réentraînement, publication du tableau de bord |

### Performance (validation temporelle : apprentissage 2013-2022, test 2023-2026)

| Modèle | AUC-ROC (3 mois) | AUC-PR | Sensibilité | Spécificité | AUC-ROC (1 mois) |
|---|---|---|---|---|---|
| A – logistique | 0,818 | 0,540 | 0,73 | 0,79 | 0,870 |
| B – XGBoost | 0,844 | 0,570 | 0,70 | 0,83 | 0,906 |
| Ensemble A+B | **0,843** | **0,574** | 0,76 | 0,80 | 0,901 |
| Naïf (persistance) | 0,655 | 0,331 | 0,36 | 0,95 | 0,735 |

Validation à origine glissante 2018-2025 : AUC moyenne 0,81 (A), 0,83 (B), 0,84 (ensemble).

## Installation rapide

```bash
git clone https://github.com/apo1024/measles-outbreak-africa.git
cd measles-outbreak-africa
python -m venv .venv
# Windows : .venv\Scripts\activate     macOS/Linux : source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

```bash
# Prévision pour tous les pays (CSV dans outputs/forecast_latest.csv)
python -m measles_predict forecast

# Historique et prévision d'un pays (code ISO3)
python -m measles_predict country NGA

# Scénario : que se passe-t-il si le Nigéria notifie 3 000 cas et que MCV1 tombe à 50 % ?
python -m measles_predict scenario NGA --cases 3000 --mcv1 50

# Validation des modèles
python -m measles_predict evaluate --rolling

# Analyses croisées (corrélation + tableau croisé)
python -m measles_predict cross --x mcv1 --y incidence_pm --logy --row ivr_class --col incidence_class --years 2023-2025
python -m measles_predict cross --list            # liste des 31 variables

# Limites infranationales et localités en KML
python -m measles_predict boundaries COD                          # niveaux disponibles
python -m measles_predict boundaries COD --level ADM2 --out rdc_districts.kml
python -m measles_predict boundaries NER --level LOCALITES --out niger_localites.kml

# Surveillance par district à partir de vos données
python -m measles_predict districts mes_cas.csv --iso3 NER --name-col district --period-col period --cases-col cases --pop-col population

# Application web interactive
streamlit run app/streamlit_app.py

# Mise à jour complète (données -> base -> modèles -> prévisions -> tableau de bord)
python scripts/update_all.py
```

Utilisation en Python :

```python
from measles_predict.pipeline import load_inputs, forecast
from measles_predict.models import load_bundle

d = load_inputs("data/model_inputs.csv")
model_a, model_b, meta = load_bundle("models")
fc = forecast(d, model_a, model_b, meta["thresholds"])
print(fc.head())
```

## Structure du dépôt

```
measles_predict/   paquet Python (features, modèles A/B, pipeline, CLI)
app/               application Streamlit (7 onglets)
docs/              tableau de bord GitHub Pages + documentation
data/              table pays x mois, tables annuelles, IVR, inventaire des limites, localités, exemples
models/            modèles entraînés (A : JSON ; B : XGBoost JSON) + métadonnées
outputs/           prévisions et performances
scripts/           collecte, construction de la base, export du tableau de bord
tests/             tests automatisés (pytest)
```

## Sources de données

- OMS – données provisoires mensuelles rougeole/rubéole par pays (immunizationdata.who.int)
- OMS GHO – couverture vaccinale WUENIC (MCV1, MCV2), cas annuels notifiés
- Banque mondiale – World Development Indicators (CC BY 4.0)
- geoBoundaries – limites ADM0 à ADM4 ; priorité aux limites humanitaires OCHA COD-AB (gbHumanitarian), sinon gbOpen (ODbL / CC BY / CC BY-IGO)
- GeoNames – villes et villages de plus de 1 000 habitants (CC BY 4.0)

## Limites importantes

Les données mensuelles OMS sont **provisoires** : les derniers mois sont incomplets (retard de notification).
L'outil ignore le dernier mois de la série et signale les pays dont le dernier mois est probablement incomplet
(`possibly_incomplete`) ou ancien (`stale`). L'analyse est nationale : une flambée infranationale peut ne pas
dépasser le seuil national. **Outil de recherche** : il complète, sans la remplacer, l'investigation de terrain.

## Citation

Apollon, M. D. (2026). *measles-outbreak-africa : surveillance et prédiction des flambées de rougeole en Afrique* (v1.0.0). https://github.com/apo1024/measles-outbreak-africa

---

## English summary

Research tool that predicts measles outbreaks (monthly cases above the country-specific epidemic threshold,
mean + 2 SD of the previous 24 months, min. 10 cases) within the next 3 months for African countries, using
WHO provisional monthly surveillance data, WUENIC immunisation coverage, World Bank indicators and a composite
Measles Vulnerability Index. Model A is a logistic regression (plus a negative binomial count model), Model B is
XGBoost; their ensemble reaches AUC 0.84 on 2023-2026 out-of-time data. Use the CLI (`python -m measles_predict
forecast`), the Streamlit app, or the static dashboard on GitHub Pages. Code: MIT licence.
