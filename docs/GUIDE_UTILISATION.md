# Guide d'utilisation de l'outil de prédiction des flambées de rougeole

Version 1.0 – octobre 2026 – Miracle Destine Apollon (info@idreamlore.com)

Ce guide explique comment utiliser l'outil de trois façons, de la plus simple à la plus avancée :

1. **Tableau de bord en ligne** (aucune installation)
2. **Application interactive Streamlit** (sur votre ordinateur)
3. **Ligne de commande et Python** (pour les chercheurs et analystes)

---

## 1. Tableau de bord en ligne

Adresse : **https://apo1024.github.io/measles-outbreak-africa/**

| Élément | Utilisation |
|---|---|
| Indicateurs du haut | nombre de pays avec prévision, en alerte, à risque élevé, à vulnérabilité très élevée, seuil d'alerte |
| Carte | choisir la couche : probabilité de flambée à 3 mois, IVR ou couverture MCV1 ; survoler un pays pour ses valeurs ; cliquer pour ouvrir sa fiche |
| Fiche pays | dernier mois consolidé, cas, seuil épidémique, probabilités A, B et ensemble, niveau de risque, IVR, MCV1/MCV2 et courbe épidémique des 6 dernières années |
| Classement | cliquer sur un en-tête pour trier ; la colonne « Qualité » signale les données anciennes (> 3 mois) ou incomplètes |
| Calculateur (modèle A) | choisir un pays, modifier les cas, MCV1, MCV2 ou l'IVR : la probabilité est recalculée instantanément |

**Lire une prévision.** « 26 % » signifie que, parmi les situations passées comparables, environ un quart ont été
suivies d'une flambée (cas au-dessus du seuil épidémique) dans les 3 mois. Un pays est **en alerte** lorsque la
probabilité d'ensemble dépasse le seuil d'alerte (17,4 % dans la version 1.0), choisi pour équilibrer sensibilité
et spécificité.

---

## 2. Application interactive (Streamlit)

### Installation (une seule fois)

Prérequis : Python 3.10 ou plus récent, Git.

```bash
git clone https://github.com/apo1024/measles-outbreak-africa.git
cd measles-outbreak-africa
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### Lancement

```bash
streamlit run app/streamlit_app.py
```

Le navigateur s'ouvre sur http://localhost:8501. Cinq onglets :

1. **Tableau de bord** – carte (probabilité, IVR, MCV1, incidence 12 mois), classement complet, téléchargement CSV.
2. **Profil pays** – courbe épidémique avec seuil et flambées, probabilités rétrospectives des modèles, évolution de l'IVR et de ses 5 domaines.
3. **Simulateur** – scénarios « et si ? » avec les deux modèles (A et B) et l'ensemble.
4. **Performance** – métriques de validation, AUC par année, odds ratios du modèle A, importance des variables du modèle B.
5. **Méthodes** – résumé méthodologique.

Déploiement public optionnel : Streamlit Community Cloud (https://share.streamlit.io) → « New app » → dépôt
`apo1024/measles-outbreak-africa`, fichier `app/streamlit_app.py`.

---

## 3. Ligne de commande

Toutes les commandes s'exécutent à la racine du dépôt, environnement activé.

| Commande | Résultat |
|---|---|
| `python -m measles_predict forecast` | prévision pour tous les pays, enregistrée dans `outputs/forecast_latest.csv` |
| `python -m measles_predict forecast --top 10` | affiche les 10 pays au risque le plus élevé |
| `python -m measles_predict country COD` | 12 derniers mois et prévision de la RD Congo |
| `python -m measles_predict scenario NGA --cases 3000 --mcv1 50` | compare la situation observée et le scénario |
| `python -m measles_predict evaluate --rolling` | validation temporelle et à origine glissante |
| `python -m measles_predict train` | réentraîne les modèles sur toutes les données |
| `python scripts/update_all.py` | mise à jour complète : téléchargement, base, modèles, prévisions, tableau de bord |

### Colonnes du fichier de prévision

| Colonne | Signification |
|---|---|
| `iso3`, `country` | code et nom du pays |
| `period` | dernier mois consolidé utilisé comme point de départ (la prévision couvre les 3 mois suivants) |
| `cases`, `epidemic_threshold` | cas notifiés ce mois et seuil épidémique du pays |
| `outbreak_now` | 1 si le mois de départ est déjà en flambée |
| `p_A`, `p_B`, `p_ensemble` | probabilités de flambée des modèles A, B et de leur moyenne |
| `risk` | Faible (< 15 %), Modéré (15-30 %), Élevé (30-50 %), Très élevé (≥ 50 %) |
| `alert` | 1 si `p_ensemble` ≥ seuil d'alerte |
| `data_age_months`, `stale` | ancienneté des données ; `stale` = 1 si > 3 mois |
| `possibly_incomplete` | 1 si le dernier mois est < 25 % du niveau habituel (retard de notification probable) |
| `mcv1`, `mcv2`, `ivr_score` | couverture et vulnérabilité de l'année précédente |

### En Python

```python
from measles_predict.pipeline import load_inputs, forecast, latest_rows
from measles_predict.models import load_bundle
from measles_predict.features import make_features

d = load_inputs("data/model_inputs.csv")
a, b, meta = load_bundle("models")
fc = forecast(d, a, b, meta["thresholds"])

# scénario personnalisé
row = latest_rows(d[d.iso3 == "ETH"]).copy()
row["cases"] = 2500
row["outbreak"] = float(2500 > row["epidemic_threshold"].iloc[0])
print(forecast(d, a, b, meta["thresholds"], rows=make_features(row))[["p_A", "p_B", "p_ensemble"]])
```

---

## 4. Utiliser vos propres données (par exemple infranationales)

Le modèle fonctionne sur toute table au format de `data/model_inputs.csv` (une ligne = unité spatiale × mois).
Colonnes minimales : `iso3` (ou identifiant de district), `year`, `month`, `period` (AAAA-MM), `cases`,
`population` (via `log_population`), `mcv1`, `mcv2`, `ivr_score`, `subregion`. Les colonnes dérivées (retards,
seuils, flambées) peuvent être recalculées avec `scripts/02_build_database.py` (fonction `build_model_inputs`).
Les fichiers KML ADM1 fournis avec l'étude permettent ensuite la cartographie dans QGIS.
Pour des districts, réentraînez les modèles (`python -m measles_predict --data votre_fichier.csv train`) :
les seuils et coefficients nationaux ne sont pas directement transposables.

## 5. Mise à jour automatique

Le workflow GitHub `monthly-update` s'exécute le 15 de chaque mois (ou manuellement : onglet *Actions* →
*monthly-update* → *Run workflow*). Il télécharge les nouvelles données, réentraîne, teste et publie le tableau de bord.

## 6. Bonnes pratiques d'interprétation

- Toujours vérifier les colonnes `stale` et `possibly_incomplete` avant d'agir.
- Une probabilité faible n'exclut pas une flambée locale (analyse nationale).
- Croiser avec l'IVR : un pays très vulnérable avec une probabilité modérée mérite une vigilance accrue.
- Confirmer tout signal par l'investigation de terrain et les données de laboratoire.

## 7. Dépannage

| Problème | Solution |
|---|---|
| `ModuleNotFoundError` | activer l'environnement puis `pip install -r requirements.txt` |
| La collecte échoue (URL OMS) | vérifier l'adresse dans `scripts/01_collect_data.py` (`WHO_MONTHLY_URL`) |
| Port 8501 occupé | `streamlit run app/streamlit_app.py --server.port 8502` |
| Le tableau de bord en ligne ne change pas | vérifier l'exécution du workflow dans l'onglet *Actions* |

Contact : info@idreamlore.com
