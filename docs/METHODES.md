# Méthodes

## 1. Unité d'analyse et période
Pays × mois, 53 pays africains disposant de données mensuelles OMS (Maurice n'en a pas), janvier 2012 à la
dernière période disponible (septembre 2026 pour la version 1.0). 9 381 pays-mois.

## 2. Données
| Source | Variables |
|---|---|
| OMS, données provisoires mensuelles | cas suspects, cliniques, épi-liés, confirmés en laboratoire, total, écartés |
| OMS GHO / WUENIC | couverture MCV1 et MCV2 (%) annuelle |
| Banque mondiale (WDI) | population totale et 0-14 ans, densité, urbanisation, natalité, PIB/hab, pauvreté (2,15 $), mortalité < 5 ans, retard de croissance, dépenses de santé/hab, médecins/1000, accouchements assistés |
| geoBoundaries | limites ADM0 et ADM1 (fichiers KML / GeoJSON) |

Valeurs manquantes des indicateurs annuels : interpolation linéaire intra-pays, prolongation aux extrémités,
puis médiane sous-région × année. MCV2 manquant = vaccin non introduit (0 %).

## 3. Définition d'une flambée
Seuil épidémique du pays au mois *t* : `max(10 ; moyenne + 2 × écart-type des cas des 24 mois précédents)`
(au moins 12 mois observés). Flambée au mois *t* si `cas(t) > seuil(t)`. Cette définition repère les
**poussées** au-dessus du canal endémique ; un pays à forte transmission continue peut donc avoir une incidence
élevée sans « flambée » au sens de l'outil.

Cibles prédites : `outbreak_next3` (flambée au cours des mois t+1 à t+3, cible principale) et `outbreak_next1`.

## 4. Indice de Vulnérabilité Rougeole (IVR)
Chaque variable est normalisée 0-1 (bornes = percentiles 2,5 et 97,5 de l'ensemble pays-années 2010-2025,
orientée pour que 1 = défavorable), moyennée dans son domaine, puis pondérée :

| Domaine | Poids | Variables |
|---|---|---|
| Déficit vaccinal | 0,35 | MCV1, MCV2 (inversées) |
| Accès aux soins | 0,20 | mortalité < 5 ans ; accouchements assistés, dépenses de santé, médecins (inversés) |
| Démographie | 0,15 | % 0-14 ans, natalité, log densité |
| Pauvreté | 0,15 | log PIB/hab (inversé), pauvreté 2,15 $ |
| Malnutrition | 0,15 | retard de croissance < 5 ans |

`IVR = 100 × Σ poids × domaine`. Classes = quartiles de la distribution 2010-2025 : Faible, Modérée, Élevée, Très élevée.

## 5. Variables prédictives
Retards des cas (log), somme des 3 et 12 mois précédents, moyenne de référence, rapport cas/seuil, flambée en
cours, mois depuis la dernière flambée, saisonnalité (sinus/cosinus), MCV1, MCV2 et IVR **de l'année précédente**
(pas de fuite d'information : la couverture de l'année N n'est publiée qu'en juillet N+1), structure d'âge,
densité, urbanisation, population, sous-région.

## 6. Modèles
- **Modèle A** : GLM binomial (logistique) sur 11 variables standardisées ; odds ratios par écart-type.
  Complément : GLM binomial négatif (α = 0,5) pour le nombre de cas du mois suivant.
- **Modèle B** : XGBoost (500 arbres, profondeur 4, taux 0,03, sous-échantillonnage 0,8) sur 22 variables ;
  forêt aléatoire (500 arbres) en comparaison.
- **Ensemble** : moyenne des probabilités A et B.
- **Seuil d'alerte** : indice de Youden sur les données d'apprentissage.
- **Niveaux de risque** : Faible < 15 %, Modéré 15-30 %, Élevé 30-50 %, Très élevé ≥ 50 %.

## 7. Validation
- Validation temporelle : apprentissage 2013-2022, test 2023-2026.
- Validation à origine glissante : pour chaque année Y de 2018 à 2025, apprentissage sur les années < Y.
- Métriques : AUC-ROC, AUC-PR, score de Brier, sensibilité, spécificité, VPP, VPN ; comparaison avec un modèle
  naïf (persistance de la flambée en cours).

## 8. Garde-fous opérationnels
- Le dernier mois de la série provisoire n'est pas utilisé comme origine de prévision (délai de consolidation).
- `possibly_incomplete` : dernier mois < 25 % de la moyenne du même mois des 3 années précédentes (si ≥ 20 cas).
- `stale` : dernière donnée consolidée de plus de 3 mois.

## 9. Limites
Données agrégées nationales et provisoires ; sous-notification variable ; pas de données infranationales
publiques (le portail AFRO de liste linéaire était indisponible lors de la collecte) ; campagnes de vaccination
de masse (AVS), mobilité et conflits non modélisés ; IVR pondéré a priori.

## 10. Analyses croisées
Unité : pays, pour une année ou la moyenne d'une période. Corrélations de Pearson (échelle affichée, log10 facultatif) et de Spearman ; régression linéaire simple ; tableaux de contingence avec test du χ² d'indépendance et V de Cramér ; test de Kruskal-Wallis entre groupes ; carte bivariée par tertiles. Classes : MCV1 (< 60, 60-79, 80-89, ≥ 90 %), incidence (< 5, 5-20, 20-50, 50-100, ≥ 100 cas/million), IVR (quartiles), risque prédit (< 15, 15-30, 30-50, ≥ 50 %). Il s'agit d'associations écologiques.

## 11. Données infranationales
Limites : geoBoundaries, version humanitaire (OCHA COD-AB) en priorité, sinon gbOpen ; géométries simplifiées. Localités : GeoNames « cities1000 » (lieux habités de plus de 1 000 habitants). Surveillance par district : seuil = max(minimum de cas ; moyenne + k × écart-type des périodes de la fenêtre), au moins 6 périodes d'historique ; « Vigilance » si cas ≥ 75 % du seuil. Rapprochement des noms : normalisation (accents, casse, ponctuation, mots génériques) puis similarité de chaînes (difflib) au-dessus d'un seuil réglable.
