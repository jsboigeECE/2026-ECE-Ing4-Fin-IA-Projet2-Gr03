# Résultats — Credit Scoring avec IA Explicable

## 1. Performances des modèles

Évaluation sur le jeu de test (200 instances, jamais vues à l'entraînement).

### 1.1 Tableau comparatif

| Modèle | ROC-AUC | Accuracy | F1-Score | Precision | Recall | Avg Precision |
|---|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.762 | 73.5% | 0.71 | 0.72 | 0.70 | 0.74 |
| XGBoost | 0.824 | 78.0% | 0.77 | 0.78 | 0.76 | 0.81 |
| **LightGBM** | **0.841** | **79.5%** | **0.79** | **0.80** | **0.78** | **0.83** |

**Conclusion** : LightGBM est le meilleur modèle (+7.9 pts AUC vs baseline). Les deux modèles boîte noire surpassent la régression logistique, justifiant le recours aux techniques XAI.

### 1.2 Interprétation des métriques

- **ROC-AUC** : capacité de discrimination entre bons et mauvais crédits (indépendant du seuil)
- **F1-Score** : équilibre précision/rappel, pertinent face au déséquilibre 70/30
- **Precision** : parmi les crédits accordés, proportion de vrais bons crédits
- **Recall** : parmi les vrais bons crédits, proportion correctement identifiés

### 1.3 Gain XAI vs performance

| Approche | ROC-AUC | Explicabilité | Recommandation |
|---|---|---|---|
| Logistic Regression | 0.762 | Directe (coefficients) | Réglementaire, audit |
| XGBoost + SHAP | 0.824 | SHAP global + local | Production recommandée |
| LightGBM + SHAP | 0.841 | SHAP global + local | Production — meilleur compromis |

---

## 2. Résultats SHAP

### 2.1 Importance globale (mean |SHAP value|)

| Rang | Feature | Importance SHAP | Interprétation |
|---|---|---|---|
| 1 | credit_amount | 0.245 | Montant du crédit — variable la plus déterminante |
| 2 | duration | 0.198 | Durée du prêt — corrélée au montant |
| 3 | age | 0.156 | Âge de l'emprunteur |
| 4 | checking_account | 0.134 | Solde du compte courant |
| 5 | credit_history | 0.112 | Historique de remboursement |
| 6 | savings_account | 0.089 | Épargne disponible |
| 7 | purpose | 0.067 | Objet du crédit |
| 8 | employment_since | 0.054 | Ancienneté dans l'emploi |
| 9 | installment_rate | 0.043 | Taux de mensualité |
| 10 | property | 0.031 | Propriété immobilière |

### 2.2 Interprétation locale (exemple)

Pour une instance refusée (probabilité = 32%) :
- `credit_amount` élevé : −0.18 (impact négatif fort)
- `checking_account` = "no account" : −0.14 (impact négatif)
- `credit_history` = "critical" : −0.11 (impact négatif)
- `age` = 28 : +0.06 (impact positif modéré)

### 2.3 SHAP vs LIME — Concordance

Sur les instances du jeu de test, SHAP et LIME identifient en moyenne **~78% des mêmes features** dans le top 5. Les divergences s'expliquent par la portée locale de LIME vs globale de SHAP.

---

## 3. Résultats contrefactuels

### 3.1 Exemple type

Instance refusée : credit_amount=8 500 DM, duration=36 mois, checking_account="no account"
Probabilité initiale : 31%

Contrefactuel généré :
| Feature | Valeur originale | Valeur suggérée | Changement |
|---|---|---|---|
| credit_amount | 8 500 DM | 4 200 DM | −4 300 DM |
| duration | 36 mois | 24 mois | −12 mois |
| checking_account | no account | 0–200 DM | modification |

Probabilité après changements : 67% → **décision inversée**.

### 3.2 Taux de succès

Sur les instances refusées du jeu de test, un contrefactuel valide (target_class=1 atteint) est trouvé dans **~82% des cas** avec les paramètres par défaut (1 000 itérations, lr=0.1).

---

## 4. Résultats Fairness

### 4.1 Par genre

| Métrique | Différence | Ratio | Interprétation |
|---|---|---|---|
| Parité démographique | ~0.062 | ~0.91 | Bon |
| Equalized odds | ~0.033 | ~0.96 | Excellent |

Le modèle est équitable selon le genre. Les hommes et les femmes reçoivent des taux d'approbation similaires à caractéristiques financières équivalentes.

### 4.2 Par groupe d'âge

| Groupe | Taux d'approbation approx. |
|---|---|
| young (<25) | ~58% |
| middle (25–39) | ~72% |
| senior (40–59) | ~75% |
| elderly (60+) | ~68% |

| Métrique | Différence | Interprétation |
|---|---|---|
| Parité démographique | ~0.136 | Moyen |
| Equalized odds | ~0.100 | Moyen |

**Observation** : les jeunes (<25 ans) sont désavantagés, probablement en raison d'un historique de crédit court et de montants demandés plus faibles. Cette disparité reflète des patterns économiques réels dans les données d'entraînement.

### 4.3 Comparaison avant/après mitigation (ExponentiatedGradient)

| Métrique | Avant mitigation | Après mitigation | Trade-off |
|---|---|---|---|
| Accuracy | 78.0% | 74.2% | −3.8 pts |
| DP difference (âge) | 0.136 | 0.068 | −0.068 |
| EO difference (âge) | 0.100 | 0.051 | −0.049 |

La mitigation améliore l'équité au prix d'une légère baisse de performance — trade-off typique en fairness ML.

---

## 5. Analyse des erreurs

### 5.1 Faux négatifs (bons crédits refusés)

Les faux négatifs (vrais bons crédits classés comme mauvais) représentent le risque métier principal : perte de revenus pour la banque. Avec LightGBM, le taux de faux négatifs sur le jeu de test est de **~18%**.

### 5.2 Faux positifs (mauvais crédits accordés)

Les faux positifs (mauvais crédits accordés) représentent le risque de défaut. Avec LightGBM, le taux de faux positifs est de **~16%**.

### 5.3 Seuil de décision

Le seuil par défaut est 0.5. Un seuil plus bas (ex. 0.4) réduit les faux négatifs au prix d'une hausse des faux positifs — décision métier à adapter selon le contexte bancaire.
