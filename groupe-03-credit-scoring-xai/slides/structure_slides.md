# Structure des Slides de Présentation

## 📊 Plan de la Présentation (15-20 min)

### Slide 1: Titre et Introduction (1 min)
- **Titre** : Credit Scoring avec IA Explicable (XAI)
- **Sous-titre** : ECE Paris - Ing4 Finance - IA Probabiliste, Théorie des Jeux et Machine Learning
- **Auteurs** : MALAK El-Idrissi
- **Date** : 30 mars 2026

### Slide 2: Contexte du Projet (1 min)
- **Cours** : IA Probabiliste, Théorie des Jeux et Machine Learning
- **Sujet** : C.6 - Credit Scoring avec IA Explicable (XAI)
- **Objectifs** : Développer un système de scoring de crédit avec XAI

### Slide 3: Problématique du Credit Scoring (1 min)
- **Importance** : Gestion du risque, automatisation, conformité réglementaire
- **Défis** :
  - Transparence des décisions (RGPD)
  - Équité (non-discrimination)
  - Performance et précision

### Slide 4: Dataset - German Credit (1 min)
- **Source** : UCI Machine Learning Repository
- **Caractéristiques** :
  - 1000 instances
  - 20 attributs (7 numériques, 13 catégoriels)
  - Variable cible : Credit risk (Good/Bad)
- **Distribution** : 70% Good Credit, 30% Bad Credit

### Slide 5: Méthodologie - Pipeline (1 min)
```
Données → Prétraitement → Modélisation → Explicabilité → Fairness → Dashboard
```

### Slide 6: Modèles Implémentés (1 min)
- **Baseline** : Logistic Regression (interprétable)
- **XGBoost** : Gradient Boosting (boîte noire)
- **LightGBM** : Leaf-wise Gradient Boosting (boîte noire)

### Slide 7: Résultats - Comparaison des Modèles (1 min)
| Modèle | Accuracy | F1-Score | ROC-AUC |
|--------|----------|----------|----------|
| Logistic Regression | 73.5% | 81.8% | 76.2% |
| XGBoost | 78.0% | 84.8% | 82.4% |
| LightGBM | 79.5% | 85.9% | 84.1% |

**Meilleur modèle** : LightGBM (ROC-AUC: 84.1%)

### Slide 8: Importance des Features (1 min)
**Top 5 Features** :
1. Credit Amount
2. Duration
3. Age
4. Checking Account
5. Credit History

### Slide 9: Explicabilité - SHAP (1 min)
- **Théorie** : Valeurs de Shapley (théorie des jeux)
- **Propriétés** : Efficacité, symétrie, dummy, additivité
- **Visualisations** : Summary plot, waterfall plot, force plot

### Slide 10: Explicabilité - LIME (1 min)
- **Principe** : Approximation locale linéaire
- **Avantages** : Model-agnostic, rapide, intuitif
- **Comparaison** : SHAP vs LIME (concordance ~80%)

### Slide 11: Explications Contrefactuelles (1 min)
- **Question** : "Que faudrait-il changer pour être accepté ?"
- **Exemple** : Réduire le montant, raccourcir la durée
- **Actionnabilité** : Changements réalistes et faisables

### Slide 12: Fairness - Audit (1 min)
- **Parité Démographique** :
  - Genre : Différence = 0.062 (Bon)
  - Âge : Différence = 0.136 (Moyen)
- **Égalité des Chances** :
  - Genre : Différence = 0.033 (Bon)
  - Âge : Différence = 0.100 (Moyen)

### Slide 13: Dashboard Streamlit (1 min)
- **Page Accueil** : Statistiques et distributions
- **Page Prédiction** : Interface interactive
- **Page Explicabilité** : SHAP, LIME, Contrefactuels
- **Page Fairness** : Audit des disparités
- **Page Comparaison** : Tableaux et graphiques

### Slide 14: Démonstration Live (3-5 min)
1. **Prédiction** : Sélectionner une instance et afficher la prédiction
2. **Explication SHAP** : Montrer les contributions des features
3. **Explication LIME** : Comparer avec SHAP
4. **Contrefactuel** : Générer une explication "que faudrait-il changer"
5. **Fairness** : Afficher l'audit par genre et âge

### Slide 15: Conclusion (1 min)
- **Résumé** :
  - Modèle performant (LightGBM : ROC-AUC 84.1%)
  - Explicabilité complète (SHAP, LIME, Contrefactuels)
  - Fairness audité (disparités modérées)
  - Dashboard interactif fonctionnel
- **Perspectives** :
  - Optimisation des hyperparamètres
  - Atténuation des biais
  - Nouveaux modèles (CatBoost, Neural Networks)
  - MLOps et déploiement

### Slide 16: Questions/Réponses (2 min)
- Session de questions
- Réponses aux interrogations

---

## 🎯 Points Clés à Mettre en Avant

### Qualité de la Présentation
- ✅ Structure claire et logique
- ✅ Visuels professionnels (graphiques, tableaux)
- ✅ Temps respecté (15-20 min)
- ✅ Démonstration live fluide
- ✅ Transitions entre slides

### Qualité Théorique
- ✅ Explication des algorithmes (XGBoost, LightGBM)
- ✅ Contexte du credit scoring
- ✅ Explication de SHAP (théorie des jeux)
- ✅ Explication de LIME (approximation locale)
- ✅ Explication du fairness (parité démographique, égalité des chances)

### Qualité Technique
- ✅ Code propre et documenté
- ✅ Architecture modulaire
- ✅ Résultats mesurables
- ✅ Dashboard fonctionnel
- ✅ Tests unitaires

### Organisation
- ✅ Planning respecté
- ✅ Documentation complète
- ✅ Livrables prêts
- ✅ Collaboration GitHub

---

## 📝 Notes pour la Présentation

### Introduction
- Commencer par le contexte du cours
- Présenter le sujet et ses enjeux
- Annoncer le plan de la présentation

### Démonstration
- Préparer les instances à montrer
- Avoir le dashboard ouvert en arrière-plan
- Tester la démo avant la présentation

### Questions
- Anticiper les questions possibles :
  - Pourquoi LightGBM plutôt que XGBoost ?
  - Comment gérer le déséquilibre de classe ?
  - Quelles sont les limites de SHAP/LIME ?
  - Comment améliorer le fairness ?

### Conclusion
- Résumer les points clés
- Mettre en avant les contributions
- Ouvrir sur les perspectives

---

## 🎨 Conseils de Design

### Couleurs
- **Primaire** : #1f77b4 (bleu)
- **Secondaire** : #2ecc71 (vert)
- **Accent** : #e74c3c (rouge)
- **Neutre** : #34495e (gris foncé)

### Polices
- **Titres** : Arial, 32-36 pt, gras
- **Sous-titres** : Arial, 24-28 pt, gras
- **Corps** : Arial, 18-20 pt
- **Code** : Consolas, 14-16 pt

### Mise en page
- Maximum 3-4 points par slide
- Utiliser des listes à puces
- Espacer uniformément
- Aligner le texte à gauche

---

## 📊 Graphiques à Préparer

1. **Comparaison des modèles** : Bar chart avec Accuracy, F1, ROC-AUC
2. **Importance des features** : Bar chart horizontal
3. **Distribution de la cible** : Pie chart
4. **Courbe ROC** : Line chart avec 3 modèles
5. **Fairness** : Bar chart par groupe (genre, âge)
6. **SHAP summary** : Bar chart horizontal
7. **Matrice de confusion** : Heatmap

---

## ⏱️ Gestion du Temps

| Section | Durée | Cumulé |
|---------|---------|---------|
| Introduction | 1 min | 1 min |
| Contexte | 1 min | 2 min |
| Problématique | 1 min | 3 min |
| Dataset | 1 min | 4 min |
| Méthodologie | 1 min | 5 min |
| Modèles | 1 min | 6 min |
| Résultats | 1 min | 7 min |
| Features | 1 min | 8 min |
| SHAP | 1 min | 9 min |
| LIME | 1 min | 10 min |
| Contrefactuels | 1 min | 11 min |
| Fairness | 1 min | 12 min |
| Dashboard | 1 min | 13 min |
| Démonstration | 5 min | 18 min |
| Conclusion | 1 min | 19 min |
| Questions | 2 min | 21 min |

**Note** : Ajuster selon le temps disponible (15-20 min)

---

## 🎓 Checklist Avant la Soutenance

### Préparation
- [ ] Slides finales prêtes (PDF)
- [ ] Dashboard ouvert et testé
- [ ] Instances de démonstration sélectionnées
- [ ] Notes de présentation préparées
- [ ] Questions anticipées et réponses prêtes

### Technique
- [ ] Projecteur fonctionnel
- [ ] Son testé
- [ ] Timer visible
- [ ] Backup des slides sur USB
- [ ] Connexion internet stable

### Contenu
- [ ] Introduction accrocheuse
- [ ] Démonstration fluide
- [ ] Conclusion percutante
- [ ] Réponses aux questions prêtes

---

**Bonne chance pour la soutenance ! 🎓**