# Documentation Technique — Optimisation de Portefeuille Bayésien

**Groupe 03 | ECE Paris 2026**

---

## 1. Architecture du Code

Le projet est structuré en 7 modules indépendants dans le dossier `src/`. Chaque module a une responsabilité unique et bien définie, ce qui facilite la maintenance et les tests. Tous les paramètres sont centralisés dans `config.py` — c'est le seul fichier à modifier pour changer les actifs, les dates, les views ou les contraintes.

```
config.py → data.py → stats.py → markowitz.py → black_litterman.py → ml_views.py / backtest.py
```

---

## 2. Module `data.py` — Récupération des données

Ce module est le point d'entrée de tout le pipeline. Il s'occupe de télécharger les prix historiques des actifs financiers depuis Yahoo Finance via la bibliothèque `yfinance`, puis de les transformer en rendements utilisables par le modèle.

### Téléchargement des prix

La fonction `download_prices(tickers, start, end)` récupère les **prix de clôture ajustés** pour une liste d'actifs sur une période donnée. Les prix ajustés tiennent compte des dividendes et des splits d'actions, ce qui donne une image fidèle de la performance réelle.

### Calcul des rendements logarithmiques

La fonction `compute_returns(prices)` transforme les prix en rendements journaliers. On utilise les **rendements logarithmiques** plutôt que les rendements simples pour deux raisons :

1. Ils sont **additifs dans le temps** : le rendement sur 2 jours est la somme des rendements journaliers
2. Ils sont **plus stables numériquement** pour les calculs matriciels

La formule est :

```
r_t = log(P_t / P_{t-1})
```

Pour annualiser, on multiplie la moyenne par 252 (nombre de jours de bourse dans une année) et la covariance par 252 également.

---

## 3. Module `stats.py` — Statistiques de base

Ce module calcule les grandeurs statistiques fondamentales nécessaires à toute optimisation de portefeuille : les rendements moyens attendus, la matrice de covariance, et les indicateurs de performance.

### Matrice de covariance

La matrice de covariance `Σ` est une matrice carrée de dimension n×n (où n est le nombre d'actifs). L'élément `Σ_ij` mesure à quel point les actifs i et j évoluent ensemble. Sur la diagonale, `Σ_ii` est la variance de l'actif i (son risque propre). En dehors de la diagonale, une valeur positive signifie que les deux actifs montent et baissent en même temps, une valeur négative signifie qu'ils évoluent en sens inverse (diversification).

### Ratio de Sharpe

Le ratio de Sharpe est l'indicateur central pour comparer deux portefeuilles. Il mesure le rendement **au-dessus du taux sans risque** (ce qu'on aurait gagné sans risquer son argent) par unité de risque pris :

```
Sharpe = (Rendement du portefeuille - Taux sans risque) / Volatilité
```

Un Sharpe de 0.5 signifie que pour chaque 1% de risque pris, le portefeuille génère 0.5% de rendement supplémentaire par rapport au placement sans risque. En pratique, un Sharpe supérieur à 1 est considéré comme bon, supérieur à 2 comme excellent.

---

## 4. Module `markowitz.py` — Optimisation Classique

Ce module implémente la **théorie moderne du portefeuille de Markowitz (1952)**, qui sert de référence de comparaison pour évaluer l'apport du modèle Black-Litterman.

### Principe de Markowitz

L'idée centrale est que le risque d'un portefeuille ne dépend pas seulement du risque de chaque actif pris individuellement, mais aussi des **corrélations entre les actifs**. En mélangeant des actifs peu corrélés, on peut réduire le risque global sans sacrifier le rendement — c'est le principe de la diversification.

### Optimisation

La fonction `markowitz_weights(mu, cov)` résout le problème d'optimisation suivant : trouver les poids `w` de chaque actif qui maximisent le ratio de Sharpe, sous contrainte que les poids somment à 1 et restent positifs (pas de vente à découvert) :

```
Maximiser  : (w'μ - rf) / sqrt(w'Σw)
Sous        : Σwi = 1
              wi ≥ 0  pour tout i
```

Ce problème est résolu numériquement par la méthode SLSQP de `scipy.optimize.minimize`.

### Frontière Efficiente

La frontière efficiente est la courbe qui relie tous les portefeuilles optimaux. Pour chaque niveau de rendement cible, on calcule le portefeuille de variance minimale. L'ensemble de ces portefeuilles forme la frontière — tout portefeuille en dehors de cette courbe est sous-optimal car on peut obtenir le même rendement avec moins de risque.

---

## 5. Module `black_litterman.py` — Modèle Bayésien

C'est le cœur du projet. Le modèle Black-Litterman résout la principale faiblesse de Markowitz : sa sensibilité excessive aux estimations des rendements. En intégrant des opinions (views) via un cadre bayésien, il produit des allocations plus stables et plus intuitives.

### Étape 1 — Prior : Rendements d'équilibre du marché (Π)

Le prior représente ce que le marché "croit" en l'absence d'opinion particulière. On l'obtient en inversant le CAPM : plutôt que de calculer le rendement attendu à partir des bêtas, on déduit les rendements implicites à partir des **poids de capitalisation boursière** du marché.

L'intuition est la suivante : si les marchés sont en équilibre, les prix actuels reflètent les anticipations de tous les investisseurs. Les capitalisations boursières donnent donc les "poids naturels" d'un portefeuille de marché.

La formule est :

```
Π = δ × Σ × w_marché
```

Où :
- `δ` est le coefficient d'aversion au risque (typiquement 2.5 pour un investisseur institutionnel)
- `Σ` est la matrice de covariance annualisée
- `w_marché` sont les poids du portefeuille de marché (capitalisations boursières normalisées)

### Étape 2 — Views : Opinions de l'Investisseur

Les views représentent les opinions personnelles de l'investisseur sur les rendements futurs. On distingue deux types :

**View absolue** : l'investisseur a une opinion sur un actif seul.
Exemple : "Je pense qu'Apple va faire +10% l'année prochaine."
Dans la matrice P, la ligne correspondante a un 1 sur la colonne d'Apple et des 0 partout ailleurs.

**View relative** : l'investisseur compare deux actifs.
Exemple : "Je pense que Microsoft surperformera Google de 5%."
Dans la matrice P, la ligne correspondante a un +1 sur Microsoft et un -1 sur Google.

Le vecteur Q contient les rendements attendus associés à chaque view. L'ensemble des views est donc encodé par la paire (P, Q) où P est une matrice k×n (k views, n actifs) et Q est un vecteur de taille k.

### Étape 3 — Omega : Incertitude sur les Views

Omega (Ω) est une matrice diagonale qui encode le **niveau d'incertitude** de l'investisseur sur chacune de ses views. Plus un investisseur est confiant dans une opinion, plus Omega est petit, et plus la view aura d'impact sur le résultat final.

Avec un niveau de confiance `c_i` entre 0 et 1 pour chaque view i :

```
Ω_ii = ((1 - c_i) / c_i) × τ × (P Σ P')_ii
```

Où `τ` (tau) est un scalaire de scaling (typiquement 0.05) qui règle l'importance globale des views par rapport au prior. Si `c_i = 0.8` (80% de confiance), Omega sera petit et la view dominera. Si `c_i = 0.2` (20% de confiance), Omega sera grand et le prior dominera.

### Étape 4 — Posterior : Formule de Black-Litterman

C'est l'étape centrale qui applique le théorème de Bayes pour combiner le prior du marché et les views de l'investisseur. La formule est :

```
M     = (τΣ)⁻¹ + P'Ω⁻¹P
μ_BL  = M⁻¹ × [(τΣ)⁻¹ × Π + P'Ω⁻¹ × Q]
Σ_BL  = Σ + M⁻¹
```

Le résultat `μ_BL` est une **moyenne pondérée** entre le prior Π et les views Q. La pondération est déterminée par la précision relative de chaque source d'information :
- Si l'investisseur est très confiant (Omega petit), `μ_BL` se rapproche de Q
- Si l'investisseur est peu confiant (Omega grand), `μ_BL` reste proche de Π

La covariance postérieure `Σ_BL` est toujours plus grande que Σ, car l'ajout d'opinions introduit une incertitude supplémentaire. C'est une propriété fondamentale de l'inférence bayésienne.

---

## 6. Module `ml_views.py` — Views par Machine Learning

Ce module atteint le niveau Excellent du projet en générant automatiquement les views à l'aide d'un algorithme de Machine Learning basé sur le **momentum de prix**.

### Principe du Momentum

Le momentum est une anomalie de marché bien documentée depuis les travaux de Jegadeesh & Titman (1993) : les actifs qui ont bien performé sur les 3 à 12 derniers mois tendent statistiquement à continuer à surperformer à court terme. C'est le principe du "trend following" utilisé par de nombreux fonds quantitatifs.

### Implémentation

Pour chaque actif, on calcule le rendement brut sur les 63 derniers jours de bourse (~3 mois) :

```
momentum_i = (P_aujourd'hui / P_{il y a 63 jours}) - 1
```

Si ce rendement est positif, on génère une **view positive** automatique sur cet actif. Si négatif, une **view négative**. La magnitude de la view est proportionnelle au momentum observé, plafonnée à `view_scale` (10% par défaut) :

```
view_i = view_scale × sign(momentum_i) × min(|momentum_i|, 1.0)
```

L'avantage de cette approche est qu'elle élimine le biais subjectif de l'investisseur : les opinions sont entièrement déterminées par les données, ce qui est reproductible et testable.

---

## 7. Module `backtest.py` — Validation et Robustesse

Ce module valide la stratégie sur des données historiques et mesure sa robustesse aux erreurs d'estimation.

### Backtesting par Fenêtre Glissante

Le backtesting simule le comportement qu'aurait eu la stratégie dans le passé, sans regarder le futur. On utilise une fenêtre glissante (rolling window) :

1. On sélectionne une fenêtre d'entraînement de 252 jours (1 an)
2. On estime le modèle BL sur cette fenêtre et on calcule les poids optimaux
3. On applique ces poids sur les 21 jours suivants (1 mois) et on mesure le rendement réel
4. On décale la fenêtre d'un mois et on recommence

Cette méthode est plus réaliste qu'un simple test statique car elle simule les rééquilibrages périodiques qu'un gestionnaire ferait en pratique.

### Analyse de Sensibilité

L'analyse de sensibilité répond à la question : "Que se passe-t-il si mes opinions sont fausses ?" On multiplie le vecteur Q par des scalaires allant de 0.25 à 2.0 et on observe l'impact sur les poids optimaux et le ratio de Sharpe. Un modèle robuste ne doit pas être trop sensible à de petites perturbations des views.

---

## 8. Tests Unitaires

Les tests unitaires vérifient automatiquement que chaque fonction respecte ses propriétés mathématiques fondamentales. Ils permettent de s'assurer que le code est correct, et de détecter immédiatement toute régression si le code est modifié.

**Exemples de propriétés testées :**
- La matrice de covariance est symétrique et définie positive (propriété mathématique obligatoire)
- Les poids optimaux somment à 1 et sont tous positifs (pas de vente à découvert)
- Le posterior BL tire bien vers les views quand la confiance est élevée
- Omega diminue quand la confiance augmente

**Lancer les tests :**
```bash
py -m pytest tests/ -v
```

Résultat attendu : **30 tests passent** en moins de 2 secondes.

---

## 9. Choix Techniques et Justifications

| Choix | Valeur | Justification |
|-------|--------|---------------|
| Rendements logarithmiques | — | Additifs dans le temps, stables numériquement |
| Méthode d'optimisation | SLSQP | Robuste pour les problèmes quadratiques sous contraintes |
| Tau (τ) | 0.05 | Valeur standard dans la littérature Black-Litterman |
| Delta (δ) | 2.5 | Aversion au risque typique d'un investisseur institutionnel |
| Max weight | 40% | Limite par actif pour éviter la concentration totale sur un seul actif |
| Fenêtre momentum | 63 jours | ~3 mois, fenêtre standard dans la littérature momentum |
| Fenêtre d'entraînement | 252 jours | 1 an, suffisant pour estimer la covariance stablement |
| Fréquence de rééquilibrage | 21 jours | ~1 mois, bon compromis entre réactivité et coûts |
| Taux sans risque | 2% | Approximation du taux des obligations d'État en 2024 |
