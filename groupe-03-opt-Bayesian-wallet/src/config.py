"""
config.py — Configuration centrale du projet.
Modifie ce fichier pour changer les actifs, les dates et les views.
Tous les autres fichiers lisent depuis ici automatiquement.
"""

# --- Actifs analysés ---
TICKERS = ["AAPL", "MSFT", "GOOGL", "META", "AMZN"]

# --- Période d'analyse (laisser END à None pour utiliser aujourd'hui) ---
from datetime import date
START = f"{date.today().year - 3}-01-01"
END   = date.today().strftime("%Y-%m-%d")

# --- Paramètres du modèle ---
RISK_FREE_RATE = 0.02   # Taux sans risque (2%)
RISK_AVERSION  = 2.5    # Aversion au risque du marché
TAU            = 0.05   # Scaling du prior
MAX_WEIGHT     = 0.40   # Poids maximum par actif (40%)

# --- Views manuelles (laisser vide [] pour utiliser le momentum automatique) ---
# Type "absolute"  : opinion sur un seul actif
# Type "relative"  : comparaison entre deux actifs
VIEWS = [
    {"type": "absolute", "asset": "AAPL",  "return": -0.05},
    {"type": "absolute", "asset": "META",   "return":  0.08},
    {"type": "absolute", "asset": "GOOGL",  "return": -0.10},
    {"type": "absolute", "asset": "MSFT",   "return":  0.16},
    #{"type": "relative", "outperformer": "MSFT", "underperformer": "GOOGL", "return": 0.05},
]

# Confiance associée à chaque view (entre 0 et 1, une valeur par view)
CONFIDENCES = [0.8, 0.7, 0.6, 0.4]

# --- Backtesting ---
START_BACKTEST  = f"{date.today().year - 5}-01-01"
TRAIN_WINDOW    = 252   # Jours d'historique pour estimer le modèle (~1 an)
REBALANCE_FREQ  = 21    # Fréquence de rééquilibrage (~1 mois)

# --- Momentum (si VIEWS est vide) ---
MOMENTUM_LOOKBACK = 63  # Jours pour calculer le momentum (~3 mois)
