"""
Configuration centrale — Projet C.5 : Optimisation de Portefeuille Bayésien (Black-Litterman)
ECE Paris — ING4 Finance — Groupe 03
"""
import numpy as np

# ─── Univers d'actifs ────────────────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "BRK-B", "JNJ", "UNH", "AMZN", "XOM"]

ASSET_NAMES = {
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "GOOGL": "Alphabet",
    "NVDA":  "NVIDIA",
    "JPM":   "JPMorgan",
    "BRK-B": "Berkshire",
    "JNJ":   "J&J",
    "UNH":   "UnitedHealth",
    "AMZN":  "Amazon",
    "XOM":   "ExxonMobil",
}

SECTORS = {
    "AAPL":  "Technologie",
    "MSFT":  "Technologie",
    "GOOGL": "Technologie",
    "NVDA":  "Semi-conducteurs",
    "JPM":   "Finance",
    "BRK-B": "Diversifié",
    "JNJ":   "Santé",
    "UNH":   "Santé",
    "AMZN":  "Consommation",
    "XOM":   "Énergie",
}

# ─── Périodes ────────────────────────────────────────────────────────────────
TRAIN_START = "2018-01-01"
TRAIN_END   = "2022-12-31"
TEST_START  = "2023-01-01"
TEST_END    = "2024-06-30"

# ─── Paramètres Black-Litterman ──────────────────────────────────────────────
LAMBDA          = 2.5   # Aversion au risque implicite du marché
TAU             = 0.05  # Incertitude sur le prior (He & Litterman recommandent 0.025-0.05)
RISK_FREE_RATE  = 0.05  # Taux sans risque annuel (Fed Funds ~2023)

# ─── Capitalisations boursières relatives (Mds USD, approximatif 2022) ───────
MARKET_CAPS = {
    "AAPL":  2500, "MSFT": 2200, "GOOGL": 1400, "NVDA": 1200,
    "JPM":    450, "BRK-B": 700, "JNJ":    400, "UNH":   450,
    "AMZN": 1100, "XOM":    440,
}

# ─── Views de l'investisseur (méthode Idzorek) ───────────────────────────────
# Chaque view est un dict avec :
#   name       : description lisible
#   P          : vecteur n-dim (somme = 0 pour relative, somme > 0 pour absolue)
#   Q          : rendement attendu annualisé (float)
#   confidence : niveau de confiance [0, 1] (Idzorek 2005)
#
# View 1 (relative) : MSFT surperformera GOOGL de +4 %/an
# View 2 (absolue)  : NVDA rapportera 28 %/an
# View 3 (relative) : Tech (AAPL+MSFT+NVDA) surperformera Énergie (XOM) de +5 %/an

def build_views(tickers):
    n = len(tickers)
    idx = {t: i for i, t in enumerate(tickers)}

    views = []

    # View 1 — MSFT vs GOOGL (relative)
    p1 = np.zeros(n)
    p1[idx["MSFT"]]  = +1.0
    p1[idx["GOOGL"]] = -1.0
    views.append({"name": "MSFT > GOOGL  (+4 %)", "P": p1, "Q": 0.04, "confidence": 0.55})

    # View 2 — NVDA absolue
    p2 = np.zeros(n)
    p2[idx["NVDA"]] = 1.0
    views.append({"name": "NVDA = 28 %/an",        "P": p2, "Q": 0.28, "confidence": 0.70})

    # View 3 — Tech vs Energie (relative)
    p3 = np.zeros(n)
    p3[idx["AAPL"]]  = +1/3
    p3[idx["MSFT"]]  = +1/3
    p3[idx["NVDA"]]  = +1/3
    p3[idx["XOM"]]   = -1.0
    views.append({"name": "Tech > Énergie (+5 %)", "P": p3, "Q": 0.05, "confidence": 0.45})

    return views

# ─── Paramètres Monte Carlo / optimisation ───────────────────────────────────
N_PORTFOLIOS  = 8_000   # Portefeuilles aléatoires pour la frontière MC
N_FRONTIER_PTS = 120    # Points de la frontière efficiente analytique
SECTOR_MAX_WEIGHTS = {
    "Technologie": 0.45,
    "Santé": 0.25,
}
BACKTEST_INITIAL_WEALTH = 100.0
SENSITIVITY_VIEW_INDEX = 1
SENSITIVITY_CONFIDENCE_GRID = np.linspace(0.05, 0.95, 19)

# ─── Paramètres visuels ──────────────────────────────────────────────────────
FIGURE_DPI  = 150
FIGURE_SIZE = (10, 6)

COLORS = {
    "markowitz": "#E74C3C",
    "min_vol":   "#95A5A6",
    "bl":        "#2980B9",
    "equal":     "#27AE60",
    "market":    "#F39C12",
    "prior":     "#9B59B6",
    "posterior": "#2980B9",
    "random":    "#BDC3C7",
}

RESULTS_DIR  = "results"
FIGURES_DIR  = "results/figures"
DATA_CACHE   = "results/prices.csv"
