"""
scripts/run_convergence_analysis.py
=====================================
Analyse de convergence du PINN Black-Scholes :

  1. MAE vs N_collocation  (résolution spatiale)
  2. MAE vs N_epochs       (profondeur d'entraînement)
  3. MAE vs architecture   (largeur du réseau)

Pour chaque configuration, un modèle est entraîné from scratch sur
N_EPOCHS_SHORT epochs (suffisant pour comparer les asymptotes).
Les résultats sont tracés sur des graphiques log-log.

Sortie :
    outputs/figures/convergence_n_coll.png
    outputs/figures/convergence_epochs.png
    outputs/figures/convergence_arch.png
    outputs/results/convergence_results.csv

Exécution :
    python scripts/run_convergence_analysis.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from src.models.black_scholes_pinn import BlackScholesPINN
from src.analytics.black_scholes_formula import bs_call_price

# ─────────────────────────────────────────────────────────────────────────────
# Paramètres fixes
# ─────────────────────────────────────────────────────────────────────────────
K, T, r, sigma = 100.0, 1.0, 0.05, 0.20
N_BC        = 200
LR          = 1e-3
LAMBDA_PDE  = 1.0
LAMBDA_BC   = 10.0
LAMBDA_IC   = 10.0

# Grille de test (identique pour toutes les runs)
S_test   = np.linspace(10, 290, 100)
tau_test = np.linspace(0.05, T, 50)
SS, TT   = np.meshgrid(S_test, tau_test, indexing="ij")
V_true   = bs_call_price(SS, K, TT, r, sigma)

device = torch.device("cpu")
OUT_FIG = "outputs/figures"
OUT_RES = "outputs/results"
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_RES, exist_ok=True)


def compute_mae(model):
    """MAE of the model on the fixed test grid."""
    model.eval()
    S_flat   = torch.tensor(SS.ravel(), dtype=torch.float32).unsqueeze(1)
    tau_flat = torch.tensor(TT.ravel(), dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        V_pred = model(S_flat, tau_flat).numpy().flatten()
    return float(np.abs(V_pred - V_true.ravel()).mean())


def train_model(hidden_sizes, n_epochs, n_coll, verbose=False):
    """
    Train a BlackScholesPINN and return MAE on the test grid.
    Uses Adam only (no L-BFGS) for speed.
    """
    model = BlackScholesPINN(
        K=K, r=r, sigma=sigma, T=T,
        hidden_sizes=hidden_sizes,
        lambda_pde=LAMBDA_PDE, lambda_bc=LAMBDA_BC, lambda_ic=LAMBDA_IC,
    )
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=200, min_lr=1e-6
    )

    pbar = range(1, n_epochs + 1)
    if verbose:
        pbar = tqdm(pbar, desc=f"  N_coll={n_coll}", ncols=70)

    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        ld = model.compute_loss(n_coll=n_coll, n_bc=N_BC, device=device)
        ld["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(ld["total"].detach())

    return compute_mae(model), model


# ─────────────────────────────────────────────────────────────────────────────
# 1. Convergence vs N_collocation
# ─────────────────────────────────────────────────────────────────────────────
N_EPOCHS_FIXED  = 3_000
HIDDEN_FIXED    = [50, 50, 50, 50]
N_COLL_VALS     = [200, 500, 1000, 2000, 3500, 5000, 8000]

print("=== Analyse 1 : MAE vs N_collocation ===")
mae_coll, time_coll = [], []
for nc in N_COLL_VALS:
    t0 = time.time()
    mae, _ = train_model(HIDDEN_FIXED, N_EPOCHS_FIXED, nc, verbose=True)
    dt = time.time() - t0
    mae_coll.append(mae)
    time_coll.append(dt)
    print(f"  N_coll={nc:5d} -> MAE={mae:.4f}$  ({dt:.0f}s)")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Convergence vs N_epochs (learning curve)
# ─────────────────────────────────────────────────────────────────────────────
N_COLL_FIXED   = 3_000
EPOCH_SNAPSHOTS = [500, 1000, 2000, 3000, 5000, 7000, 10_000]

print("\n=== Analyse 2 : MAE vs N_epochs (learning curve) ===")
# Train a single model and record MAE at snapshots
model_lc = BlackScholesPINN(
    K=K, r=r, sigma=sigma, T=T, hidden_sizes=HIDDEN_FIXED,
    lambda_pde=LAMBDA_PDE, lambda_bc=LAMBDA_BC, lambda_ic=LAMBDA_IC,
)
optimizer_lc = optim.Adam(model_lc.parameters(), lr=LR)
scheduler_lc = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_lc, factor=0.5, patience=200, min_lr=1e-6
)

mae_epochs = []
snapshot_set = set(EPOCH_SNAPSHOTS)
total_epochs = max(EPOCH_SNAPSHOTS)

pbar = tqdm(range(1, total_epochs + 1), desc="Learning curve", ncols=70)
for epoch in pbar:
    model_lc.train()
    optimizer_lc.zero_grad()
    ld = model_lc.compute_loss(n_coll=N_COLL_FIXED, n_bc=N_BC, device=device)
    ld["total"].backward()
    torch.nn.utils.clip_grad_norm_(model_lc.parameters(), 1.0)
    optimizer_lc.step()
    scheduler_lc.step(ld["total"].detach())
    if epoch in snapshot_set:
        mae = compute_mae(model_lc)
        mae_epochs.append(mae)
        pbar.set_postfix(mae=f"{mae:.4f}")

print(f"  MAE apres {total_epochs} epochs : {mae_epochs[-1]:.4f}$")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Convergence vs architecture réseau
# ─────────────────────────────────────────────────────────────────────────────
N_EPOCHS_ARCH = 3_000
ARCH_CONFIGS  = [
    [20, 20],
    [20, 20, 20],
    [50, 50],
    [50, 50, 50],
    [50, 50, 50, 50],
    [100, 100, 100],
    [100, 100, 100, 100],
]

print("\n=== Analyse 3 : MAE vs Architecture ===")
mae_arch, n_params_arch, labels_arch = [], [], []
for arch in ARCH_CONFIGS:
    lbl = "x".join(str(h) for h in arch)
    n_p = sum(p.numel() for p in BlackScholesPINN(hidden_sizes=arch).parameters())
    t0 = time.time()
    mae, _ = train_model(arch, N_EPOCHS_ARCH, N_COLL_FIXED, verbose=False)
    dt = time.time() - t0
    mae_arch.append(mae)
    n_params_arch.append(n_p)
    labels_arch.append(lbl)
    print(f"  [{lbl}] {n_p:6,} params -> MAE={mae:.4f}$  ({dt:.0f}s)")

# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

# Figure 1 : MAE vs N_coll
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.loglog(N_COLL_VALS, mae_coll, "b-o", ms=6, lw=2)
# Fit a power law for reference
valid = np.array(mae_coll) > 1e-6
if valid.sum() >= 2:
    c = np.polyfit(np.log(np.array(N_COLL_VALS)[valid]),
                   np.log(np.array(mae_coll)[valid]), 1)
    fit_y = np.exp(c[1]) * np.array(N_COLL_VALS) ** c[0]
    ax.loglog(N_COLL_VALS, fit_y, "r--", lw=1.5,
              label=f"Fit: MAE ~ N^{{{c[0]:.2f}}}")
ax.set(xlabel="N_collocation", ylabel="MAE ($)",
       title="Convergence vs N_collocation\n(3000 epochs, reseau 50x4)")
ax.legend(); ax.grid(True, which="both", alpha=0.3)
ax.set_xticks(N_COLL_VALS)
ax.set_xticklabels([str(n) for n in N_COLL_VALS], rotation=30, fontsize=8)

# Figure 2 : Learning curve
ax = axes[1]
ax.semilogy(EPOCH_SNAPSHOTS, mae_epochs, "g-s", ms=6, lw=2)
ax.set(xlabel="Epochs", ylabel="MAE ($)",
       title="Courbe d'apprentissage\n(N_coll=3000, reseau 50x4)")
ax.grid(True, which="both", alpha=0.3)

# Figure 3 : MAE vs nb paramètres
ax = axes[2]
ax.loglog(n_params_arch, mae_arch, "m-^", ms=6, lw=2)
for i, lbl in enumerate(labels_arch):
    ax.annotate(lbl, (n_params_arch[i], mae_arch[i]),
                textcoords="offset points", xytext=(4, 4), fontsize=7)
ax.set(xlabel="Nombre de parametres", ylabel="MAE ($)",
       title="MAE vs Architecture\n(3000 epochs, N_coll=3000)")
ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG, "convergence_analysis.png"), dpi=150)
plt.close()
print("\nFigure -> outputs/figures/convergence_analysis.png")

# Figures individuelles (pour lisibilité dans le rapport)
for data, x_vals, xlabel, title, fname, xlog in [
    (mae_coll, N_COLL_VALS, "N_collocation", "MAE vs N_coll", "convergence_n_coll.png", True),
    (mae_epochs, EPOCH_SNAPSHOTS, "Epochs",  "Courbe d'apprentissage", "convergence_epochs.png", False),
    (mae_arch,  n_params_arch, "Nb parametres", "MAE vs Architecture", "convergence_arch.png", True),
]:
    fig, ax = plt.subplots(figsize=(7, 4))
    if xlog:
        ax.loglog(x_vals, data, "b-o", ms=6, lw=2)
    else:
        ax.semilogy(x_vals, data, "b-o", ms=6, lw=2)
    ax.set(xlabel=xlabel, ylabel="MAE ($)", title=title)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, fname), dpi=150)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Sauvegarde CSV
# ─────────────────────────────────────────────────────────────────────────────
rows = []
for nc, mae, dt in zip(N_COLL_VALS, mae_coll, time_coll):
    rows.append({"analysis": "n_coll", "x_label": "N_coll",
                 "x_val": nc, "MAE": mae, "time_s": dt})
for ep, mae in zip(EPOCH_SNAPSHOTS, mae_epochs):
    rows.append({"analysis": "epochs", "x_label": "epochs",
                 "x_val": ep, "MAE": mae, "time_s": np.nan})
for lbl, np_, mae in zip(labels_arch, n_params_arch, mae_arch):
    rows.append({"analysis": "arch", "x_label": lbl,
                 "x_val": np_, "MAE": mae, "time_s": np.nan})

pd.DataFrame(rows).to_csv(
    os.path.join(OUT_RES, "convergence_results.csv"), index=False
)

print("\n========== RESUME CONVERGENCE ==========")
print(f"  N_coll analyse : min MAE = {min(mae_coll):.4f}$ @ N={N_COLL_VALS[np.argmin(mae_coll)]}")
print(f"  Learning curve : MAE a 10k epochs = {mae_epochs[-1]:.4f}$")
print(f"  Architecture   : min MAE = {min(mae_arch):.4f}$ avec [{labels_arch[np.argmin(mae_arch)]}]")
print("=========================================\n")
