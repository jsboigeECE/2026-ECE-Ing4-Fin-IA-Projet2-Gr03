"""
scripts/train_american_put.py
==============================
Entraîne le PINN pour la put américaine et sauvegarde dans outputs/ :

    outputs/figures/am_loss_curve.png
    outputs/figures/am_price_surface.png
    outputs/figures/am_vs_european.png
    outputs/figures/am_free_boundary.png
    outputs/results/american_metrics.csv
    outputs/checkpoints/american_put_final.pt

Exécution :
    python scripts/train_american_put.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.models.american_put_pinn import AmericanPutPINN
from src.models.black_scholes_pinn import BlackScholesPINN
from src.analytics.black_scholes_formula import bs_put_price
from src.training.trainer import PINNTrainer

# ─────────────────────────────────────────────────────────────────────────────
# Paramètres
# ─────────────────────────────────────────────────────────────────────────────
K     = 100.0
T     = 1.0
r     = 0.05
sigma = 0.20
S_max = 300.0

HIDDEN_SIZES  = [64, 64, 64, 64]
N_EPOCHS      = 10_000
N_COLL        = 5_000
N_BC          = 500
LR            = 1e-3
LAMBDA_PDE    = 1.0
LAMBDA_BC     = 10.0
LAMBDA_IC     = 10.0
LAMBDA_PEN    = 150.0
LBFGS_EPOCHS  = 500
SAVE_EVERY    = 1_000
PATIENCE      = 0      # désactivé : la put américaine a besoin des 10k epochs

OUT_FIG  = "outputs/figures"
OUT_CKPT = "outputs/checkpoints"
OUT_RES  = "outputs/results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_CKPT, exist_ok=True)
os.makedirs(OUT_RES, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Modèle put américaine
# ─────────────────────────────────────────────────────────────────────────────
model = AmericanPutPINN(
    K=K, r=r, sigma=sigma, T=T, S_max=S_max,
    hidden_sizes=HIDDEN_SIZES,
    lambda_pde=LAMBDA_PDE, lambda_bc=LAMBDA_BC,
    lambda_ic=LAMBDA_IC, lambda_pen=LAMBDA_PEN,
)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parametres reseau : {n_params:,}")

trainer = PINNTrainer(
    model=model,
    n_epochs=N_EPOCHS,
    lr=LR,
    n_coll=N_COLL,
    n_bc=N_BC,
    lbfgs_epochs=LBFGS_EPOCHS,
    save_every=SAVE_EVERY,
    checkpoint_dir=OUT_CKPT,
    patience=PATIENCE,
    device=device,
)

t_start = time.time()
history = trainer.train()
train_time = time.time() - t_start
print(f"Duree entraînement : {train_time:.1f} s")

# Checkpoint
ckpt_path = os.path.join(OUT_CKPT, "american_put_final.pt")
torch.save(model.state_dict(), ckpt_path)
print(f"Modele sauvegarde -> {ckpt_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 : Courbes de loss (avec la composante penalty)
# ─────────────────────────────────────────────────────────────────────────────
epochs_arr = np.arange(1, len(history.total) + 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
ax = axes[0]
ax.semilogy(epochs_arr, history.total,   label="Total")
ax.semilogy(epochs_arr, history.pde,     label="L_pde",     ls="--")
ax.semilogy(epochs_arr, history.bc,      label="L_bc",      ls=":")
ax.semilogy(epochs_arr, history.ic,      label="L_ic",      ls="-.")
if hasattr(history, "penalty") and history.penalty:
    ax.semilogy(epochs_arr, history.penalty, label="L_penalty", ls=(0, (3, 1, 1, 1)))
ax.set(xlabel="Epoch", ylabel="Loss (log)", title="Decomposition de la loss - Put Americaine")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
ax.semilogy(epochs_arr, history.total)
ax.set(xlabel="Epoch", ylabel="Loss totale (log)", title="Convergence globale")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG, "am_loss_curve.png"), dpi=150)
plt.close()
print("Figure sauvegardee -> outputs/figures/am_loss_curve.png")

# ─────────────────────────────────────────────────────────────────────────────
# Grilles d'évaluation
# ─────────────────────────────────────────────────────────────────────────────
model.eval()
S_vals   = np.linspace(10, 250, 120)
tau_vals = np.linspace(0.01, T, 80)
SS, TT = np.meshgrid(S_vals, tau_vals, indexing="ij")  # shape (120, 80)

# PINN American put surface
S_flat   = torch.tensor(SS.ravel(), dtype=torch.float32, device=device).unsqueeze(1)
tau_flat = torch.tensor(TT.ravel(), dtype=torch.float32, device=device).unsqueeze(1)
with torch.no_grad():
    V_am = model(S_flat, tau_flat).cpu().numpy().reshape(SS.shape)

# European put (analytical) for comparison
V_eu = bs_put_price(SS, K, TT, r, sigma)

# Intrinsic value surface
intrinsic_grid = np.maximum(K - SS, 0.0)

# Early exercise premium
premium = np.maximum(V_am - V_eu, 0.0)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 : Surface de prix 3D (American vs European put)
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 5))

ax1 = fig.add_subplot(131, projection="3d")
ax1.plot_surface(SS, TT, V_am, cmap="plasma", alpha=0.85)
ax1.set(xlabel="S", ylabel="tau", zlabel="V", title="Put Americaine (PINN)")

ax2 = fig.add_subplot(132, projection="3d")
ax2.plot_surface(SS, TT, V_eu, cmap="viridis", alpha=0.85)
ax2.set(xlabel="S", ylabel="tau", zlabel="V", title="Put Europeenne (analytique)")

ax3 = fig.add_subplot(133, projection="3d")
ax3.plot_surface(SS, TT, premium, cmap="Reds", alpha=0.85)
ax3.set(xlabel="S", ylabel="tau", zlabel="Premium", title="Prime d'exercice anticipe")

plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG, "am_price_surface.png"), dpi=150)
plt.close()
print("Figure sauvegardee -> outputs/figures/am_price_surface.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 : Coupes V(S) — American vs European vs Intrinsic
# ─────────────────────────────────────────────────────────────────────────────
S_line = np.linspace(10, 200, 400)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ax = axes[0]
for tau_val in [T, T / 2, T / 4]:
    V_eu_line  = bs_put_price(S_line, K, tau_val, r, sigma)
    S_t   = torch.tensor(S_line, dtype=torch.float32, device=device).unsqueeze(1)
    tau_t = torch.full_like(S_t, tau_val)
    with torch.no_grad():
        V_am_line = model(S_t, tau_t).cpu().numpy().flatten()
    ax.plot(S_line, V_am_line, lw=2,     label=f"Americaine tau={tau_val:.2f}")
    ax.plot(S_line, V_eu_line, lw=1.5, ls="--", label=f"Europeenne tau={tau_val:.2f}", alpha=0.7)

intrinsic_line = np.maximum(K - S_line, 0.0)
ax.plot(S_line, intrinsic_line, "k:", lw=1.5, label="Valeur intrinseque")
ax.axvline(K, ls=":", color="gray", alpha=0.5)
ax.set(xlabel="Spot S", ylabel="V", title="Put Americaine vs Europeenne")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# Early-exercise premium at tau = T
ax = axes[1]
V_eu_T  = bs_put_price(S_line, K, T, r, sigma)
S_t = torch.tensor(S_line, dtype=torch.float32, device=device).unsqueeze(1)
tau_T = torch.full_like(S_t, T)
with torch.no_grad():
    V_am_T = model(S_t, tau_T).cpu().numpy().flatten()
ax.fill_between(S_line, np.maximum(V_am_T - V_eu_T, 0),
                alpha=0.4, color="orange", label="Prime exercice anticipe")
ax.plot(S_line, np.maximum(V_am_T - V_eu_T, 0), color="darkorange", lw=2)
ax.axvline(K, ls=":", color="gray", alpha=0.5)
ax.set(xlabel="Spot S", ylabel="V_am - V_eu",
       title="Prime d'exercice anticipe (tau=T)")
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG, "am_vs_european.png"), dpi=150)
plt.close()
print("Figure sauvegardee -> outputs/figures/am_vs_european.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 : Frontière libre (free boundary S*(tau))
# ─────────────────────────────────────────────────────────────────────────────
tau_fb = np.linspace(0.02, T, 60)
S_star = model.exercise_boundary(tau_fb, device=device)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ax = axes[0]
ax.plot(tau_fb, S_star, "b-o", ms=3, lw=2, label="S*(tau) - PINN")
ax.axhline(K, ls="--", color="gray", label=f"K = {K}")
ax.set(xlabel="Temps avant maturite (tau)", ylabel="Spot critique S*",
       title="Frontiere libre - exercice anticipe")
ax.legend(); ax.grid(alpha=0.3)
ax.invert_xaxis()  # tau=T a gauche (loin de maturite), tau=0 a droite

# Carte 2D avec frontière libre
ax = axes[1]
im = ax.contourf(SS, TT, V_am, levels=30, cmap="viridis")
plt.colorbar(im, ax=ax, label="V")
# Superposer la frontière libre
valid = ~np.isnan(S_star)
ax.plot(S_star[valid], tau_fb[valid], "r-", lw=2.5, label="Frontiere libre")
ax.set(xlabel="Spot S", ylabel="tau",
       title="Surface de prix + frontiere libre")
ax.legend(); ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG, "am_free_boundary.png"), dpi=150)
plt.close()
print("Figure sauvegardee -> outputs/figures/am_free_boundary.png")

# ─────────────────────────────────────────────────────────────────────────────
# Métriques
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd

# Verifier la contrainte V >= intrinsic
violation_grid = np.maximum(intrinsic_grid - V_am, 0.0)
max_violation  = violation_grid.max()
mean_violation = violation_grid.mean()

# Comparer avec put europeenne (lower bound)
mae_vs_eu  = np.abs(V_am - V_eu).mean()
premium_mean = premium.mean()

metrics = {
    "MAE_vs_european":     round(float(mae_vs_eu), 4),
    "early_exercise_premium_mean": round(float(premium_mean), 4),
    "max_early_exercise_constraint_violation": round(float(max_violation), 6),
    "mean_early_exercise_constraint_violation": round(float(mean_violation), 6),
    "final_loss":          round(float(history.total[-1]), 6),
    "train_time_s":        round(train_time, 1),
    "epochs":              len(history.total),
}

pd.DataFrame([metrics]).to_csv(
    os.path.join(OUT_RES, "american_metrics.csv"), index=False
)
print(f"Metriques -> {os.path.join(OUT_RES, 'american_metrics.csv')}")

print("\n========== RESULTATS ==========")
print(f"  Duree entraînement    : {train_time:.1f} s")
print(f"  Loss finale           : {history.total[-1]:.4e}")
print(f"  MAE vs put europeenne : {mae_vs_eu:.4f} $")
print(f"  Prime exercice anticipe (moy.) : {premium_mean:.4f} $")
print(f"  Contrainte V>=intrinsic - violation max  : {max_violation:.6f}")
print(f"  Contrainte V>=intrinsic - violation moy  : {mean_violation:.6f}")
ok = max_violation < 0.5
print(f"\n  {'OK' if ok else 'ATTENTION'} Contrainte early exercise : {'respectee' if ok else 'violation > 0.5$'}")
print("================================\n")
