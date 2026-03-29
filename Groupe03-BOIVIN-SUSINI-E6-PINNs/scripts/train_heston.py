"""
scripts/train_heston.py
=======================
Entraîne le PINN pour le modèle de Heston (volatilité stochastique).
Produit dans outputs/ :

    figures/heston_loss_curve.png
    figures/heston_vs_bs.png            Heston vs BS : effet du vol stochastique
    figures/heston_surface_SV.png       Surface V(S,v) à maturité T
    figures/heston_implied_vol_smile.png Smile de volatilité implicite
    results/heston_metrics.csv
    checkpoints/heston_final.pt

Exécution :
    python scripts/train_heston.py
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
import pandas as pd

from src.models.heston_pinn import HestonPINN
from src.analytics.heston_formula import heston_call_price, heston_mc_price, implied_vol
from src.analytics.black_scholes_formula import bs_call_price
from src.training.trainer import PINNTrainer

# ─────────────────────────────────────────────────────────────────────────────
# Paramètres Heston
# ─────────────────────────────────────────────────────────────────────────────
K       = 100.0
T       = 1.0
r       = 0.05
kappa   = 2.0       # mean-reversion speed
theta   = 0.04      # long-run variance (equiv. 20% vol)
sigma_v = 0.3       # vol-of-vol
rho     = -0.7      # correlation spot-vol (negative skew)
v0      = 0.04      # initial variance (same as theta for ATM)

S_max   = 300.0
v_max   = 1.0
HIDDEN  = [64, 64, 64, 64, 64]

N_EPOCHS     = 5_000
N_COLL       = 5_000
N_BC         = 500
LR           = 1e-3
LBFGS_EPOCHS = 300
SAVE_EVERY   = 1_000

OUT_FIG  = "outputs/figures"
OUT_CKPT = "outputs/checkpoints"
OUT_RES  = "outputs/results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_CKPT, exist_ok=True)
os.makedirs(OUT_RES, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Modèle
# ─────────────────────────────────────────────────────────────────────────────
model = HestonPINN(
    K=K, r=r, kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho,
    v0=v0, T=T, S_max=S_max, v_max=v_max,
    hidden=HIDDEN, lam_pde=1.0, lam_bc=10.0, lam_ic=10.0,
)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parametres reseau : {n_params:,}")

trainer = PINNTrainer(
    model=model, n_epochs=N_EPOCHS, lr=LR,
    n_coll=N_COLL, n_bc=N_BC,
    lbfgs_epochs=LBFGS_EPOCHS,
    save_every=SAVE_EVERY, checkpoint_dir=OUT_CKPT,
    patience=0, device=device,
)

t_start = time.time()
history = trainer.train()
train_time = time.time() - t_start
print(f"Duree : {train_time:.1f} s")

torch.save(model.state_dict(), os.path.join(OUT_CKPT, "heston_final.pt"))

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 : Courbes de loss
# ─────────────────────────────────────────────────────────────────────────────
ep = np.arange(1, len(history.total) + 1)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].semilogy(ep, history.total, label="Total")
axes[0].semilogy(ep, history.pde,   label="L_pde", ls="--")
axes[0].semilogy(ep, history.bc,    label="L_bc",  ls=":")
axes[0].semilogy(ep, history.ic,    label="L_ic",  ls="-.")
axes[0].set(xlabel="Epoch", ylabel="Loss (log)",
            title="Convergence - Heston PINN")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].semilogy(ep, history.total)
axes[1].set(xlabel="Epoch", ylabel="Loss totale", title="Convergence globale")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG, "heston_loss_curve.png"), dpi=150)
plt.close()
print("Figure -> outputs/figures/heston_loss_curve.png")

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark Monte Carlo (quelques points spot)
# ─────────────────────────────────────────────────────────────────────────────
print("\nCalcul des prix de reference (Monte Carlo Heston)...")
S_bench = np.array([70., 80., 90., 100., 110., 120., 130.])
mc_prices, mc_stderr = zip(*[
    heston_mc_price(s, K, T, r, kappa, theta, sigma_v, rho, v0, n_paths=200_000, seed=42)
    for s in S_bench
])
mc_prices = np.array(mc_prices)
mc_stderr = np.array(mc_stderr)

# PINN prices at same spots (v = v0)
model.eval()
S_t   = torch.tensor(S_bench, dtype=torch.float32, device=device).unsqueeze(1)
v_t   = torch.full_like(S_t, v0)
tau_t = torch.full_like(S_t, T)
with torch.no_grad():
    pinn_bench = model(S_t, v_t, tau_t).cpu().numpy().flatten()

# BS prices for comparison
bs_bench = bs_call_price(S_bench, K, T, r, np.sqrt(theta))

print("\n  S    | MC Heston | PINN Heston | BS (sigma=sqrt(theta))")
print("  " + "-" * 55)
for i, s in enumerate(S_bench):
    print(f"  {s:5.0f} | {mc_prices[i]:9.3f} | {pinn_bench[i]:11.3f} | {bs_bench[i]:.3f}")

mae_vs_mc = np.abs(pinn_bench - mc_prices).mean()
print(f"\n  MAE PINN vs MC : {mae_vs_mc:.4f} $")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 : Heston vs BS — effet de la corrélation et vol stochastique
# ─────────────────────────────────────────────────────────────────────────────
S_line = np.linspace(60, 160, 200)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ax = axes[0]
for v_val, lbl in [(0.01, "v=0.01 (10% vol)"),
                    (0.04, "v=0.04 (20% vol)"),
                    (0.09, "v=0.09 (30% vol)"),
                    (0.16, "v=0.16 (40% vol)")]:
    S_t = torch.tensor(S_line, dtype=torch.float32, device=device).unsqueeze(1)
    v_t = torch.full_like(S_t, v_val)
    tau_t = torch.full_like(S_t, T)
    with torch.no_grad():
        V_pinn = model(S_t, v_t, tau_t).cpu().numpy().flatten()
    sigma_equiv = np.sqrt(v_val)
    V_bs = bs_call_price(S_line, K, T, r, sigma_equiv)
    ax.plot(S_line, V_pinn, lw=2, label=f"Heston {lbl}")
    ax.plot(S_line, V_bs, lw=1.2, ls="--", alpha=0.6)

ax.axvline(K, ls=":", color="gray", alpha=0.5)
ax.set(xlabel="Spot S", ylabel="V", title="Heston PINN (trait) vs BS equivalent (tiret)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Différence Heston - BS à v=v0
ax = axes[1]
S_t = torch.tensor(S_line, dtype=torch.float32, device=device).unsqueeze(1)
v_t = torch.full_like(S_t, v0)
tau_t = torch.full_like(S_t, T)
with torch.no_grad():
    V_heston = model(S_t, v_t, tau_t).cpu().numpy().flatten()
V_bs_ref = bs_call_price(S_line, K, T, r, np.sqrt(v0))
diff = V_heston - V_bs_ref
ax.plot(S_line, diff, color="darkred", lw=2)
ax.axhline(0, ls="--", color="gray")
ax.axvline(K, ls=":", color="gray", alpha=0.5)
ax.fill_between(S_line, diff, 0, where=(diff > 0), alpha=0.3, color="green",
                label="Heston > BS")
ax.fill_between(S_line, diff, 0, where=(diff < 0), alpha=0.3, color="red",
                label="Heston < BS")
ax.set(xlabel="Spot S", ylabel="V_Heston - V_BS",
       title="Effet vol stochastique (v0=0.04, rho=-0.7)")
ax.legend(); ax.grid(alpha=0.3)

# Points MC
for i, s in enumerate(S_bench):
    axes[0].errorbar(s, mc_prices[i], yerr=2*mc_stderr[i],
                     fmt="ko", ms=4, capsize=3, zorder=5)

plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG, "heston_vs_bs.png"), dpi=150)
plt.close()
print("Figure -> outputs/figures/heston_vs_bs.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 : Surface V(S, v) à tau = T
# ─────────────────────────────────────────────────────────────────────────────
S_vals = np.linspace(50, 200, 60)
v_vals = np.linspace(0.005, 0.5, 40)
V_surf, SS, VV = model.predict(S_vals, v_vals, tau_val=T, device=device)
V_surf_bs = np.vectorize(
    lambda s, v: bs_call_price(s, K, T, r, np.sqrt(v))
)(SS, VV)

fig = plt.figure(figsize=(16, 5))
ax1 = fig.add_subplot(131, projection="3d")
ax1.plot_surface(SS, np.sqrt(VV) * 100, V_surf, cmap="plasma", alpha=0.85)
ax1.set(xlabel="S", ylabel="Vol (%)", zlabel="V",
        title="Surface Heston PINN")

ax2 = fig.add_subplot(132, projection="3d")
ax2.plot_surface(SS, np.sqrt(VV) * 100, V_surf_bs, cmap="viridis", alpha=0.85)
ax2.set(xlabel="S", ylabel="Vol (%)", zlabel="V",
        title="Surface BS (sigma=sqrt(v))")

ax3 = fig.add_subplot(133, projection="3d")
diff_surf = V_surf - V_surf_bs
ax3.plot_surface(SS, np.sqrt(VV) * 100, diff_surf, cmap="RdBu_r", alpha=0.85)
ax3.set(xlabel="S", ylabel="Vol (%)", zlabel="Diff",
        title="Heston - BS")

plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG, "heston_surface_SV.png"), dpi=150)
plt.close()
print("Figure -> outputs/figures/heston_surface_SV.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 : Smile de volatilité implicite (strikes variés, tau=T)
# ─────────────────────────────────────────────────────────────────────────────
K_vals = np.linspace(70, 140, 25)
S_atm  = 100.0

# Prices from PINN at v=v0
S_t = torch.tensor(np.full_like(K_vals, S_atm), dtype=torch.float32, device=device).unsqueeze(1)
v_t = torch.full_like(S_t, v0)
tau_t = torch.full_like(S_t, T)
with torch.no_grad():
    pinn_prices = model(S_t, v_t, tau_t).cpu().numpy().flatten()

# Semi-analytical prices
print("\nCalcul du smile de vol implicite (semi-analytique)...")
heston_prices_sa = np.array([
    heston_call_price(S_atm, k, T, r, kappa, theta, sigma_v, rho, v0)
    for k in K_vals
])

# Implied vols
iv_pinn   = np.array([implied_vol(p, S_atm, k, T, r) for p, k in zip(pinn_prices, K_vals)])
iv_heston = np.array([implied_vol(p, S_atm, k, T, r) for p, k in zip(heston_prices_sa, K_vals)])
iv_bs_flat = np.full_like(K_vals, np.sqrt(v0))  # BS flat vol

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
ax = axes[0]
ax.plot(K_vals, iv_heston * 100, "b-o", ms=4, lw=2, label="Heston (semi-analytique)")
ax.plot(K_vals, iv_pinn   * 100, "r--s", ms=4, lw=2, label="Heston PINN")
ax.axhline(np.sqrt(v0) * 100, ls=":", color="gray", label=f"BS flat ({np.sqrt(v0)*100:.0f}%)")
ax.axvline(K, ls=":", color="gray", alpha=0.5)
ax.set(xlabel="Strike K", ylabel="Vol implicite (%)",
       title="Smile de vol implicite (S=100, T=1)")
ax.legend(); ax.grid(alpha=0.3)

# Différence IV
ax = axes[1]
iv_diff = iv_pinn - iv_heston
ax.plot(K_vals, iv_diff * 100, "g-o", ms=3)
ax.axhline(0, ls="--", color="gray")
ax.set(xlabel="Strike K", ylabel="IV_PINN - IV_Heston (%)",
       title="Ecart de vol implicite PINN vs semi-analytique")
ax.grid(alpha=0.3)
valid = ~np.isnan(iv_diff)
if valid.sum() > 0:
    mae_iv = np.abs(iv_diff[valid]).mean() * 100
    ax.set_title(f"Ecart IV PINN vs analytique\nMAE = {mae_iv:.2f} vol%")

plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG, "heston_implied_vol_smile.png"), dpi=150)
plt.close()
print("Figure -> outputs/figures/heston_implied_vol_smile.png")

# ─────────────────────────────────────────────────────────────────────────────
# Métriques finales
# ─────────────────────────────────────────────────────────────────────────────
valid_iv = ~np.isnan(iv_pinn) & ~np.isnan(iv_heston)
mae_iv_pct = float(np.abs(iv_pinn[valid_iv] - iv_heston[valid_iv]).mean() * 100) \
    if valid_iv.sum() > 0 else float("nan")

metrics = {
    "MAE_vs_MC":       round(float(mae_vs_mc), 4),
    "MAE_iv_pct":      round(mae_iv_pct, 3),
    "final_loss":      round(float(history.total[-1]), 6),
    "train_time_s":    round(train_time, 1),
    "epochs":          len(history.total),
    "n_params":        n_params,
}
pd.DataFrame([metrics]).to_csv(
    os.path.join(OUT_RES, "heston_metrics.csv"), index=False
)

print("\n========== RESULTATS HESTON ==========")
print(f"  MAE PINN vs MC (7 spots) : {mae_vs_mc:.4f} $")
print(f"  MAE vol implicite        : {mae_iv_pct:.3f} vol%")
print(f"  Loss finale              : {history.total[-1]:.4e}")
print(f"  Duree                    : {train_time:.1f} s")
print(f"  Parametres               : {n_params:,}")
print("=======================================\n")
