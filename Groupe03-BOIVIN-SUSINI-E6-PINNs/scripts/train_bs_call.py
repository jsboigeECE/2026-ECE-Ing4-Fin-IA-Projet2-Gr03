"""
scripts/train_bs_call.py
========================
Entraîne le PINN Black-Scholes (call européen) et sauvegarde dans outputs/ :

    outputs/figures/loss_curve.png
    outputs/figures/pinn_vs_analytical.png
    outputs/figures/error_map.png
    outputs/figures/price_surface.png
    outputs/results/metrics.csv
    outputs/checkpoints/model_final.pt

Exécution :
    python scripts/train_bs_call.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # pas d'affichage interactif (safe en script)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (active la projection 3D)

from src.analytics.black_scholes_formula import bs_call_price
from src.models.black_scholes_pinn import BlackScholesPINN
from src.training.trainer import PINNTrainer
from src.training.metrics import evaluate_on_grid, save_metrics

# ─────────────────────────────────────────────────────────────────────────────
# Paramètres
# ─────────────────────────────────────────────────────────────────────────────
K     = 100.0
T     = 1.0
r     = 0.05
sigma = 0.20
S_max = 300.0

HIDDEN_SIZES  = [50, 50, 50, 50]
N_EPOCHS      = 10_000
N_COLL        = 5_000
N_BC          = 500
LR            = 1e-3
LAMBDA_PDE    = 1.0
LAMBDA_BC     = 10.0
LAMBDA_IC     = 10.0
LBFGS_EPOCHS  = 200
SAVE_EVERY    = 1_000
PATIENCE      = 500

OUT_FIG  = "outputs/figures"
OUT_CKPT = "outputs/checkpoints"
OUT_RES  = "outputs/results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_CKPT, exist_ok=True)
os.makedirs(OUT_RES, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Modèle + entraînement
# ─────────────────────────────────────────────────────────────────────────────
model = BlackScholesPINN(
    K=K, r=r, sigma=sigma, T=T, S_max=S_max,
    hidden_sizes=HIDDEN_SIZES,
    lambda_pde=LAMBDA_PDE, lambda_bc=LAMBDA_BC, lambda_ic=LAMBDA_IC,
)
n_params = sum(p.numel() for p in model.parameters())
print(f"Paramètres réseau : {n_params:,}")

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

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint final
# ─────────────────────────────────────────────────────────────────────────────
ckpt_path = os.path.join(OUT_CKPT, "model_final.pt")
torch.save(model.state_dict(), ckpt_path)
print(f"Modèle sauvegardé -> {ckpt_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 : Courbes de loss
# ─────────────────────────────────────────────────────────────────────────────
epochs = np.arange(1, len(history.total) + 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
ax = axes[0]
ax.semilogy(epochs, history.total, label="Total")
ax.semilogy(epochs, history.pde,   label="L_pde", ls="--")
ax.semilogy(epochs, history.bc,    label="L_bc",  ls=":")
ax.semilogy(epochs, history.ic,    label="L_ic",  ls="-.")
ax.set(xlabel="Epoch", ylabel="Loss (log)", title="Décomposition de la loss")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.semilogy(epochs, history.total)
ax.set(xlabel="Epoch", ylabel="Loss totale (log)", title="Convergence globale")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG, "loss_curve.png"), dpi=150)
plt.close()
print("Figure sauvegardée -> outputs/figures/loss_curve.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 : PINN vs Analytique — coupes à différentes maturités
# ─────────────────────────────────────────────────────────────────────────────
model.eval()
S_vals = np.linspace(10, 280, 400)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
ax = axes[0]
for tau_val in [T, T / 2, T / 4]:
    V_ana = bs_call_price(S_vals, K, tau_val, r, sigma)
    S_t   = torch.tensor(S_vals, dtype=torch.float32, device=device).unsqueeze(1)
    tau_t = torch.full_like(S_t, tau_val)
    with torch.no_grad():
        V_pinn = model(S_t, tau_t).cpu().numpy().flatten()
    ax.plot(S_vals, V_ana,  lw=2,           label=f"Analytique τ={tau_val:.2f}")
    ax.plot(S_vals, V_pinn, lw=1.5, ls="--", label=f"PINN      τ={tau_val:.2f}")

ax.axvline(K, ls=":", color="gray", alpha=0.6)
ax.set(xlabel="Spot S", ylabel="V", title="PINN vs Analytique")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Erreur absolue à tau = T
V_ana_T  = bs_call_price(S_vals, K, T, r, sigma)
S_t = torch.tensor(S_vals, dtype=torch.float32, device=device).unsqueeze(1)
tau_T = torch.full_like(S_t, T)
with torch.no_grad():
    V_pinn_T = model(S_t, tau_T).cpu().numpy().flatten()

ax = axes[1]
ax.plot(S_vals, np.abs(V_pinn_T - V_ana_T), color="red")
ax.axvline(K, ls=":", color="gray", alpha=0.6)
ax.set(xlabel="Spot S", ylabel="|V_PINN − V_BS|",
       title=f"Erreur absolue (τ={T})")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG, "pinn_vs_analytical.png"), dpi=150)
plt.close()
print("Figure sauvegardée -> outputs/figures/pinn_vs_analytical.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 : Carte d'erreur absolue sur la grille (S, tau)
# ─────────────────────────────────────────────────────────────────────────────
S_grid   = np.linspace(10, 280, 80)
tau_grid = np.linspace(0.01, T, 60)

res = evaluate_on_grid(model, bs_call_price, K, T, r, sigma,
                       S_vals=S_grid, tau_vals=tau_grid, device=device)
err_grid = np.abs(res["V_pred"] - res["V_true"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
im0 = axes[0].contourf(res["S_grid"], res["tau_grid"], res["V_pred"],
                        levels=30, cmap="viridis")
plt.colorbar(im0, ax=axes[0])
axes[0].set(xlabel="Spot S", ylabel="τ", title="Surface de prix PINN")

im1 = axes[1].contourf(res["S_grid"], res["tau_grid"], err_grid,
                        levels=30, cmap="Reds")
plt.colorbar(im1, ax=axes[1])
axes[1].set(xlabel="Spot S", ylabel="τ", title="|V_PINN − V_BS|")

plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG, "error_map.png"), dpi=150)
plt.close()
print("Figure sauvegardée -> outputs/figures/error_map.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 : Surface 3D V(S, tau)
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5))
ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(res["S_grid"], res["tau_grid"], res["V_pred"],
                 cmap="plasma", alpha=0.85)
ax1.set(xlabel="S", ylabel="τ", zlabel="V", title="Surface PINN")

ax2 = fig.add_subplot(122, projection="3d")
ax2.plot_surface(res["S_grid"], res["tau_grid"], res["V_true"],
                 cmap="viridis", alpha=0.85)
ax2.set(xlabel="S", ylabel="τ", zlabel="V", title="Surface Analytique")

plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG, "price_surface.png"), dpi=150)
plt.close()
print("Figure sauvegardée -> outputs/figures/price_surface.png")

# ─────────────────────────────────────────────────────────────────────────────
# Métriques finales
# ─────────────────────────────────────────────────────────────────────────────
save_metrics(
    res,
    path=os.path.join(OUT_RES, "metrics.csv"),
    extra={
        "epochs":      len(history.total),
        "final_loss":  history.total[-1],
        "train_time_s": round(train_time, 1),
    },
)

print("\n========== RÉSULTATS ==========")
print(f"  MAE      : {res['MAE']:.4f} $")
print(f"  RMSE     : {res['RMSE']:.4f} $")
print(f"  Max err  : {res['max_err']:.4f} $")
print(f"  Rel. MAE : {res['rel_mae']:.2f} %")
print(f"  Durée    : {train_time:.1f} s")
mae_ok = res["MAE"] < 0.5
print(f"\n  {'✅' if mae_ok else '❌'} Critère minimum (MAE < 0.50$) : {'OK' if mae_ok else 'NON ATTEINT'}")
print("================================\n")
