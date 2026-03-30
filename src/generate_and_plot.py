import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats  # AJOUT : Pour les calculs de niveau "Bon"
from src.model import DiffusionModel

def generate_price_simulation():
    # 1. Configuration
    device = torch.device("cpu")
    model = DiffusionModel(input_dim=1).to(device)
    model.load_state_dict(torch.load("model_diffusion_E3.pth"))
    model.eval()

    # 2. Génération des rendements (Le "cerveau" de l'IA)
    x = torch.randn(1, 64, 1) # Bruit de départ
    steps = 300
    
    with torch.no_grad():
        for i in reversed(range(steps)):
            t = torch.tensor([i])
            pred_noise = model(x, t)
            x = x - (0.05 * pred_noise) # On nettoie doucement
            if i > 0:
                x = x + torch.randn_like(x) * 0.01 # Petit chaos réaliste

    # 3. Transformation en "Vrais Chiffres"
    # On ramène les rendements à une échelle réaliste
    rendements = x.squeeze().numpy() * 0.1 
    
    # 4. ANALYSE QUANTITATIVE (OBJECTIF "BON")
    # On calcule les indicateurs que le prof attend
    kurt = stats.kurtosis(rendements)  # Mesure des "Queues épaisses"
    skew = stats.skew(rendements)      # Mesure de l'asymétrie
    # Volatilité annualisée (on multiplie par racine de 252 jours de bourse)
    vol_annuelle = np.std(rendements) * np.sqrt(252) * 100

    # Affichage des stats dans le terminal
    print("\n" + "="*40)
    print("🎯 ANALYSE DES FAITS STYLISÉS ")
    print("="*40)
    print(f"Kurtosis (Excès de risque) : {kurt:.2f}")
    print(f"Skewness (Asymétrie)       : {skew:.2f}")
    print(f"Volatilité Annualisée      : {vol_annuelle:.2f}%")
    print("="*40)
    if kurt > 0:
        print("✅ SUCCÈS : Présence de 'Fat Tails' (Queues épaisses).")
    else:
        print("ℹ️ NOTE : Distribution proche d'une loi normale.")
    print("="*40 + "\n")

    # 5. Conversion en Prix
    prix_initial = 5100
    prix_simules = [prix_initial]
    
    for r in rendements:
        nouveau_prix = prix_simules[-1] * (1 + r)
        prix_simules.append(nouveau_prix)

    # 6. Affichage Double
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4)

    # Graphique du haut : Les variations (Rendements)
    ax1.plot(rendements * 100, color='blue', label='Variation Journalière (%)')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.set_title(f"Ce que l'IA génère : Rendements (Kurtosis: {kurt:.2f})")
    ax1.set_ylabel("Hausse / Baisse en %")
    ax1.legend()
    ax1.grid(True)

    # Graphique du bas : Le Prix en Dollars
    ax2.plot(prix_simules, color='green', linewidth=2, label='Prix S&P 500 Simulé ($)')
    ax2.set_title(f"Simulation du Prix (Volatilité: {vol_annuelle:.1f}%)")
    ax2.set_xlabel("Jours")
    ax2.set_ylabel("Prix en Dollars ($)")
    ax2.legend()
    ax2.grid(True)

    print(f"Simulation terminée ! Prix de départ : {prix_initial}$ -> Prix final : {prix_simules[-1]:.2f}$")
    plt.show()

if __name__ == "__main__":
    generate_price_simulation()