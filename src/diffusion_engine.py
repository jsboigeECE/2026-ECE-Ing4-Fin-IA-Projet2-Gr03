import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class DiffusionEngine:
    def __init__(self, timesteps=300, beta_start=0.0001, beta_end=0.02):
        self.T = timesteps
        
        # 1. Définition du barème de bruit (Beta)
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        
        # 2. Calcul des variables intermédiaires (Alpha)
        # alpha = 1 - beta
        self.alphas = 1. - self.betas
        # alpha_cumprod = produit cumulé des alphas (noté \bar{\alpha} dans les papiers)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        
        # 3. Calcul pour l'échantillonnage direct (Forward Process)
        # Formule : x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * epsilon
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def forward_diffusion(self, x_0, t):
        """
        Ajoute du bruit à x_0 à l'étape t de manière directe.
        x_0: Données initiales (Rendements normalisés)
        t: Index de l'étape de diffusion
        """
        noise = torch.randn_like(x_0)
        
        # Récupération des coefficients pour l'étape t
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        # Application de la formule de diffusion
        x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise
        
        return x_t, noise