"""
Module d'explicabilité SHAP pour le Credit Scoring XAI.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Union
import logging

from src.config import SHAP_CONFIG

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    Classe pour générer des explications SHAP pour les modèles de scoring de crédit.
    """
    
    def __init__(self, model, feature_names: Optional[List[str]] = None):
        """
        Initialise l'explainer SHAP.
        
        Args:
            model: Modèle entraîné (doit avoir une méthode predict_proba)
            feature_names: Noms des features
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        
    def fit(self, X_background: pd.DataFrame, explainer_type: str = 'auto') -> None:
        """
        Initialise l'explainer SHAP avec des données de fond.
        
        Args:
            X_background: Données de fond pour l'explainer
            explainer_type: Type d'explainer ('auto', 'tree', 'kernel', 'deep')
        """
        logger.info(f"Initialisation de l'explainer SHAP (type: {explainer_type})...")
        
        # Déterminer le type d'explainer
        if explainer_type == 'auto':
            # Essayer TreeExplainer pour les modèles basés sur des arbres
            try:
                self.explainer = shap.TreeExplainer(self.model.model)
                logger.info("TreeExplainer utilisé")
            except:
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    shap.sample(X_background, 100)
                )
                logger.info("KernelExplainer utilisé")
        elif explainer_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model.model)
        elif explainer_type == 'kernel':
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                shap.sample(X_background, 100)
            )
        else:
            raise ValueError(f"Type d'explainer non supporté: {explainer_type}")
        
        self.expected_value = self.explainer.expected_value
        if isinstance(self.expected_value, list):
            self.expected_value = self.expected_value[1]  # Classe positive
        self.expected_value = float(np.asarray(self.expected_value).ravel()[0])

        logger.info(f"Explainer SHAP initialisé. Expected value: {self.expected_value:.4f}")
    
    def explain(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calcule les valeurs SHAP pour les données fournies.
        
        Args:
            X: Données à expliquer
            
        Returns:
            np.ndarray: Valeurs SHAP
        """
        if self.explainer is None:
            raise ValueError("L'explainer doit être initialisé avec fit() d'abord")
        
        logger.info(f"Calcul des valeurs SHAP pour {X.shape[0]} échantillons...")
        
        # Calculer les valeurs SHAP
        self.shap_values = self.explainer.shap_values(X)
        
        # Si shap_values est une liste (classification binaire), prendre la classe positive
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        
        logger.info(f"Valeurs SHAP calculées. Shape: {self.shap_values.shape}")
        
        return self.shap_values
    
    def get_feature_importance(self, importance_type: str = 'mean_abs') -> pd.DataFrame:
        """
        Calcule l'importance globale des features basée sur les valeurs SHAP.
        
        Args:
            importance_type: Type d'importance ('mean_abs', 'mean', 'std')
            
        Returns:
            pd.DataFrame: DataFrame avec importance des features
        """
        if self.shap_values is None:
            raise ValueError("Les valeurs SHAP doivent être calculées d'abord")
        
        if importance_type == 'mean_abs':
            importance = np.mean(np.abs(self.shap_values), axis=0)
        elif importance_type == 'mean':
            importance = np.mean(self.shap_values, axis=0)
        elif importance_type == 'std':
            importance = np.std(self.shap_values, axis=0)
        else:
            raise ValueError(f"Type d'importance non supporté: {importance_type}")
        
        feature_names = self.feature_names if self.feature_names else [f'feature_{i}' for i in range(len(importance))]
        
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df_importance
    
    def plot_summary(self, shap_values: Optional[np.ndarray] = None, max_display: Optional[int] = None) -> plt.Figure:
        """
        Génère un plot summary SHAP.
        
        Args:
            shap_values: Valeurs SHAP (si None, utilise self.shap_values)
            max_display: Nombre maximum de features à afficher
            
        Returns:
            plt.Figure: Figure matplotlib
        """
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("Les valeurs SHAP doivent être calculées d'abord")
        
        if max_display is None:
            max_display = SHAP_CONFIG['max_display']
        
        fig, ax = plt.subplots(figsize=SHAP_CONFIG['plot_size'])
        shap.summary_plot(shap_values, feature_names=self.feature_names, 
                         max_display=max_display, show=False)
        plt.tight_layout()
        
        return fig
    
    def plot_waterfall(self, instance_idx: int, X: pd.DataFrame, 
                      shap_values: Optional[np.ndarray] = None) -> plt.Figure:
        """
        Génère un plot waterfall pour une instance spécifique.
        
        Args:
            instance_idx: Index de l'instance
            X: Données
            shap_values: Valeurs SHAP (si None, utilise self.shap_values)
            
        Returns:
            plt.Figure: Figure matplotlib
        """
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("Les valeurs SHAP doivent être calculées d'abord")
        
        fig, ax = plt.subplots(figsize=SHAP_CONFIG['plot_size'])
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[instance_idx],
                base_values=self.expected_value,
                data=X.iloc[instance_idx],
                feature_names=self.feature_names
            ),
            show=False
        )
        plt.tight_layout()
        
        return fig
    
    def plot_force(self, instance_idx: int, X: pd.DataFrame,
                   shap_values: Optional[np.ndarray] = None) -> plt.Figure:
        """
        Génère un plot force pour une instance spécifique.
        
        Args:
            instance_idx: Index de l'instance
            X: Données
            shap_values: Valeurs SHAP (si None, utilise self.shap_values)
            
        Returns:
            plt.Figure: Figure matplotlib
        """
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("Les valeurs SHAP doivent être calculées d'abord")
        
        fig, ax = plt.subplots(figsize=SHAP_CONFIG['plot_size'])
        shap.force_plot(
            self.expected_value,
            shap_values[instance_idx],
            X.iloc[instance_idx],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        plt.tight_layout()
        
        return fig
    
    def plot_dependence(self, feature: str, X: pd.DataFrame,
                       shap_values: Optional[np.ndarray] = None) -> plt.Figure:
        """
        Génère un plot de dépendance pour une feature spécifique.
        
        Args:
            feature: Nom de la feature
            X: Données
            shap_values: Valeurs SHAP (si None, utilise self.shap_values)
            
        Returns:
            plt.Figure: Figure matplotlib
        """
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("Les valeurs SHAP doivent être calculées d'abord")
        
        if self.feature_names:
            feature_idx = self.feature_names.index(feature)
        else:
            feature_idx = int(feature)
        
        fig, ax = plt.subplots(figsize=SHAP_CONFIG['plot_size'])
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X,
            feature_names=self.feature_names,
            show=False
        )
        plt.tight_layout()
        
        return fig
    
    def get_local_explanation(self, instance_idx: int, X: pd.DataFrame,
                             shap_values: Optional[np.ndarray] = None) -> Dict:
        """
        Retourne une explication locale pour une instance spécifique.
        
        Args:
            instance_idx: Index de l'instance
            X: Données
            shap_values: Valeurs SHAP (si None, utilise self.shap_values)
            
        Returns:
            Dict: Explication locale
        """
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("Les valeurs SHAP doivent être calculées d'abord")
        
        instance_shap = shap_values[instance_idx]
        instance_data = X.iloc[instance_idx]
        
        # Trier par valeur absolue
        sorted_indices = np.argsort(np.abs(instance_shap))[::-1]
        
        explanation = {
            'base_value': float(self.expected_value),
            'prediction': float(self.expected_value + np.sum(instance_shap)),
            'features': []
        }
        
        for idx in sorted_indices:
            feature_name = self.feature_names[idx] if self.feature_names else f'feature_{idx}'
            explanation['features'].append({
                'feature': feature_name,
                'value': float(instance_data[idx]),
                'shap_value': float(instance_shap[idx]),
                'impact': 'positive' if instance_shap[idx] > 0 else 'negative'
            })
        
        return explanation
    
    def save_shap_values(self, filepath: str) -> None:
        """
        Sauvegarde les valeurs SHAP.
        
        Args:
            filepath: Chemin de sauvegarde
        """
        if self.shap_values is None:
            raise ValueError("Les valeurs SHAP doivent être calculées d'abord")
        
        np.save(filepath, self.shap_values)
        logger.info(f"Valeurs SHAP sauvegardées dans {filepath}")
    
    def load_shap_values(self, filepath: str) -> None:
        """
        Charge les valeurs SHAP.
        
        Args:
            filepath: Chemin du fichier à charger
        """
        self.shap_values = np.load(filepath)
        logger.info(f"Valeurs SHAP chargées depuis {filepath}")


if __name__ == "__main__":
    # Test du module
    print("Test du module SHAPExplainer...")
    from src.data_loader import load_data
    from src.preprocessing import DataPreprocessor, split_data
    from src.models.xgboost_model import XGBoostModel
    
    # Charger et préparer les données
    X, y = load_data()
    preprocessor = DataPreprocessor()
    X_transformed = preprocessor.fit_transform(X)
    
    # Split des données
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_transformed, y)
    
    # Créer et entraîner le modèle
    model = XGBoostModel()
    model.train(X_train, y_train, X_val, y_val)
    
    # Créer l'explainer SHAP
    explainer = SHAPExplainer(model, feature_names=preprocessor.feature_names)
    explainer.fit(X_train)
    
    # Calculer les valeurs SHAP
    shap_values = explainer.explain(X_test)
    
    # Importance des features
    importance = explainer.get_feature_importance()
    print(f"\nTop 10 features par importance SHAP:")
    print(importance.head(10))
    
    # Explication locale
    local_expl = explainer.get_local_explanation(0, X_test)
    print(f"\nExplication locale pour l'instance 0:")
    print(f"Base value: {local_expl['base_value']:.4f}")
    print(f"Prediction: {local_expl['prediction']:.4f}")
    print(f"Top 5 features:")
    for feat in local_expl['features'][:5]:
        print(f"  {feat['feature']}: {feat['shap_value']:.4f} ({feat['impact']})")