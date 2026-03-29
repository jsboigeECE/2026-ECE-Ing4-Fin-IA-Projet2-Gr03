"""
Module d'explicabilité LIME pour le Credit Scoring XAI.
"""

import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple
import logging

from src.config import LIME_CONFIG

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LIMEExplainer:
    """
    Classe pour générer des explications LIME pour les modèles de scoring de crédit.
    """
    
    def __init__(
        self,
        model,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialise l'explainer LIME.
        
        Args:
            model: Modèle entraîné (doit avoir une méthode predict_proba)
            feature_names: Noms des features
            class_names: Noms des classes
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names or ['Bad Credit', 'Good Credit']
        self.explainer = None
        self.explanations = {}
        
    def fit(self, X_train: pd.DataFrame, **kwargs) -> None:
        """
        Initialise l'explainer LIME avec les données d'entraînement.
        
        Args:
            X_train: Données d'entraînement
            **kwargs: Arguments supplémentaires pour LimeTabularExplainer
        """
        logger.info("Initialisation de l'explainer LIME...")
        
        # Paramètres par défaut
        default_params = {
            'training_data': X_train.values,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'mode': 'classification',
            'discretize_continuous': True,
            'kernel_width': 3,
            'verbose': False
        }
        
        # Fusionner avec les paramètres fournis
        params = {**default_params, **kwargs}
        
        self.explainer = lime.lime_tabular.LimeTabularExplainer(**params)
        
        logger.info(f"Explainer LIME initialisé avec {X_train.shape[0]} échantillons d'entraînement")
    
    def explain_instance(
        self,
        instance: np.ndarray,
        num_features: Optional[int] = None,
        num_samples: Optional[int] = None
    ) -> lime.explanation.Explanation:
        """
        Génère une explication LIME pour une instance spécifique.
        
        Args:
            instance: Instance à expliquer
            num_features: Nombre de features à inclure dans l'explication
            num_samples: Nombre d'échantillons pour l'approximation locale
            
        Returns:
            lime.explanation.Explanation: Explication LIME
        """
        if self.explainer is None:
            raise ValueError("L'explainer doit être initialisé avec fit() d'abord")
        
        if num_features is None:
            num_features = LIME_CONFIG['num_features']
        
        if num_samples is None:
            num_samples = LIME_CONFIG['num_samples']
        
        # Fonction de prédiction pour LIME
        def predict_fn(x):
            return self.model.model.predict_proba(x)
        
        # Générer l'explication (labels=(0,1) garantit les deux classes)
        explanation = self.explainer.explain_instance(
            instance,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            labels=(0, 1)
        )
        
        return explanation
    
    def explain_batch(
        self,
        X: pd.DataFrame,
        num_features: Optional[int] = None,
        num_samples: Optional[int] = None
    ) -> Dict[int, lime.explanation.Explanation]:
        """
        Génère des explications LIME pour un batch d'instances.
        
        Args:
            X: Données à expliquer
            num_features: Nombre de features à inclure dans l'explication
            num_samples: Nombre d'échantillons pour l'approximation locale
            
        Returns:
            Dict[int, lime.explanation.Explanation]: Dictionnaire des explications
        """
        logger.info(f"Génération d'explications LIME pour {X.shape[0]} instances...")
        
        explanations = {}
        for idx in range(X.shape[0]):
            explanations[idx] = self.explain_instance(
                X.iloc[idx].values,
                num_features=num_features,
                num_samples=num_samples
            )
        
        self.explanations = explanations
        logger.info(f"Explications LIME générées pour {len(explanations)} instances")
        
        return explanations
    
    def get_local_explanation(
        self,
        instance: np.ndarray,
        explanation: Optional[lime.explanation.Explanation] = None
    ) -> Dict:
        """
        Retourne une explication locale structurée.
        
        Args:
            instance: Instance à expliquer
            explanation: Explication LIME (si None, génère une nouvelle)
            
        Returns:
            Dict: Explication locale structurée
        """
        if explanation is None:
            explanation = self.explain_instance(instance)
        
        # Probabilité de la classe positive via le modèle (fiable indépendamment de la version LIME)
        local_pred = float(self.model.predict_proba(instance.reshape(1, -1))[0])

        # Intercept : dict ou array selon la version LIME
        intercept_raw = explanation.intercept
        if isinstance(intercept_raw, dict):
            intercept = intercept_raw.get(1, intercept_raw.get(0, 0.0))
        elif hasattr(intercept_raw, '__len__') and len(intercept_raw) > 1:
            intercept = float(intercept_raw[1])
        elif hasattr(intercept_raw, '__len__') and len(intercept_raw) > 0:
            intercept = float(intercept_raw[0])
        else:
            intercept = float(intercept_raw) if intercept_raw is not None else 0.0

        # Label pour as_list : essayer 1 (classe positive), fallback 0
        try:
            raw_features = explanation.as_list(label=1)
        except Exception:
            try:
                raw_features = explanation.as_list(label=0)
            except Exception:
                raw_features = explanation.as_list()

        # Extraire les features et leurs contributions
        features_list = []
        for feature, value in raw_features:
            features_list.append({
                'feature': feature,
                'contribution': value,
                'impact': 'positive' if value > 0 else 'negative'
            })
        
        # Trier par contribution absolue
        features_list.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        explanation_dict = {
            'predicted_class': explanation.top_labels[0],
            'predicted_probability': float(local_pred),
            'intercept': float(intercept),
            'features': features_list
        }
        
        return explanation_dict
    
    def plot_explanation(
        self,
        instance: np.ndarray,
        explanation: Optional[lime.explanation.Explanation] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Génère un plot de l'explication LIME.
        
        Args:
            instance: Instance à expliquer
            explanation: Explication LIME (si None, génère une nouvelle)
            figsize: Taille de la figure
            
        Returns:
            plt.Figure: Figure matplotlib
        """
        if explanation is None:
            explanation = self.explain_instance(instance)
        
        fig = plt.figure(figsize=figsize)
        explanation.as_pyplot_figure()
        plt.tight_layout()
        
        return fig
    
    def compare_with_shap(
        self,
        instance: np.ndarray,
        shap_explainer,
        X: pd.DataFrame,
        instance_idx: int
    ) -> Dict:
        """
        Compare les explications LIME et SHAP pour une instance.
        
        Args:
            instance: Instance à expliquer
            shap_explainer: Explainer SHAP
            X: Données
            instance_idx: Index de l'instance
            
        Returns:
            Dict: Comparaison des explications
        """
        # Explication LIME
        lime_exp = self.explain_instance(instance)
        lime_dict = self.get_local_explanation(instance, lime_exp)
        
        # Explication SHAP
        shap_dict = shap_explainer.get_local_explanation(instance_idx, X)
        
        # Comparer les top features
        lime_top_features = [f['feature'] for f in lime_dict['features'][:5]]
        shap_top_features = [f['feature'] for f in shap_dict['features'][:5]]
        
        comparison = {
            'lime': lime_dict,
            'shap': shap_dict,
            'common_top_features': list(set(lime_top_features) & set(shap_top_features)),
            'lime_only_top_features': list(set(lime_top_features) - set(shap_top_features)),
            'shap_only_top_features': list(set(shap_top_features) - set(lime_top_features))
        }
        
        return comparison
    
    def get_feature_importance_aggregate(
        self,
        explanations: Optional[Dict[int, lime.explanation.Explanation]] = None
    ) -> pd.DataFrame:
        """
        Calcule l'importance agrégée des features sur plusieurs explications.
        
        Args:
            explanations: Dictionnaire des explications (si None, utilise self.explanations)
            
        Returns:
            pd.DataFrame: DataFrame avec importance agrégée des features
        """
        if explanations is None:
            explanations = self.explanations
        
        if not explanations:
            raise ValueError("Aucune explication disponible")
        
        # Agréger les contributions
        feature_contributions = {}
        
        for exp in explanations.values():
            for feature, contribution in exp.as_list():
                if feature not in feature_contributions:
                    feature_contributions[feature] = []
                feature_contributions[feature].append(abs(contribution))
        
        # Calculer la moyenne
        feature_importance = {
            'feature': [],
            'mean_abs_contribution': [],
            'std_contribution': []
        }
        
        for feature, contributions in feature_contributions.items():
            feature_importance['feature'].append(feature)
            feature_importance['mean_abs_contribution'].append(np.mean(contributions))
            feature_importance['std_contribution'].append(np.std(contributions))
        
        df_importance = pd.DataFrame(feature_importance)
        df_importance = df_importance.sort_values('mean_abs_contribution', ascending=False)
        
        return df_importance


if __name__ == "__main__":
    # Test du module
    print("Test du module LIMEExplainer...")
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
    
    # Créer l'explainer LIME
    explainer = LIMEExplainer(model, feature_names=preprocessor.feature_names)
    explainer.fit(X_train)
    
    # Expliquer une instance
    instance = X_test.iloc[0].values
    explanation = explainer.explain_instance(instance)
    
    print(f"\nExplication LIME pour l'instance 0:")
    print(f"Classe prédite: {explanation.top_labels[0]}")
    print(f"Probabilité: {model.predict_proba(instance.reshape(1, -1))[0]:.4f}")
    print(f"\nTop 5 features:")
    for feature, contribution in explanation.as_list()[:5]:
        print(f"  {feature}: {contribution:.4f}")
    
    # Explication structurée
    local_expl = explainer.get_local_explanation(instance, explanation)
    print(f"\nExplication structurée:")
    print(f"Probabilité prédite: {local_expl['predicted_probability']:.4f}")
    print(f"Intercept: {local_expl['intercept']:.4f}")