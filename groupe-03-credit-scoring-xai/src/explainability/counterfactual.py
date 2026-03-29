"""
Module d'explications contrefactuelles pour le Credit Scoring XAI.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging

from src.config import NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CounterfactualExplainer:
    """
    Classe pour générer des explications contrefactuelles pour les modèles de scoring de crédit.
    
    Les explications contrefactuelles répondent à la question : "Que faudrait-il changer
    pour que cette demande de crédit soit acceptée ?"
    """
    
    def __init__(
        self,
        model,
        feature_names: List[str],
        categorical_features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None
    ):
        """
        Initialise l'explainer contrefactuel.
        
        Args:
            model: Modèle entraîné (doit avoir une méthode predict_proba)
            feature_names: Noms des features
            categorical_features: Liste des features catégorielles
            numerical_features: Liste des features numériques
        """
        self.model = model
        self.feature_names = feature_names
        self.categorical_features = categorical_features or CATEGORICAL_COLUMNS
        self.numerical_features = numerical_features or NUMERICAL_COLUMNS
        
    def generate_counterfactual(
        self,
        instance: pd.Series,
        target_class: int = 1,
        max_iterations: int = 1000,
        learning_rate: float = 0.1,
        lambda_param: float = 0.1,
        tolerance: float = 0.01
    ) -> Dict:
        """
        Génère une explication contrefactuelle pour une instance.
        
        Args:
            instance: Instance à expliquer
            target_class: Classe cible (1 = Good Credit, 0 = Bad Credit)
            max_iterations: Nombre maximum d'itérations
            learning_rate: Taux d'apprentissage pour la descente de gradient
            lambda_param: Paramètre de régularisation
            tolerance: Tolérance pour la convergence
            
        Returns:
            Dict: Explication contrefactuelle
        """
        logger.info(f"Génération d'explication contrefactuelle pour target_class={target_class}...")
        
        # Prédiction initiale
        initial_pred = self.model.predict_proba(instance.values.reshape(1, -1))[0]
        initial_class = self.model.predict(instance.values.reshape(1, -1))[0]
        
        logger.info(f"Prédiction initiale: {initial_pred:.4f} (classe: {initial_class})")
        
        # Si déjà dans la classe cible, retourner l'instance originale
        if initial_class == target_class:
            return {
                'success': True,
                'message': 'L\'instance est déjà dans la classe cible',
                'original_instance': instance.to_dict(),
                'counterfactual_instance': instance.to_dict(),
                'changes': [],
                'original_prediction': float(initial_pred),
                'counterfactual_prediction': float(initial_pred)
            }
        
        # Initialiser l'instance contrefactuelle
        cf_instance = instance.copy().values.astype(float)
        
        # Indices des features numériques et catégorielles
        num_indices = [self.feature_names.index(f) for f in self.numerical_features if f in self.feature_names]
        cat_indices = [self.feature_names.index(f) for f in self.categorical_features if f in self.feature_names]
        
        # Générer l'explication contrefactuelle
        for iteration in range(max_iterations):
            # Calculer la prédiction actuelle
            current_pred = self.model.predict_proba(cf_instance.reshape(1, -1))[0]
            current_class = self.model.predict(cf_instance.reshape(1, -1))[0]
            
            # Vérifier si on a atteint la classe cible
            if current_class == target_class:
                logger.info(f"Classe cible atteinte à l'itération {iteration}")
                break
            
            # Calculer le gradient (approximation)
            gradient = self._compute_gradient(cf_instance, target_class)
            
            # Mettre à jour l'instance contrefactuelle
            for idx in num_indices:
                cf_instance[idx] -= learning_rate * gradient[idx]
            
            # Pour les features catégorielles, on ne peut pas utiliser le gradient directement
            # On utilise une approche basée sur la distance
            for idx in cat_indices:
                # Essayer différentes valeurs catégorielles
                best_value = cf_instance[idx]
                best_score = float('inf')
                
                for possible_value in range(int(cf_instance[idx]) - 2, int(cf_instance[idx]) + 3):
                    if possible_value >= 0:  # Valeur valide
                        test_instance = cf_instance.copy()
                        test_instance[idx] = possible_value
                        test_pred = self.model.predict_proba(test_instance.reshape(1, -1))[0]
                        
                        # Calculer le score (distance + pénalité de prédiction)
                        score = (
                            lambda_param * np.sum(np.abs(test_instance - instance.values)) +
                            (1 - lambda_param) * abs(test_pred - target_class)
                        )
                        
                        if score < best_score:
                            best_score = score
                            best_value = possible_value
                
                cf_instance[idx] = best_value
        
        # Prédiction finale
        final_pred = self.model.predict_proba(cf_instance.reshape(1, -1))[0]
        final_class = self.model.predict(cf_instance.reshape(1, -1))[0]
        
        # Calculer les changements
        changes = self._compute_changes(instance, cf_instance)
        
        # Créer le résultat
        result = {
            'success': final_class == target_class,
            'original_instance': instance.to_dict(),
            'counterfactual_instance': dict(zip(self.feature_names, cf_instance)),
            'changes': changes,
            'original_prediction': float(initial_pred),
            'counterfactual_prediction': float(final_pred),
            'iterations': iteration + 1,
            'distance': float(np.sum(np.abs(cf_instance - instance.values)))
        }
        
        if result['success']:
            logger.info(f"Explication contrefactuelle générée avec succès")
        else:
            logger.warning(f"Impossible de générer une explication contrefactuelle après {max_iterations} itérations")
        
        return result
    
    def _compute_gradient(self, instance: np.ndarray, target_class: int) -> np.ndarray:
        """
        Calcule le gradient de la fonction de perte.
        
        Args:
            instance: Instance courante
            target_class: Classe cible
            
        Returns:
            np.ndarray: Gradient
        """
        epsilon = 1e-5
        gradient = np.zeros_like(instance)
        
        # Prédiction actuelle
        current_pred = self.model.predict_proba(instance.reshape(1, -1))[0]
        
        # Calculer le gradient par différences finies
        for i in range(len(instance)):
            instance_plus = instance.copy()
            instance_plus[i] += epsilon
            
            pred_plus = self.model.predict_proba(instance_plus.reshape(1, -1))[0]
            
            # Gradient de la perte (cross-entropy)
            gradient[i] = (pred_plus - current_pred) / epsilon
        
        return gradient
    
    def _compute_changes(
        self,
        original: pd.Series,
        counterfactual: np.ndarray
    ) -> List[Dict]:
        """
        Calcule les changements entre l'instance originale et contrefactuelle.
        
        Args:
            original: Instance originale
            counterfactual: Instance contrefactuelle
            
        Returns:
            List[Dict]: Liste des changements
        """
        changes = []
        
        for i, feature_name in enumerate(self.feature_names):
            original_value = original.iloc[i]
            cf_value = counterfactual[i]
            
            if abs(original_value - cf_value) > 0.01:  # Tolérance pour les valeurs numériques
                change_type = 'numerical' if feature_name in self.numerical_features else 'categorical'
                
                changes.append({
                    'feature': feature_name,
                    'original_value': float(original_value),
                    'counterfactual_value': float(cf_value),
                    'change': float(cf_value - original_value),
                    'change_type': change_type,
                    'direction': 'increase' if cf_value > original_value else 'decrease'
                })
        
        # Trier par importance absolue du changement
        changes.sort(key=lambda x: abs(x['change']), reverse=True)
        
        return changes
    
    def generate_multiple_counterfactuals(
        self,
        instance: pd.Series,
        target_class: int = 1,
        num_counterfactuals: int = 5,
        **kwargs
    ) -> List[Dict]:
        """
        Génère plusieurs explications contrefactuelles pour une instance.
        
        Args:
            instance: Instance à expliquer
            target_class: Classe cible
            num_counterfactuals: Nombre d'explications à générer
            **kwargs: Arguments supplémentaires pour generate_counterfactual
            
        Returns:
            List[Dict]: Liste des explications contrefactuelles
        """
        logger.info(f"Génération de {num_counterfactuals} explications contrefactuelles...")
        
        counterfactuals = []
        
        for i in range(num_counterfactuals):
            # Varier les paramètres pour obtenir des explications différentes
            cf = self.generate_counterfactual(
                instance,
                target_class=target_class,
                learning_rate=kwargs.get('learning_rate', 0.1) * (0.8 + 0.4 * np.random.random()),
                lambda_param=kwargs.get('lambda_param', 0.1) * (0.8 + 0.4 * np.random.random()),
                **{k: v for k, v in kwargs.items() if k not in ['learning_rate', 'lambda_param']}
            )
            
            if cf['success']:
                counterfactuals.append(cf)
        
        # Dédupliquer les explications similaires
        unique_counterfactuals = self._deduplicate_counterfactuals(counterfactuals)
        
        logger.info(f"{len(unique_counterfactuals)} explications contrefactuelles uniques générées")
        
        return unique_counterfactuals[:num_counterfactuals]
    
    def _deduplicate_counterfactuals(
        self,
        counterfactuals: List[Dict],
        similarity_threshold: float = 0.9
    ) -> List[Dict]:
        """
        Déduplique les explications contrefactuelles similaires.
        
        Args:
            counterfactuals: Liste des explications contrefactuelles
            similarity_threshold: Seuil de similarité
            
        Returns:
            List[Dict]: Liste des explications uniques
        """
        if not counterfactuals:
            return []
        
        unique = [counterfactuals[0]]
        
        for cf in counterfactuals[1:]:
            is_similar = False
            
            for existing_cf in unique:
                similarity = self._compute_similarity(cf, existing_cf)
                if similarity > similarity_threshold:
                    is_similar = True
                    break
            
            if not is_similar:
                unique.append(cf)
        
        return unique
    
    def _compute_similarity(self, cf1: Dict, cf2: Dict) -> float:
        """
        Calcule la similarité entre deux explications contrefactuelles.
        
        Args:
            cf1: Première explication contrefactuelle
            cf2: Deuxième explication contrefactuelle
            
        Returns:
            float: Score de similarité (0-1)
        """
        # Extraire les vecteurs de changements
        changes1 = {c['feature']: c['change'] for c in cf1['changes']}
        changes2 = {c['feature']: c['change'] for c in cf2['changes']}
        
        # Calculer la similarité cosinus
        all_features = set(changes1.keys()) | set(changes2.keys())
        
        if not all_features:
            return 1.0
        
        vec1 = np.array([changes1.get(f, 0) for f in all_features])
        vec2 = np.array([changes2.get(f, 0) for f in all_features])
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        
        return float(abs(similarity))
    
    def explain_counterfactual(self, counterfactual: Dict) -> str:
        """
        Génère une explication textuelle d'une explication contrefactuelle.
        
        Args:
            counterfactual: Explication contrefactuelle
            
        Returns:
            str: Explication textuelle
        """
        if not counterfactual['success']:
            return "Impossible de générer une explication contrefactuelle."
        
        explanation = []
        
        # Introduction
        original_pred = counterfactual['original_prediction']
        cf_pred = counterfactual['counterfactual_prediction']
        
        if original_pred < 0.5 and cf_pred >= 0.5:
            explanation.append("Pour améliorer votre score de crédit et augmenter vos chances d'approbation :")
        elif original_pred >= 0.5 and cf_pred < 0.5:
            explanation.append("Voici ce qui pourrait faire baisser votre score de crédit :")
        else:
            explanation.append("Voici les changements suggérés :")
        
        # Changements
        for change in counterfactual['changes'][:5]:  # Top 5 changements
            feature = change['feature']
            direction = change['direction']
            
            if change['change_type'] == 'numerical':
                if direction == 'increase':
                    explanation.append(f"- Augmenter {feature}")
                else:
                    explanation.append(f"- Réduire {feature}")
            else:
                explanation.append(f"- Modifier {feature}")
        
        # Conclusion
        explanation.append(f"\nCes changements augmenteraient votre probabilité d'approbation de {original_pred:.1%} à {cf_pred:.1%}.")
        
        return "\n".join(explanation)


if __name__ == "__main__":
    # Test du module
    print("Test du module CounterfactualExplainer...")
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
    
    # Créer l'explainer contrefactuel
    explainer = CounterfactualExplainer(model, feature_names=preprocessor.feature_names)
    
    # Trouver une instance rejetée
    rejected_indices = np.where(model.predict(X_test) == 0)[0]
    if len(rejected_indices) > 0:
        instance_idx = rejected_indices[0]
        instance = X_test.iloc[instance_idx]
        
        print(f"\nInstance {instance_idx} (rejetée):")
        print(f"Prédiction: {model.predict_proba(instance.values.reshape(1, -1))[0]:.4f}")
        
        # Générer une explication contrefactuelle
        cf = explainer.generate_counterfactual(instance, target_class=1)
        
        print(f"\nExplication contrefactuelle:")
        print(f"Succès: {cf['success']}")
        print(f"Prédiction originale: {cf['original_prediction']:.4f}")
        print(f"Prédiction contrefactuelle: {cf['counterfactual_prediction']:.4f}")
        print(f"Distance: {cf['distance']:.4f}")
        print(f"\nChangements:")
        for change in cf['changes'][:5]:
            print(f"  {change['feature']}: {change['original_value']:.2f} -> {change['counterfactual_value']:.2f}")
        
        # Explication textuelle
        print(f"\nExplication textuelle:")
        print(explainer.explain_counterfactual(cf))
    else:
        print("Aucune instance rejetée trouvée dans le test set")