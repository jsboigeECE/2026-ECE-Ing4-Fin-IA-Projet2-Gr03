"""
Modèle baseline (Logistic Regression) pour le Credit Scoring XAI.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, Tuple, Optional
import joblib
import logging

from src.config import RANDOM_STATE, MODELS_DIR

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineModel:
    """
    Modèle baseline utilisant Logistic Regression.
    """
    
    def __init__(self, random_state: int = RANDOM_STATE):
        """
        Initialise le modèle baseline.
        
        Args:
            random_state: Seed pour la reproductibilité
        """
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        self.is_fitted = False
        self.feature_names: Optional[list] = None
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Entraîne le modèle baseline.
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            X_val: Features de validation (optionnel)
            y_val: Target de validation (optionnel)
            
        Returns:
            Dict[str, float]: Métriques d'entraînement
        """
        logger.info("Entraînement du modèle baseline (Logistic Regression)...")
        
        # Stocker les noms des features
        self.feature_names = X_train.columns.tolist()
        
        # Entraîner le modèle
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Prédictions sur l'entraînement
        y_train_pred = self.model.predict(X_train)
        y_train_proba = self.model.predict_proba(X_train)[:, 1]
        
        # Calculer les métriques d'entraînement
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'train_precision': precision_score(y_train, y_train_pred),
            'train_recall': recall_score(y_train, y_train_pred),
            'train_f1': f1_score(y_train, y_train_pred),
            'train_roc_auc': roc_auc_score(y_train, y_train_proba)
        }
        
        # Évaluer sur validation si fourni
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            y_val_proba = self.model.predict_proba(X_val)[:, 1]
            
            metrics.update({
                'val_accuracy': accuracy_score(y_val, y_val_pred),
                'val_precision': precision_score(y_val, y_val_pred),
                'val_recall': recall_score(y_val, y_val_pred),
                'val_f1': f1_score(y_val, y_val_pred),
                'val_roc_auc': roc_auc_score(y_val, y_val_proba)
            })
            
            logger.info(f"Validation ROC-AUC: {metrics['val_roc_auc']:.4f}")
        
        logger.info(f"Entraînement terminé. Train ROC-AUC: {metrics['train_roc_auc']:.4f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fait des prédictions binaires.
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Prédictions binaires
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fait des prédictions de probabilités.
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Probabilités pour la classe positive
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Évalue le modèle sur un dataset.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Dict[str, float]: Métriques d'évaluation
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_proba)
        }
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Retourne l'importance des features (coefficients absolus).
        
        Returns:
            pd.DataFrame: DataFrame avec importance des features
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné")
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_[0],
            'abs_coefficient': np.abs(self.model.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)
        
        return importance
    
    def get_confusion_matrix(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        Retourne la matrice de confusion.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            np.ndarray: Matrice de confusion
        """
        y_pred = self.predict(X)
        return confusion_matrix(y, y_pred)
    
    def save(self, filepath: str) -> None:
        """
        Sauvegarde le modèle.
        
        Args:
            filepath: Chemin de sauvegarde
        """
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }, filepath)
        logger.info(f"Modèle sauvegardé dans {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Charge le modèle.
        
        Args:
            filepath: Chemin du fichier à charger
        """
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.is_fitted = data['is_fitted']
        logger.info(f"Modèle chargé depuis {filepath}")


if __name__ == "__main__":
    # Test du module
    print("Test du module BaselineModel...")
    from src.data_loader import load_data
    from src.preprocessing import DataPreprocessor, split_data
    
    # Charger et préparer les données
    X, y = load_data()
    preprocessor = DataPreprocessor()
    X_transformed = preprocessor.fit_transform(X)
    
    # Split des données
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_transformed, y)
    
    # Créer et entraîner le modèle
    model = BaselineModel()
    metrics = model.train(X_train, y_train, X_val, y_val)
    
    print(f"\nMétriques d'entraînement:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Évaluer sur le test
    test_metrics = model.evaluate(X_test, y_test)
    print(f"\nMétriques de test:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Importance des features
    importance = model.get_feature_importance()
    print(f"\nTop 10 features les plus importantes:")
    print(importance.head(10))
    
    # Sauvegarder le modèle
    model.save(str(MODELS_DIR / "baseline_model.pkl"))