"""
Modèle XGBoost pour le Credit Scoring XAI.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from typing import Dict, Tuple, Optional, List
import joblib
import logging

from src.config import RANDOM_STATE, MODELS_DIR, XGBOOST_PARAMS

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    Modèle XGBoost pour le scoring de crédit.
    """
    
    def __init__(
        self,
        params: Optional[Dict] = None,
        random_state: int = RANDOM_STATE
    ):
        """
        Initialise le modèle XGBoost.
        
        Args:
            params: Paramètres du modèle XGBoost
            random_state: Seed pour la reproductibilité
        """
        if params is None:
            params = XGBOOST_PARAMS.copy()
        
        self.params = params
        self.params['random_state'] = random_state
        self.model = xgb.XGBClassifier(**self.params)
        self.is_fitted = False
        self.feature_names: Optional[list] = None
        self.best_params: Optional[Dict] = None
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 10
    ) -> Dict[str, float]:
        """
        Entraîne le modèle XGBoost.
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            X_val: Features de validation (optionnel)
            y_val: Target de validation (optionnel)
            early_stopping_rounds: Nombre de rounds pour early stopping
            
        Returns:
            Dict[str, float]: Métriques d'entraînement
        """
        logger.info("Entraînement du modèle XGBoost...")
        
        # Stocker les noms des features
        self.feature_names = X_train.columns.tolist()
        
        # Entraîner avec ou sans validation set
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
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
    
    def hyperparameter_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        method: str = 'grid',
        n_iter: int = 50,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Optimise les hyperparamètres du modèle.
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            X_val: Features de validation (optionnel)
            y_val: Target de validation (optionnel)
            method: Méthode d'optimisation ('grid' ou 'random')
            n_iter: Nombre d'itérations pour RandomizedSearchCV
            cv: Nombre de folds pour la validation croisée
            
        Returns:
            Dict[str, float]: Métriques avec les meilleurs paramètres
        """
        logger.info(f"Optimisation des hyperparamètres avec {method} search...")
        
        # Définir la grille de paramètres
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 4, 5, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2]
        }
        
        # Créer le modèle de base
        base_model = xgb.XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # Choisir la méthode d'optimisation
        if method == 'grid':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
        else:  # random
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1,
                random_state=RANDOM_STATE
            )
        
        # Effectuer la recherche
        search.fit(X_train, y_train)
        
        # Mettre à jour le modèle avec les meilleurs paramètres
        self.best_params = search.best_params_
        self.params = {**self.params, **self.best_params}
        self.model = xgb.XGBClassifier(**self.params)
        
        logger.info(f"Meilleurs paramètres: {self.best_params}")
        logger.info(f"Meilleur score CV: {search.best_score_:.4f}")
        
        # Réentraîner avec les meilleurs paramètres
        return self.train(X_train, y_train, X_val, y_val)
    
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
    
    def get_feature_importance(self, importance_type: str = 'weight') -> pd.DataFrame:
        """
        Retourne l'importance des features.
        
        Args:
            importance_type: Type d'importance ('weight', 'gain', 'cover', 'total_gain', 'total_cover')
            
        Returns:
            pd.DataFrame: DataFrame avec importance des features
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné")
        
        score_dict = self.model.get_booster().get_score(importance_type=importance_type)
        importance = pd.DataFrame([
            {'feature': k, 'importance': v} for k, v in score_dict.items()
        ])
        
        # Ajouter les features manquantes avec importance 0
        all_features = set(self.feature_names)
        scored_features = set(importance['feature'])
        missing_features = all_features - scored_features
        
        for feature in missing_features:
            importance = pd.concat([
                importance,
                pd.DataFrame({'feature': [feature], 'importance': [0]})
            ], ignore_index=True)
        
        importance = importance.sort_values('importance', ascending=False)
        
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
            'params': self.params,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'best_params': self.best_params
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
        self.params = data['params']
        self.feature_names = data['feature_names']
        self.is_fitted = data['is_fitted']
        self.best_params = data.get('best_params')
        logger.info(f"Modèle chargé depuis {filepath}")


if __name__ == "__main__":
    # Test du module
    print("Test du module XGBoostModel...")
    from src.data_loader import load_data
    from src.preprocessing import DataPreprocessor, split_data
    
    # Charger et préparer les données
    X, y = load_data()
    preprocessor = DataPreprocessor()
    X_transformed = preprocessor.fit_transform(X)
    
    # Split des données
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_transformed, y)
    
    # Créer et entraîner le modèle
    model = XGBoostModel()
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
    importance = model.get_feature_importance(importance_type='gain')
    print(f"\nTop 10 features les plus importantes:")
    print(importance.head(10))
    
    # Sauvegarder le modèle
    model.save(str(MODELS_DIR / "xgboost_model.pkl"))