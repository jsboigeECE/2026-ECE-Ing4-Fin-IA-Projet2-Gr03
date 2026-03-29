"""
Module de prétraitement des données pour le Credit Scoring XAI.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, Dict, List, Optional
import joblib
import logging

from src.config import (
    NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS,
    RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE,
    MODELS_DIR
)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Classe pour prétraiter les données du German Credit Dataset.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.column_transformer: Optional[ColumnTransformer] = None
        self.feature_names: List[str] = []
        
    def fit_label_encoders(self, X: pd.DataFrame) -> None:
        """
        Ajuste les label encoders pour les colonnes catégorielles.
        
        Args:
            X: DataFrame avec features catégorielles
        """
        for col in CATEGORICAL_COLUMNS:
            if col in X.columns:
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"Label encoder ajusté pour {col}: {len(le.classes_)} classes")
    
    def transform_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforme les colonnes catégorielles en valeurs numériques.
        
        Args:
            X: DataFrame avec features catégorielles
            
        Returns:
            pd.DataFrame: DataFrame avec features catégorielles encodées
        """
        X_transformed = X.copy()
        
        for col, le in self.label_encoders.items():
            if col in X_transformed.columns:
                X_transformed[col] = le.transform(X_transformed[col].astype(str))
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Ajuste et transforme les données.
        
        Args:
            X: DataFrame avec features
            
        Returns:
            pd.DataFrame: DataFrame transformé
        """
        # Ajuster les label encoders
        self.fit_label_encoders(X)
        
        # Transformer les colonnes catégorielles
        X_encoded = self.transform_categorical(X)
        
        # Sélectionner uniquement les colonnes numériques et catégorielles encodées
        all_columns = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS
        X_selected = X_encoded[all_columns].copy()
        
        # Standardiser les colonnes numériques
        X_selected[NUMERICAL_COLUMNS] = self.scaler.fit_transform(
            X_selected[NUMERICAL_COLUMNS]
        )
        
        self.feature_names = all_columns
        logger.info(f"Données transformées: {X_selected.shape}")
        
        return X_selected
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforme les données avec les paramètres déjà ajustés.
        
        Args:
            X: DataFrame avec features
            
        Returns:
            pd.DataFrame: DataFrame transformé
        """
        # Transformer les colonnes catégorielles
        X_encoded = self.transform_categorical(X)
        
        # Sélectionner uniquement les colonnes numériques et catégorielles encodées
        all_columns = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS
        X_selected = X_encoded[all_columns].copy()
        
        # Standardiser les colonnes numériques
        X_selected[NUMERICAL_COLUMNS] = self.scaler.transform(
            X_selected[NUMERICAL_COLUMNS]
        )
        
        return X_selected
    
    def save(self, filepath: str) -> None:
        """
        Sauvegarde le préprocesseur.
        
        Args:
            filepath: Chemin de sauvegarde
        """
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }, filepath)
        logger.info(f"Préprocesseur sauvegardé dans {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Charge le préprocesseur.
        
        Args:
            filepath: Chemin du fichier à charger
        """
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_names = data['feature_names']
        logger.info(f"Préprocesseur chargé depuis {filepath}")


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    validation_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Divise les données en ensembles d'entraînement, de validation et de test.
    
    Args:
        X: Features
        y: Target
        test_size: Proportion de l'ensemble de test
        validation_size: Proportion de l'ensemble de validation (par rapport à train)
        random_state: Seed pour la reproductibilité
        
    Returns:
        Tuple contenant X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Premier split: train + val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Deuxième split: train vs val
    val_size_adjusted = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted,
        random_state=random_state, stratify=y_train_val
    )
    
    logger.info(f"Split des données:")
    logger.info(f"  Train: {X_train.shape[0]} échantillons")
    logger.info(f"  Validation: {X_val.shape[0]} échantillons")
    logger.info(f"  Test: {X_test.shape[0]} échantillons")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_preprocessing_pipeline() -> ColumnTransformer:
    """
    Crée un pipeline de prétraitement pour sklearn.
    
    Returns:
        ColumnTransformer: Pipeline de prétraitement
    """
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('label_encoder', 'passthrough')  # Sera géré manuellement
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_COLUMNS),
            ('cat', categorical_transformer, CATEGORICAL_COLUMNS)
        ]
    )
    
    return preprocessor


def get_feature_importance_data(
    X: pd.DataFrame,
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Prépare les données pour l'analyse de l'importance des features.
    
    Args:
        X: Features transformées
        feature_names: Noms des features (si None, utilise les colonnes de X)
        
    Returns:
        pd.DataFrame: DataFrame avec noms de features
    """
    if feature_names is None:
        feature_names = X.columns.tolist()
    
    return pd.DataFrame(X, columns=feature_names)


if __name__ == "__main__":
    # Test du module
    print("Test du module Preprocessing...")
    from src.data_loader import load_data
    
    # Charger les données
    X, y = load_data()
    
    # Créer le préprocesseur
    preprocessor = DataPreprocessor()
    
    # Transformer les données
    X_transformed = preprocessor.fit_transform(X)
    
    print(f"\nShape avant transformation: {X.shape}")
    print(f"Shape après transformation: {X_transformed.shape}")
    print(f"\nTypes de données après transformation:")
    print(X_transformed.dtypes)
    
    # Split des données
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_transformed, y)
    
    print(f"\nDistribution de la cible:")
    print(f"Train: {y_train.value_counts().to_dict()}")
    print(f"Val: {y_val.value_counts().to_dict()}")
    print(f"Test: {y_test.value_counts().to_dict()}")
    
    # Sauvegarder le préprocesseur
    preprocessor.save(str(MODELS_DIR / "preprocessor.pkl"))