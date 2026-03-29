"""
Module de chargement et de préparation des données pour le Credit Scoring XAI.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
from typing import Tuple, Optional
import logging

from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASET_URL,
    GERMAN_CREDIT_COLUMNS, CATEGORICAL_MAPPINGS,
    NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, RANDOM_STATE
)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Classe pour charger et préparer les données du German Credit Dataset.
    """
    
    def __init__(self):
        self.raw_data_path = RAW_DATA_DIR / "german_credit.csv"
        self.processed_data_path = PROCESSED_DATA_DIR / "german_credit_processed.csv"
        
    def download_dataset(self) -> pd.DataFrame:
        """
        Télécharge le German Credit Dataset depuis l'UCI ML Repository.
        
        Returns:
            pd.DataFrame: Dataset brut
        """
        logger.info("Téléchargement du German Credit Dataset...")
        
        try:
            # Télécharger les données
            response = requests.get(DATASET_URL)
            response.raise_for_status()
            
            # Parser les données (format avec espaces comme séparateur)
            from io import StringIO
            data = StringIO(response.text)
            df = pd.read_csv(data, sep=' ', header=None, names=GERMAN_CREDIT_COLUMNS)
            
            # Sauvegarder les données brutes
            df.to_csv(self.raw_data_path, index=False)
            logger.info(f"Dataset téléchargé et sauvegardé dans {self.raw_data_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement du dataset: {e}")
            raise
    
    def load_data(self, force_download: bool = False) -> pd.DataFrame:
        """
        Charge le dataset (télécharge si nécessaire).
        
        Args:
            force_download: Force le téléchargement même si le fichier existe
            
        Returns:
            pd.DataFrame: Dataset chargé
        """
        if not self.raw_data_path.exists() or force_download:
            df = self.download_dataset()
        else:
            logger.info(f"Chargement du dataset depuis {self.raw_data_path}")
            df = pd.read_csv(self.raw_data_path)
        
        return df
    
    def map_categorical_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mappe les valeurs catégorielles codes vers des descriptions lisibles.
        
        Args:
            df: DataFrame avec valeurs catégorielles codées
            
        Returns:
            pd.DataFrame: DataFrame avec valeurs catégorielles mappées
        """
        df_mapped = df.copy()
        
        for column, mapping in CATEGORICAL_MAPPINGS.items():
            if column in df_mapped.columns:
                df_mapped[column] = df_mapped[column].map(mapping)
        
        return df_mapped
    
    def extract_gender(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrait le genre depuis la colonne personal_status.
        
        Args:
            df: DataFrame avec colonne personal_status
            
        Returns:
            pd.DataFrame: DataFrame avec colonne gender ajoutée
        """
        df = df.copy()
        
        def extract_gender_from_status(status):
            if pd.isna(status):
                return 'unknown'
            status_str = str(status)
            if 'female' in status_str:
                return 'female'
            elif 'male' in status_str:
                return 'male'
            else:
                return 'unknown'
        
        df['gender'] = df['personal_status'].apply(extract_gender_from_status)
        
        return df
    
    def create_age_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée des groupes d'âge pour l'analyse de fairness.
        
        Args:
            df: DataFrame avec colonne age
            
        Returns:
            pd.DataFrame: DataFrame avec colonne age_group ajoutée
        """
        df = df.copy()
        
        def categorize_age(age):
            if age < 25:
                return 'young'
            elif age < 40:
                return 'middle'
            elif age < 60:
                return 'senior'
            else:
                return 'elderly'
        
        df['age_group'] = df['age'].apply(categorize_age)
        
        return df
    
    def preprocess_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prétraite la variable cible (1 = Good, 2 = Bad -> 0 = Bad, 1 = Good).
        
        Args:
            df: DataFrame avec colonne credit_risk
            
        Returns:
            pd.DataFrame: DataFrame avec cible transformée
        """
        df = df.copy()
        # Transformer: 1 (Good) -> 1, 2 (Bad) -> 0
        df['credit_risk'] = df['credit_risk'].map({1: 1, 2: 0})
        
        return df
    
    def prepare_data(self, df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prépare les données pour l'entraînement des modèles.
        
        Args:
            df: DataFrame à préparer (si None, charge depuis le fichier)
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features (X) et target (y)
        """
        if df is None:
            df = self.load_data()
        
        # Mapper les valeurs catégorielles
        df = self.map_categorical_values(df)
        
        # Extraire le genre
        df = self.extract_gender(df)
        
        # Créer des groupes d'âge
        df = self.create_age_groups(df)
        
        # Prétraiter la cible
        df = self.preprocess_target(df)
        
        # Séparer features et target
        X = df.drop('credit_risk', axis=1)
        y = df['credit_risk']
        
        logger.info(f"Données préparées: {X.shape[0]} échantillons, {X.shape[1]} features")
        
        return X, y
    
    def save_processed_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Sauvegarde les données traitées.
        
        Args:
            X: Features
            y: Target
        """
        df_processed = X.copy()
        df_processed['credit_risk'] = y
        df_processed.to_csv(self.processed_data_path, index=False)
        logger.info(f"Données traitées sauvegardées dans {self.processed_data_path}")
    
    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Charge les données traitées.
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features (X) et target (y)
        """
        df = pd.read_csv(self.processed_data_path)
        X = df.drop('credit_risk', axis=1)
        y = df['credit_risk']
        
        logger.info(f"Données traitées chargées: {X.shape[0]} échantillons, {X.shape[1]} features")
        
        return X, y


def load_data(force_download: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Fonction utilitaire pour charger et préparer les données.
    
    Args:
        force_download: Force le téléchargement
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) et target (y)
    """
    loader = DataLoader()
    
    # Charger les données brutes
    df = loader.load_data(force_download=force_download)
    
    # Préparer les données
    X, y = loader.prepare_data(df)
    
    # Sauvegarder les données traitées
    loader.save_processed_data(X, y)
    
    return X, y


if __name__ == "__main__":
    # Test du module
    print("Test du module DataLoader...")
    X, y = load_data()
    print(f"\nShape de X: {X.shape}")
    print(f"Shape de y: {y.shape}")
    print(f"\nDistribution de la cible:")
    print(y.value_counts())
    print(f"\nTypes de données:")
    print(X.dtypes)
    print(f"\nPremières lignes:")
    print(X.head())