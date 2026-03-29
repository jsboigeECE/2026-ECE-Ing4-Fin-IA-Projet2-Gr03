"""
Module d'évaluation des modèles pour le Credit Scoring XAI.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Classe pour évaluer les modèles de scoring de crédit.
    """
    
    def __init__(self):
        self.results = {}
        
    def evaluate_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "model"
    ) -> Dict:
        """
        Évalue un modèle sur un dataset de test.
        
        Args:
            model: Modèle à évaluer
            X_test: Features de test
            y_test: Target de test
            model_name: Nom du modèle
            
        Returns:
            Dict: Métriques d'évaluation
        """
        logger.info(f"Évaluation du modèle {model_name}...")
        
        # Prédictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Calculer les métriques
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'average_precision': average_precision_score(y_test, y_proba)
        }
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Courbe ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr}
        
        # Courbe Precision-Recall
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        metrics['pr_curve'] = {'precision': precision, 'recall': recall}
        
        # Stocker les résultats
        self.results[model_name] = metrics
        
        logger.info(f"Évaluation terminée. ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def compare_models(
        self,
        models: Dict[str, object],
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Compare plusieurs modèles.
        
        Args:
            models: Dictionnaire des modèles {nom: modèle}
            X_test: Features de test
            y_test: Target de test
            
        Returns:
            pd.DataFrame: DataFrame de comparaison
        """
        logger.info(f"Comparaison de {len(models)} modèles...")
        
        comparison = []
        
        for model_name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            comparison.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'ROC-AUC': metrics['roc_auc'],
                'Avg Precision': metrics['average_precision']
            })
        
        df_comparison = pd.DataFrame(comparison)
        df_comparison = df_comparison.sort_values('ROC-AUC', ascending=False)
        
        logger.info("Comparaison terminée")
        
        return df_comparison
    
    def plot_confusion_matrix(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model_name: str = "Model",
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Génère un plot de la matrice de confusion.
        
        Args:
            y_true: Vraies valeurs
            y_pred: Prédictions
            model_name: Nom du modèle
            figsize: Taille de la figure
            
        Returns:
            plt.Figure: Figure matplotlib
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax,
            xticklabels=['Bad Credit', 'Good Credit'],
            yticklabels=['Bad Credit', 'Good Credit']
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {model_name}')
        
        return fig
    
    def plot_roc_curve(
        self,
        y_true: pd.Series,
        y_proba: np.ndarray,
        model_name: str = "Model",
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Génère un plot de la courbe ROC.
        
        Args:
            y_true: Vraies valeurs
            y_proba: Probabilités prédites
            model_name: Nom du modèle
            figsize: Taille de la figure
            
        Returns:
            plt.Figure: Figure matplotlib
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_precision_recall_curve(
        self,
        y_true: pd.Series,
        y_proba: np.ndarray,
        model_name: str = "Model",
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Génère un plot de la courbe Precision-Recall.
        
        Args:
            y_true: Vraies valeurs
            y_proba: Probabilités prédites
            model_name: Nom du modèle
            figsize: Taille de la figure
            
        Returns:
            plt.Figure: Figure matplotlib
        """
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.4f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_comparison_roc(
        self,
        models: Dict[str, object],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Génère un plot comparatif des courbes ROC de plusieurs modèles.
        
        Args:
            models: Dictionnaire des modèles {nom: modèle}
            X_test: Features de test
            y_test: Target de test
            figsize: Taille de la figure
            
        Returns:
            plt.Figure: Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for model_name, model in models.items():
            y_proba = model.predict_proba(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def generate_report(self, model_name: Optional[str] = None) -> str:
        """
        Génère un rapport textuel d'évaluation.
        
        Args:
            model_name: Nom du modèle (si None, génère pour tous les modèles)
            
        Returns:
            str: Rapport textuel
        """
        if not self.results:
            return "Aucun résultat d'évaluation disponible."
        
        report = []
        report.append("=" * 60)
        report.append("RAPPORT D'ÉVALUATION DES MODÈLES")
        report.append("=" * 60)
        report.append("")
        
        if model_name:
            # Rapport pour un modèle spécifique
            if model_name not in self.results:
                return f"Modèle {model_name} non trouvé."
            
            metrics = self.results[model_name]
            report.append(f"MODÈLE: {model_name}")
            report.append("-" * 40)
            report.append(f"Accuracy: {metrics['accuracy']:.4f}")
            report.append(f"Precision: {metrics['precision']:.4f}")
            report.append(f"Recall: {metrics['recall']:.4f}")
            report.append(f"F1-Score: {metrics['f1']:.4f}")
            report.append(f"ROC-AUC: {metrics['roc_auc']:.4f}")
            report.append(f"Average Precision: {metrics['average_precision']:.4f}")
        else:
            # Rapport pour tous les modèles
            for model_name, metrics in self.results.items():
                report.append(f"MODÈLE: {model_name}")
                report.append("-" * 40)
                report.append(f"Accuracy: {metrics['accuracy']:.4f}")
                report.append(f"Precision: {metrics['precision']:.4f}")
                report.append(f"Recall: {metrics['recall']:.4f}")
                report.append(f"F1-Score: {metrics['f1']:.4f}")
                report.append(f"ROC-AUC: {metrics['roc_auc']:.4f}")
                report.append(f"Average Precision: {metrics['average_precision']:.4f}")
                report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def get_best_model(self, metric: str = 'roc_auc') -> Tuple[str, Dict]:
        """
        Retourne le meilleur modèle selon une métrique.
        
        Args:
            metric: Métrique à utiliser ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')
            
        Returns:
            Tuple[str, Dict]: Nom du modèle et ses métriques
        """
        if not self.results:
            raise ValueError("Aucun résultat d'évaluation disponible.")
        
        best_model = max(self.results.items(), key=lambda x: x[1][metric])
        
        return best_model


if __name__ == "__main__":
    # Test du module
    print("Test du module ModelEvaluator...")
    from src.data_loader import load_data
    from src.preprocessing import DataPreprocessor, split_data
    from src.models.baseline_model import BaselineModel
    from src.models.xgboost_model import XGBoostModel
    from src.models.lightgbm_model import LightGBMModel
    
    # Charger et préparer les données
    X, y = load_data()
    preprocessor = DataPreprocessor()
    X_transformed = preprocessor.fit_transform(X)
    
    # Split des données
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_transformed, y)
    
    # Créer et entraîner les modèles
    models = {}
    
    baseline = BaselineModel()
    baseline.train(X_train, y_train, X_val, y_val)
    models['Baseline (Logistic Regression)'] = baseline
    
    xgb = XGBoostModel()
    xgb.train(X_train, y_train, X_val, y_val)
    models['XGBoost'] = xgb
    
    lgb = LightGBMModel()
    lgb.train(X_train, y_train, X_val, y_val)
    models['LightGBM'] = lgb
    
    # Évaluer les modèles
    evaluator = ModelEvaluator()
    comparison = evaluator.compare_models(models, X_test, y_test)
    
    print(f"\nComparaison des modèles:")
    print(comparison.to_string(index=False))
    
    # Meilleur modèle
    best_name, best_metrics = evaluator.get_best_model('roc_auc')
    print(f"\nMeilleur modèle: {best_name} (ROC-AUC: {best_metrics['roc_auc']:.4f})")
    
    # Rapport
    print(f"\n{evaluator.generate_report()}")