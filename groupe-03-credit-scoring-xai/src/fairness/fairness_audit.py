"""
Module d'audit de fairness pour le Credit Scoring XAI utilisant Fairlearn.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    selection_rate,
    count
)
from fairlearn.reductions import ExponentiatedGradient, GridSearch, DemographicParity, EqualizedOdds
from typing import Dict, List, Optional, Tuple, Union
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FairnessAuditor:
    """
    Classe pour auditer le fairness des modèles de scoring de crédit.
    """
    
    def __init__(self, model):
        """
        Initialise l'auditeur de fairness.
        
        Args:
            model: Modèle à auditer (doit avoir des méthodes predict et predict_proba)
        """
        self.model = model
        self.audit_results = {}
        
    def compute_metrics_by_group(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_features: pd.DataFrame,
        metrics: Optional[Dict] = None
    ) -> MetricFrame:
        """
        Calcule les métriques par groupe sensible.
        
        Args:
            X: Features
            y: Target
            sensitive_features: Features sensibles (ex: gender, age)
            metrics: Dictionnaire de métriques à calculer
            
        Returns:
            MetricFrame: Métriques par groupe
        """
        if metrics is None:
            metrics = {
                'accuracy': accuracy_score,
                'precision': precision_score,
                'recall': recall_score,
                'f1': f1_score,
                'selection_rate': selection_rate,
                'count': count
            }
        
        # Prédictions
        y_pred = self.model.predict(X)
        
        # Calculer les métriques par groupe
        metric_frame = MetricFrame(
            metrics=metrics,
            y_true=y,
            y_pred=y_pred,
            sensitive_features=sensitive_features
        )
        
        logger.info(f"Métriques calculées pour {len(sensitive_features.columns)} features sensibles")
        
        return metric_frame
    
    def audit_demographic_parity(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_feature: pd.Series
    ) -> Dict:
        """
        Audite la parité démographique (Demographic Parity).
        
        La parité démographique exige que le taux de sélection soit le même
        pour tous les groupes.
        
        Args:
            X: Features
            y: Target
            sensitive_feature: Feature sensible (ex: gender)
            
        Returns:
            Dict: Résultats de l'audit
        """
        logger.info(f"Audit de la parité démographique pour {sensitive_feature.name}...")
        
        y_pred = self.model.predict(X)
        
        # Calculer les métriques de parité démographique
        dp_diff = demographic_parity_difference(y, y_pred, sensitive_features=sensitive_feature)
        dp_ratio = demographic_parity_ratio(y, y_pred, sensitive_features=sensitive_feature)
        
        # Taux de sélection par groupe
        selection_rates = {}
        for group in sensitive_feature.unique():
            mask = sensitive_feature == group
            selection_rates[group] = selection_rate(y[mask], y_pred[mask])
        
        results = {
            'metric': 'demographic_parity',
            'difference': float(dp_diff),
            'ratio': float(dp_ratio),
            'selection_rates': selection_rates,
            'interpretation': self._interpret_demographic_parity(dp_diff, dp_ratio)
        }
        
        logger.info(f"Différence de parité démographique: {dp_diff:.4f}")
        logger.info(f"Ratio de parité démographique: {dp_ratio:.4f}")
        
        return results
    
    def audit_equalized_odds(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_feature: pd.Series
    ) -> Dict:
        """
        Audite l'égalité des chances (Equalized Odds).
        
        L'égalité des chances exige que les taux de vrais positifs et de faux
        positifs soient les mêmes pour tous les groupes.
        
        Args:
            X: Features
            y: Target
            sensitive_feature: Feature sensible (ex: gender)
            
        Returns:
            Dict: Résultats de l'audit
        """
        logger.info(f"Audit de l'égalité des chances pour {sensitive_feature.name}...")
        
        y_pred = self.model.predict(X)
        
        # Calculer les métriques d'égalité des chances
        eo_diff = equalized_odds_difference(y, y_pred, sensitive_features=sensitive_feature)
        eo_ratio = equalized_odds_ratio(y, y_pred, sensitive_features=sensitive_feature)
        
        # Taux par groupe et par classe
        tpr_by_group = {}
        fpr_by_group = {}
        
        for group in sensitive_feature.unique():
            mask = sensitive_feature == group
            y_true_group = y[mask]
            y_pred_group = y_pred[mask]
            
            # True Positive Rate
            tp = np.sum((y_true_group == 1) & (y_pred_group == 1))
            fn = np.sum((y_true_group == 1) & (y_pred_group == 0))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # False Positive Rate
            fp = np.sum((y_true_group == 0) & (y_pred_group == 1))
            tn = np.sum((y_true_group == 0) & (y_pred_group == 0))
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_by_group[group] = float(tpr)
            fpr_by_group[group] = float(fpr)
        
        results = {
            'metric': 'equalized_odds',
            'difference': float(eo_diff),
            'ratio': float(eo_ratio),
            'true_positive_rates': tpr_by_group,
            'false_positive_rates': fpr_by_group,
            'interpretation': self._interpret_equalized_odds(eo_diff, eo_ratio)
        }
        
        logger.info(f"Différence d'égalité des chances: {eo_diff:.4f}")
        logger.info(f"Ratio d'égalité des chances: {eo_ratio:.4f}")
        
        return results
    
    def audit_all(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_features: pd.DataFrame
    ) -> Dict:
        """
        Effectue un audit complet de fairness sur toutes les features sensibles.
        
        Args:
            X: Features
            y: Target
            sensitive_features: DataFrame des features sensibles
            
        Returns:
            Dict: Résultats complets de l'audit
        """
        logger.info("Début de l'audit complet de fairness...")
        
        results = {
            'overall_metrics': self.compute_metrics_by_group(X, y, sensitive_features),
            'by_feature': {}
        }
        
        # Auditer chaque feature sensible
        for feature_name in sensitive_features.columns:
            feature = sensitive_features[feature_name]
            
            results['by_feature'][feature_name] = {
                'demographic_parity': self.audit_demographic_parity(X, y, feature),
                'equalized_odds': self.audit_equalized_odds(X, y, feature)
            }
        
        self.audit_results = results
        logger.info("Audit complet de fairness terminé")
        
        return results
    
    def mitigate_fairness(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_features_train: pd.Series,
        constraint: str = 'demographic_parity',
        method: str = 'exponentiated_gradient',
        **kwargs
    ) -> Dict:
        """
        Applique une technique d'atténuation des biais.
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            sensitive_features_train: Features sensibles d'entraînement
            constraint: Contrainte de fairness ('demographic_parity' ou 'equalized_odds')
            method: Méthode d'atténuation ('exponentiated_gradient' ou 'grid_search')
            **kwargs: Arguments supplémentaires
            
        Returns:
            Dict: Résultats de l'atténuation
        """
        logger.info(f"Atténuation des biais avec {method} et contrainte {constraint}...")
        
        # Définir la contrainte
        if constraint == 'demographic_parity':
            fairness_constraint = DemographicParity()
        elif constraint == 'equalized_odds':
            fairness_constraint = EqualizedOdds()
        else:
            raise ValueError(f"Contrainte non supportée: {constraint}")
        
        # Définir la méthode
        if method == 'exponentiated_gradient':
            mitigator = ExponentiatedGradient(
                estimator=self.model.model,
                constraints=fairness_constraint,
                **kwargs
            )
        elif method == 'grid_search':
            mitigator = GridSearch(
                estimator=self.model.model,
                constraints=fairness_constraint,
                **kwargs
            )
        else:
            raise ValueError(f"Méthode non supportée: {method}")
        
        # Entraîner avec atténuation
        mitigator.fit(X_train, y_train, sensitive_features=sensitive_features_train)
        
        # Évaluer le modèle mitigé
        y_pred_mitigated = mitigator.predict(X_train)
        
        # Calculer les métriques
        metrics = {
            'accuracy': accuracy_score(y_train, y_pred_mitigated),
            'dp_difference': float(demographic_parity_difference(
                y_train, y_pred_mitigated, sensitive_features=sensitive_features_train
            )),
            'eo_difference': float(equalized_odds_difference(
                y_train, y_pred_mitigated, sensitive_features=sensitive_features_train
            ))
        }
        
        results = {
            'mitigator': mitigator,
            'metrics': metrics,
            'constraint': constraint,
            'method': method
        }
        
        logger.info(f"Atténuation terminée. Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"DP Difference: {metrics['dp_difference']:.4f}")
        
        return results
    
    def _interpret_demographic_parity(self, diff: float, ratio: float) -> str:
        """
        Interprète les résultats de la parité démographique.
        
        Args:
            diff: Différence de parité démographique
            ratio: Ratio de parité démographique
            
        Returns:
            str: Interprétation
        """
        if diff < 0.05 and ratio > 0.95:
            return "Excellent: Le modèle est très équitable selon la parité démographique."
        elif diff < 0.1 and ratio > 0.8:
            return "Bon: Le modèle est raisonnablement équitable."
        elif diff < 0.2 and ratio > 0.6:
            return "Moyen: Le modèle présente des disparités modérées."
        else:
            return "Mauvais: Le modèle présente des disparités importantes et nécessite une atténuation."
    
    def _interpret_equalized_odds(self, diff: float, ratio: float) -> str:
        """
        Interprète les résultats de l'égalité des chances.
        
        Args:
            diff: Différence d'égalité des chances
            ratio: Ratio d'égalité des chances
            
        Returns:
            str: Interprétation
        """
        if diff < 0.05 and ratio > 0.95:
            return "Excellent: Le modèle est très équitable selon l'égalité des chances."
        elif diff < 0.1 and ratio > 0.8:
            return "Bon: Le modèle est raisonnablement équitable."
        elif diff < 0.2 and ratio > 0.6:
            return "Moyen: Le modèle présente des disparités modérées."
        else:
            return "Mauvais: Le modèle présente des disparités importantes et nécessite une atténuation."
    
    def generate_report(self, results: Optional[Dict] = None) -> str:
        """
        Génère un rapport textuel de l'audit de fairness.
        
        Args:
            results: Résultats de l'audit (si None, utilise self.audit_results)
            
        Returns:
            str: Rapport textuel
        """
        if results is None:
            results = self.audit_results
        
        if not results:
            return "Aucun résultat d'audit disponible."
        
        report = []
        report.append("=" * 60)
        report.append("RAPPORT D'AUDIT DE FAIRNESS")
        report.append("=" * 60)
        report.append("")
        
        # Métriques globales
        report.append("MÉTRIQUES GLOBALES PAR GROUPE:")
        report.append("-" * 40)
        overall = results['overall_metrics']
        for metric_name in overall.by_group.columns:
            report.append(f"\n{metric_name}:")
            for group, value in overall.by_group[metric_name].items():
                report.append(f"  {group}: {value:.4f}")
        
        # Par feature sensible
        report.append("\n" + "=" * 60)
        report.append("ANALYSE PAR FEATURE SENSIBLE")
        report.append("=" * 60)
        
        for feature_name, feature_results in results['by_feature'].items():
            report.append(f"\n{feature_name.upper()}:")
            report.append("-" * 40)
            
            # Parité démographique
            dp = feature_results['demographic_parity']
            report.append(f"\nParité Démographique:")
            report.append(f"  Différence: {dp['difference']:.4f}")
            report.append(f"  Ratio: {dp['ratio']:.4f}")
            report.append(f"  Interprétation: {dp['interpretation']}")
            
            # Égalité des chances
            eo = feature_results['equalized_odds']
            report.append(f"\nÉgalité des Chances:")
            report.append(f"  Différence: {eo['difference']:.4f}")
            report.append(f"  Ratio: {eo['ratio']:.4f}")
            report.append(f"  Interprétation: {eo['interpretation']}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Test du module
    print("Test du module FairnessAuditor...")
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
    
    # Créer l'auditeur de fairness
    auditor = FairnessAuditor(model)
    
    # Préparer les features sensibles
    sensitive_features = X[['gender', 'age_group']].copy()
    
    # Audit complet
    results = auditor.audit_all(X_test, y_test, sensitive_features)
    
    # Afficher le rapport
    print("\n" + auditor.generate_report())