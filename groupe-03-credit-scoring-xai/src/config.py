"""
Configuration globale du projet Credit Scoring XAI.
"""

import os
from pathlib import Path

# Chemins du projet
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DOCS_DIR = PROJECT_ROOT / "docs"
SLIDES_DIR = PROJECT_ROOT / "slides"

# Créer les répertoires s'ils n'existent pas
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Configuration du dataset
DATASET_NAME = "german_credit"
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
DATASET_FEATURES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc"

# Colonnes du dataset German Credit
GERMAN_CREDIT_COLUMNS = [
    'checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_account', 'employment_since', 'installment_rate', 'personal_status',
    'other_debtors', 'residence_since', 'property', 'age', 'other_installment_plans',
    'housing', 'existing_credits', 'job', 'people_liable', 'telephone', 'foreign_worker',
    'credit_risk'
]

# Mapping des valeurs catégorielles
CATEGORICAL_MAPPINGS = {
    'checking_account': {
        'A11': '< 0 DM',
        'A12': '0 - 200 DM',
        'A13': '> 200 DM',
        'A14': 'no checking account'
    },
    'credit_history': {
        'A30': 'no credits taken',
        'A31': 'all credits paid back duly',
        'A32': 'existing credits paid back duly till now',
        'A33': 'delay in paying off in the past',
        'A34': 'critical account'
    },
    'purpose': {
        'A40': 'car (new)',
        'A41': 'car (used)',
        'A42': 'furniture/equipment',
        'A43': 'radio/television',
        'A44': 'domestic appliances',
        'A45': 'repairs',
        'A46': 'education',
        'A47': 'vacation',
        'A48': 'retraining',
        'A49': 'business',
        'A410': 'others'
    },
    'savings_account': {
        'A61': '< 100 DM',
        'A62': '100 - 500 DM',
        'A63': '500 - 1000 DM',
        'A64': '> 1000 DM',
        'A65': 'unknown/no savings account'
    },
    'employment_since': {
        'A71': 'unemployed',
        'A72': '< 1 year',
        'A73': '1 - 4 years',
        'A74': '4 - 7 years',
        'A75': '> 7 years'
    },
    'personal_status': {
        'A91': 'male: divorced/separated',
        'A92': 'female: divorced/separated/married',
        'A93': 'male: single',
        'A94': 'male: married/widowed',
        'A95': 'female: single'
    },
    'other_debtors': {
        'A101': 'none',
        'A102': 'co-applicant',
        'A103': 'guarantor'
    },
    'property': {
        'A121': 'real estate',
        'A122': 'building society savings agreement',
        'A123': 'car or other',
        'A124': 'unknown/no property'
    },
    'other_installment_plans': {
        'A141': 'bank',
        'A142': 'stores',
        'A143': 'none'
    },
    'housing': {
        'A151': 'rent',
        'A152': 'own',
        'A153': 'for free'
    },
    'job': {
        'A171': 'unemployed/unskilled - non-resident',
        'A172': 'unskilled - resident',
        'A173': 'skilled employee/official',
        'A174': 'management/self-employed/highly qualified'
    },
    'telephone': {
        'A191': 'none',
        'A192': 'yes, registered under customer name'
    },
    'foreign_worker': {
        'A201': 'yes',
        'A202': 'no'
    }
}

# Colonnes numériques
NUMERICAL_COLUMNS = [
    'duration', 'credit_amount', 'installment_rate', 'residence_since',
    'age', 'existing_credits', 'people_liable'
]

# Colonnes catégorielles
CATEGORICAL_COLUMNS = [
    'checking_account', 'credit_history', 'purpose', 'savings_account',
    'employment_since', 'personal_status', 'other_debtors', 'property',
    'other_installment_plans', 'housing', 'job', 'telephone', 'foreign_worker'
]

# Colonnes sensibles pour l'audit de fairness
SENSITIVE_COLUMNS = {
    'gender': 'personal_status',  # Dérivé de personal_status
    'age': 'age'
}

# Configuration du modèle
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Hyperparamètres par défaut
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss'
}

LIGHTGBM_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'verbose': -1
}

# Métriques d'évaluation
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'confusion_matrix']

# Configuration Streamlit
STREAMLIT_CONFIG = {
    'title': 'Credit Scoring XAI Dashboard',
    'page_title': 'Credit Scoring XAI',
    'layout': 'wide'
}

# Configuration SHAP
SHAP_CONFIG = {
    'max_display': 20,
    'plot_size': (10, 8)
}

# Configuration LIME
LIME_CONFIG = {
    'num_features': 10,
    'num_samples': 5000
}

# Configuration Fairlearn
FAIRLEARN_CONFIG = {
    'sensitive_features': ['gender', 'age'],
    'mitigation_methods': ['ExponentiatedGradient', 'GridSearch']
}