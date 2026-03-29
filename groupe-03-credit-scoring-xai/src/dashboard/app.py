"""
Dashboard Streamlit pour le Credit Scoring XAI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_loader import load_data
from src.preprocessing import DataPreprocessor, split_data
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.models.baseline_model import BaselineModel
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMEExplainer
from src.explainability.counterfactual import CounterfactualExplainer
from src.fairness.fairness_audit import FairnessAuditor
from src.evaluation import ModelEvaluator
from src.config import MODELS_DIR, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, CATEGORICAL_MAPPINGS

# Configuration de la page
st.set_page_config(
    page_title="Credit Scoring XAI Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-header">💳 Credit Scoring XAI Dashboard</h1>', unsafe_allow_html=True)

st.markdown("""
Ce dashboard interactif permet d'explorer un modèle de scoring de crédit avec des techniques d'IA Explicable (XAI).
Vous pouvez faire des prédictions, comprendre les décisions du modèle et auditer son équité.
""")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Sélectionnez une page",
    ["🏠 Accueil", "📊 Prédiction", "🔍 Explicabilité", "⚖️ Fairness", "📈 Comparaison Modèles"]
)

# Chargement des données et modèles
@st.cache_resource
def load_models_and_data():
    """Charge les données et les modèles."""
    with st.spinner("Chargement des données et des modèles..."):
        # Charger les données
        X, y = load_data()
        preprocessor = DataPreprocessor()
        X_transformed = preprocessor.fit_transform(X)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_transformed, y)
        
        # Charger ou entraîner les modèles
        models = {}
        
        # Essayer de charger les modèles sauvegardés
        baseline_path = MODELS_DIR / "baseline_model.pkl"
        xgb_path = MODELS_DIR / "xgboost_model.pkl"
        lgb_path = MODELS_DIR / "lightgbm_model.pkl"
        
        if baseline_path.exists():
            baseline = BaselineModel()
            baseline.load(str(baseline_path))
            models['Baseline'] = baseline
        else:
            baseline = BaselineModel()
            baseline.train(X_train, y_train, X_val, y_val)
            baseline.save(str(baseline_path))
            models['Baseline'] = baseline
        
        if xgb_path.exists():
            xgb = XGBoostModel()
            xgb.load(str(xgb_path))
            models['XGBoost'] = xgb
        else:
            xgb = XGBoostModel()
            xgb.train(X_train, y_train, X_val, y_val)
            xgb.save(str(xgb_path))
            models['XGBoost'] = xgb
        
        if lgb_path.exists():
            lgb = LightGBMModel()
            lgb.load(str(lgb_path))
            models['LightGBM'] = lgb
        else:
            lgb = LightGBMModel()
            lgb.train(X_train, y_train, X_val, y_val)
            lgb.save(str(lgb_path))
            models['LightGBM'] = lgb
        
        # Initialiser les explainers
        shap_explainer = SHAPExplainer(xgb, feature_names=preprocessor.feature_names)
        shap_explainer.fit(X_train)
        
        lime_explainer = LIMEExplainer(xgb, feature_names=preprocessor.feature_names)
        lime_explainer.fit(X_train)
        
        cf_explainer = CounterfactualExplainer(xgb, feature_names=preprocessor.feature_names)
        
        fairness_auditor = FairnessAuditor(xgb)
        
        evaluator = ModelEvaluator()
        
        return {
            'X': X,
            'y': y,
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'preprocessor': preprocessor,
            'models': models,
            'shap_explainer': shap_explainer,
            'lime_explainer': lime_explainer,
            'cf_explainer': cf_explainer,
            'fairness_auditor': fairness_auditor,
            'evaluator': evaluator
        }

# Charger les données
data = load_models_and_data()

# Page Accueil
if page == "🏠 Accueil":
    st.header("Bienvenue sur le Dashboard Credit Scoring XAI")
    
    st.markdown("""
    ### 📋 Contexte du projet
    
    Ce projet s'inscrit dans le cadre du cours d'IA Probabiliste, Théorie des Jeux et Machine Learning de l'ECE Paris.
    L'objectif est de développer un système de scoring de crédit utilisant des techniques de Machine Learning avancées,
    avec un accent particulier sur l'explicabilité des décisions (XAI - Explainable AI).
    
    ### 🎯 Objectifs
    
    - Développer un modèle de scoring de crédit performant (XGBoost/LightGBM)
    - Implémenter des techniques d'explicabilité (SHAP, LIME)
    - Générer des explications contrefactuelles
    - Auditer le fairness du modèle (Fairlearn)
    - Comparer modèle boîte noire vs modèle interprétable
    
    ### 📊 Dataset
    
    Le projet utilise le **German Credit Dataset** de l'UCI Machine Learning Repository :
    - 1000 instances
    - 20 attributs (7 numériques, 13 catégoriels)
    - Variable cible : Credit risk (Good/Bad)
    """)
    
    # Statistiques des données
    st.subheader("📈 Statistiques des données")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total des instances", len(data['X']))
    with col2:
        st.metric("Features", data['X'].shape[1])
    with col3:
        good_credit = (data['y'] == 1).sum()
        bad_credit = (data['y'] == 0).sum()
        st.metric("Bon crédit (%)", f"{good_credit/len(data['y'])*100:.1f}%")
    
    # Distribution de la cible
    fig = px.pie(
        values=[good_credit, bad_credit],
        names=['Bon crédit', 'Mauvais crédit'],
        title='Distribution du risque de crédit',
        color_discrete_sequence=['#2ecc71', '#e74c3c']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution des features numériques
    st.subheader("Distribution des features numériques")
    numerical_features = ['duration', 'credit_amount', 'age']
    
    for feature in numerical_features:
        if feature in data['X'].columns:
            fig = px.histogram(
                data['X'],
                x=feature,
                title=f'Distribution de {feature}',
                nbins=30
            )
            st.plotly_chart(fig, use_container_width=True)

# Page Prédiction
elif page == "📊 Prédiction":
    st.header("🔮 Prédiction de risque de crédit")
    
    # Sélection du modèle
    model_name = st.selectbox(
        "Sélectionnez le modèle",
        list(data['models'].keys())
    )
    model = data['models'][model_name]
    
    # Sélection de l'instance
    st.subheader("Sélectionnez une instance à prédire")
    
    option = st.radio(
        "Comment souhaitez-vous sélectionner l'instance ?",
        ["Depuis le dataset de test", "Entrer manuellement les valeurs"]
    )
    
    if option == "Depuis le dataset de test":
        instance_idx = st.slider(
            "Index de l'instance",
            0,
            len(data['X_test']) - 1,
            0
        )
        
        # Afficher les valeurs de l'instance
        instance_original = data['X'].loc[data['X_test'].index[instance_idx]]
        st.write("Valeurs de l'instance :")
        st.dataframe(instance_original.to_frame().T)
        
        # Prédire
        instance_transformed = data['X_test'].iloc[instance_idx]
        prediction = model.predict(instance_transformed.values.reshape(1, -1))[0]
        probability = model.predict_proba(instance_transformed.values.reshape(1, -1))[0]
        
    else:
        # Formulaire manuel
        st.write("Entrez les valeurs pour la prédiction :")
        
        # Créer un formulaire pour les features
        form_data = {}
        
        # Features numériques
        st.subheader("Features numériques")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            form_data['duration'] = st.number_input("Durée (mois)", 4, 72, 18)
            form_data['credit_amount'] = st.number_input("Montant du crédit (DM)", 250, 18424, 3000)
        
        with col2:
            form_data['installment_rate'] = st.slider("Taux d'installment (%)", 1, 4, 2)
            form_data['residence_since'] = st.slider("Résidence depuis (années)", 1, 4, 2)
        
        with col3:
            form_data['age'] = st.number_input("Âge", 19, 75, 35)
            form_data['existing_credits'] = st.slider("Crédits existants", 1, 4, 1)
            form_data['people_liable'] = st.slider("Personnes responsables", 1, 2, 1)
        
        # Features catégorielles
        st.subheader("Features catégorielles")
        
        col1, col2 = st.columns(2)
        
        with col1:
            form_data['checking_account'] = st.selectbox(
                "Compte courant",
                list(CATEGORICAL_MAPPINGS['checking_account'].values())
            )
            form_data['credit_history'] = st.selectbox(
                "Historique de crédit",
                list(CATEGORICAL_MAPPINGS['credit_history'].values())
            )
            form_data['purpose'] = st.selectbox(
                "Objet du crédit",
                list(CATEGORICAL_MAPPINGS['purpose'].values())
            )
            form_data['savings_account'] = st.selectbox(
                "Compte d'épargne",
                list(CATEGORICAL_MAPPINGS['savings_account'].values())
            )
        
        with col2:
            form_data['employment_since'] = st.selectbox(
                "Emploi depuis",
                list(CATEGORICAL_MAPPINGS['employment_since'].values())
            )
            form_data['personal_status'] = st.selectbox(
                "Statut personnel",
                list(CATEGORICAL_MAPPINGS['personal_status'].values())
            )
            form_data['property'] = st.selectbox(
                "Propriété",
                list(CATEGORICAL_MAPPINGS['property'].values())
            )
            form_data['housing'] = st.selectbox(
                "Logement",
                list(CATEGORICAL_MAPPINGS['housing'].values())
            )

        col3, col4 = st.columns(2)

        with col3:
            form_data['other_debtors'] = st.selectbox(
                "Autres débiteurs / garants",
                list(CATEGORICAL_MAPPINGS['other_debtors'].values())
            )
            form_data['other_installment_plans'] = st.selectbox(
                "Autres plans d'épargne",
                list(CATEGORICAL_MAPPINGS['other_installment_plans'].values())
            )
            form_data['job'] = st.selectbox(
                "Type d'emploi",
                list(CATEGORICAL_MAPPINGS['job'].values())
            )

        with col4:
            form_data['telephone'] = st.selectbox(
                "Téléphone",
                list(CATEGORICAL_MAPPINGS['telephone'].values())
            )
            form_data['foreign_worker'] = st.selectbox(
                "Travailleur étranger",
                list(CATEGORICAL_MAPPINGS['foreign_worker'].values())
            )

        # Transformer et prédire
        instance_df = pd.DataFrame([form_data])
        instance_transformed = data['preprocessor'].transform(instance_df)
        
        prediction = model.predict(instance_transformed)[0]
        probability = model.predict_proba(instance_transformed)[0]
    
    # Afficher les résultats
    st.subheader("Résultats de la prédiction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.markdown('<div class="success-box">✅ Crédit APPROUVÉ</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="danger-box">❌ Crédit REFUSÉ</div>', unsafe_allow_html=True)
    
    with col2:
        st.metric("Probabilité de bon crédit", f"{probability:.2%}")
    
    # Jauge de probabilité
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        title = {'text': "Score de confiance"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 50], 'color': "#e74c3c"},
                {'range': [50, 75], 'color': "#f39c12"},
                {'range': [75, 100], 'color': "#2ecc71"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# Page Explicabilité
elif page == "🔍 Explicabilité":
    st.header("🔍 Explicabilité des prédictions")
    
    # Sélection du modèle
    model_name = st.selectbox(
        "Sélectionnez le modèle",
        list(data['models'].keys())
    )
    model = data['models'][model_name]
    
    # Sélection de l'instance
    instance_idx = st.slider(
        "Index de l'instance (test set)",
        0,
        len(data['X_test']) - 1,
        0
    )
    
    instance_transformed = data['X_test'].iloc[instance_idx]
    instance_original = data['X'].loc[data['X_test'].index[instance_idx]]
    
    # Prédiction
    prediction = model.predict(instance_transformed.values.reshape(1, -1))[0]
    probability = model.predict_proba(instance_transformed.values.reshape(1, -1))[0]
    
    st.write(f"**Prédiction :** {'Bon crédit' if prediction == 1 else 'Mauvais crédit'} (probabilité: {probability:.2%})")
    
    # Onglets pour les différentes méthodes d'explicabilité
    tab1, tab2, tab3 = st.tabs(["SHAP", "LIME", "Contrefactuel"])
    
    with tab1:
        st.subheader("SHAP (SHapley Additive exPlanations)")
        
        st.markdown("""
        SHAP est une méthode d'explicabilité basée sur la théorie des jeux qui attribue une importance
        à chaque feature pour une prédiction spécifique.
        """)
        
        # Calculer les valeurs SHAP
        shap_values = data['shap_explainer'].explain(data['X_test'])
        
        # Importance globale
        st.subheader("Importance globale des features")
        importance = data['shap_explainer'].get_feature_importance()
        
        fig = px.bar(
            importance.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title='Top 10 des features les plus importantes (SHAP)'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

        top1 = importance.iloc[0]
        top2 = importance.iloc[1]
        top3 = importance.iloc[2]
        st.info(
            f"**Lecture du graphique :** Ce graphique montre l'importance moyenne (|valeur SHAP|) "
            f"de chaque variable sur l'ensemble du jeu de test. "
            f"La variable la plus déterminante est **{top1['feature']}** "
            f"(importance SHAP = {top1['importance']:.4f}), suivie de **{top2['feature']}** "
            f"({top2['importance']:.4f}) et **{top3['feature']}** ({top3['importance']:.4f}). "
            f"Plus la barre est longue, plus la variable influence les décisions du modèle en moyenne."
        )

        # Explication locale
        st.subheader("Explication locale")
        local_expl = data['shap_explainer'].get_local_explanation(instance_idx, data['X_test'])
        
        st.write("Contribution des features à cette prédiction :")
        
        # Créer un DataFrame pour l'affichage
        contrib_df = pd.DataFrame(local_expl['features'][:10])
        contrib_df['abs_contribution'] = contrib_df['shap_value'].abs()
        contrib_df = contrib_df.sort_values('abs_contribution', ascending=False)
        
        fig = px.bar(
            contrib_df,
            x='shap_value',
            y='feature',
            orientation='h',
            color='impact',
            color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c'},
            title='Contribution des features (Top 10)'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

        decision_label = "bon crédit" if prediction == 1 else "mauvais crédit"
        pos_feats = [f['feature'] for f in local_expl['features'][:10] if f['impact'] == 'positive'][:3]
        neg_feats = [f['feature'] for f in local_expl['features'][:10] if f['impact'] == 'negative'][:3]
        shap_top = contrib_df.iloc[0]
        st.info(
            f"**Lecture du graphique :** Pour cette instance, le modèle prédit **{decision_label}** "
            f"avec une probabilité de {probability:.1%}. "
            f"La variable **{shap_top['feature']}** est la plus influente "
            f"(contribution SHAP = {shap_top['shap_value']:.4f}, impact {shap_top['impact']}). "
            + (f"Les variables qui *favorisent* cette décision : {', '.join(pos_feats)}. " if pos_feats else "")
            + (f"Les variables qui *pénalisent* : {', '.join(neg_feats)}." if neg_feats else "")
        )

    with tab2:
        st.subheader("LIME (Local Interpretable Model-agnostic Explanations)")
        
        st.markdown("""
        LIME est une méthode d'explicabilité qui approxime localement le modèle avec un modèle
        interprétable pour expliquer une prédiction individuelle.
        """)
        
        # Générer l'explication LIME
        explanation = data['lime_explainer'].explain_instance(instance_transformed.values)

        # Probabilité via le modèle (ne dépend pas du format de local_pred LIME)
        lime_prob = float(model.predict_proba(instance_transformed.values.reshape(1, -1))[0])

        # Récupérer les features LIME de façon robuste
        try:
            lime_raw = explanation.as_list(label=1)
        except Exception:
            try:
                lime_raw = explanation.as_list(label=0)
            except Exception:
                lime_raw = explanation.as_list()

        lime_features = [
            {'feature': f, 'contribution': v, 'impact': 'positive' if v > 0 else 'negative'}
            for f, v in lime_raw
        ]
        lime_features.sort(key=lambda x: abs(x['contribution']), reverse=True)

        st.write(f"**Probabilité prédite :** {lime_prob:.2%}")

        # Afficher les features
        st.write("Contribution des features :")

        lime_df = pd.DataFrame(lime_features[:10])
        lime_df['abs_contribution'] = lime_df['contribution'].abs()
        lime_df = lime_df.sort_values('abs_contribution', ascending=False)

        fig = px.bar(
            lime_df,
            x='contribution',
            y='feature',
            orientation='h',
            color='impact',
            color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c'},
            title='Contribution des features LIME (Top 10)'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

        lime_top = lime_df.iloc[0]
        lime_pos = [f['feature'] for f in lime_features[:10] if f['impact'] == 'positive'][:3]
        lime_neg = [f['feature'] for f in lime_features[:10] if f['impact'] == 'negative'][:3]
        st.info(
            f"**Lecture du graphique :** LIME approxime le modèle localement autour de cette instance "
            f"par une régression linéaire interprétable. La probabilité estimée est "
            f"**{lime_prob:.1%}**. "
            f"La variable la plus influente selon LIME est **{lime_top['feature']}** "
            f"(poids = {lime_top['contribution']:.4f}). "
            + (f"Variables favorables : {', '.join(lime_pos)}. " if lime_pos else "")
            + (f"Variables défavorables : {', '.join(lime_neg)}." if lime_neg else "")
            + f" Contrairement à SHAP, ces poids sont valables uniquement au voisinage de cette instance."
        )

    with tab3:
        st.subheader("Explications Contrefactuelles")
        
        st.markdown("""
        Les explications contrefactuelles répondent à la question : "Que faudrait-il changer
        pour que cette demande de crédit soit acceptée ?"
        """)
        
        # Générer une explication contrefactuelle
        if prediction == 0:
            cf = data['cf_explainer'].generate_counterfactual(instance_transformed, target_class=1)
            
            if cf['success']:
                st.markdown('<div class="success-box">✅ Explication contrefactuelle générée avec succès</div>', unsafe_allow_html=True)
                
                st.write(f"**Prédiction originale :** {cf['original_prediction']:.2%}")
                st.write(f"**Prédiction contrefactuelle :** {cf['counterfactual_prediction']:.2%}")
                cf_distance = cf.get('distance')
                if cf_distance is not None:
                    st.write(f"**Distance :** {cf_distance:.4f}")
                
                # Afficher les changements
                if cf['changes']:
                    st.write("Changements suggérés :")
                    changes_df = pd.DataFrame(cf['changes'][:5])
                    st.dataframe(changes_df)
                    
                    # Explication textuelle
                    st.subheader("Explication textuelle")
                    st.write(data['cf_explainer'].explain_counterfactual(cf))

                    top_change = cf['changes'][0] if cf['changes'] else None
                    if top_change:
                        direction_fr = "augmenter" if top_change['direction'] == 'increase' else "réduire"
                        st.info(
                            f"**Lecture :** Le scénario contrefactuel montre que si le demandeur modifiait "
                            f"principalement **{top_change['feature']}** "
                            f"({top_change['original_value']:.2f} → {top_change['counterfactual_value']:.2f}, "
                            f"soit {direction_fr} de {abs(top_change['change']):.2f}), "
                            f"la probabilité d'approbation passerait de **{cf['original_prediction']:.1%}** "
                            f"à **{cf['counterfactual_prediction']:.1%}**. "
                            + (f"Distance totale de modification : {cf['distance']:.3f}." if cf.get('distance') is not None else "")
                        )
            else:
                st.markdown('<div class="warning-box">⚠️ Impossible de générer une explication contrefactuelle</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">ℹ️ Cette instance est déjà classée comme bon crédit. Aucun changement nécessaire.</div>', unsafe_allow_html=True)

# Page Fairness
elif page == "⚖️ Fairness":
    st.header("⚖️ Audit de Fairness")
    
    st.markdown("""
    L'audit de fairness évalue si le modèle traite équitablement différents groupes démographiques.
    Nous utilisons Fairlearn pour analyser les biais potentiels.
    """)
    
    # Sélectionner les features sensibles (alignées sur l'index du test set)
    sensitive_features = data['X'].loc[data['X_test'].index, ['gender', 'age_group']].copy()

    # Audit complet
    results = data['fairness_auditor'].audit_all(data['X_test'], data['y_test'], sensitive_features)
    
    # Parité démographique
    st.subheader("Parité Démographique")
    
    for feature_name in ['gender', 'age_group']:
        st.write(f"**{feature_name}**")
        dp = results['by_feature'][feature_name]['demographic_parity']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Différence", f"{dp['difference']:.4f}")
        with col2:
            st.metric("Ratio", f"{dp['ratio']:.4f}")
        with col3:
            st.write(dp['interpretation'])
        
        # Taux de sélection par groupe
        st.write("Taux de sélection par groupe :")
        selection_df = pd.DataFrame.from_dict(dp['selection_rates'], orient='index', columns=['Selection Rate'])
        st.dataframe(selection_df)
        
        st.write("---")
    
    # Égalité des chances
    st.subheader("Égalité des Chances (Equalized Odds)")
    
    for feature_name in ['gender', 'age_group']:
        st.write(f"**{feature_name}**")
        eo = results['by_feature'][feature_name]['equalized_odds']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Différence", f"{eo['difference']:.4f}")
        with col2:
            st.metric("Ratio", f"{eo['ratio']:.4f}")
        with col3:
            st.write(eo['interpretation'])
        
        # TPR et FPR par groupe
        st.write("Taux par groupe :")
        
        tpr_df = pd.DataFrame.from_dict(eo['true_positive_rates'], orient='index', columns=['True Positive Rate'])
        fpr_df = pd.DataFrame.from_dict(eo['false_positive_rates'], orient='index', columns=['False Positive Rate'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("True Positive Rate :")
            st.dataframe(tpr_df)
        with col2:
            st.write("False Positive Rate :")
            st.dataframe(fpr_df)
        
        st.write("---")
    
    # Métriques globales par groupe
    st.subheader("Métriques globales par groupe")
    
    overall = results['overall_metrics']
    
    for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
        st.write(f"**{metric_name}**")
        metric_df = pd.DataFrame(overall.by_group[metric_name])
        st.dataframe(metric_df)
        st.write("---")

# Page Comparaison Modèles
elif page == "📈 Comparaison Modèles":
    st.header("📈 Comparaison des Modèles")
    
    # Comparer tous les modèles
    comparison = data['evaluator'].compare_models(data['models'], data['X_test'], data['y_test'])
    
    st.subheader("Tableau de comparaison")
    st.dataframe(comparison)
    
    # Graphiques de comparaison
    st.subheader("Comparaison visuelle")
    
    # Graphique à barres
    fig = go.Figure()
    
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
        fig.add_trace(go.Bar(
            name=metric,
            x=comparison['Model'],
            y=comparison[metric]
        ))
    
    fig.update_layout(
        barmode='group',
        title='Comparaison des métriques par modèle',
        xaxis_title='Modèle',
        yaxis_title='Score',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Radar chart
    st.subheader("Radar Chart")
    
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    fig = go.Figure()
    
    for model_name in comparison['Model']:
        model_data = comparison[comparison['Model'] == model_name]
        values = model_data[categories].values[0].tolist()
        values += values[:1]  # Fermer le radar
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=model_name
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Comparaison radar des modèles"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Meilleur modèle
    best_model = comparison.iloc[0]
    st.subheader("Meilleur modèle")
    st.markdown(f"""
    <div class="success-box">
        <h3>🏆 {best_model['Model']}</h3>
        <p>Ce modèle obtient les meilleures performances avec un ROC-AUC de {best_model['ROC-AUC']:.4f}</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💳 Credit Scoring XAI Dashboard | ECE Paris - Ing4 Finance | 2026</p>
    <p>Cours : IA Probabiliste, Théorie des Jeux et Machine Learning</p>
</div>
""", unsafe_allow_html=True)


def main():
    """Fonction principale pour lancer le dashboard."""
    pass


if __name__ == "__main__":
    main()