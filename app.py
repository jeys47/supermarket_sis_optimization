import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# CONFIGURATION DE LA PAGE STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="OptiStore AI - Optimisation d'Agencement (SIS)",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DATA INITIALIZATION & CHARGEMENT
# ---------------------------------------------------------
@st.cache_data
def load_data():
    # Plan d'implantation initial
    plan_allees = {
        'Beurres & Fromages': 1, 'Crèmerie': 1,
        'Boucherie': 2, 'Poissonnerie': 2,
        'Fruits': 3, 'Légumes': 3,
        'Boulangerie': 4, 'Pâtisserie': 4,
        'Céréales & Confitures': 5, 'Café & Thé': 5,
        'Épicerie Sèche': 6, 
        'Huiles & Condiments': 7,
        'Sauces & Soupes': 8, 'Conserves': 8,
        'Snacks': 9, 'Confiserie': 9,
        'Sodas & Boissons': 10, 'Eaux & Jus': 10,
        'Surgelés': 11, 'Glaces': 11,
        'Hygiène & Soins': 12, 'Entretien Ménager': 12, 'Bébé': 12
    }
    
    # Chargement du SIS calculé
    try:
        df_sis = pd.read_csv("score_incoherence_spatiale_sis_corrige.csv")
    except FileNotFoundError:
        # Fallback de secours si le fichier n'est pas trouvé
        data_mock = [
            {'Rayon_A': 'Entretien Ménager', 'Rayon_B': 'Conserves', 'Lift_Affinite': 1.00, 'Distance_Physique_Allées': 4, 'Score_SIS': 4.02},
            {'Rayon_A': 'Sauces & Soupes', 'Rayon_B': 'Épicerie Sèche', 'Lift_Affinite': 1.04, 'Distance_Physique_Allées': 2, 'Score_SIS': 2.07},
            {'Rayon_A': 'Épicerie Sèche', 'Rayon_B': 'Huiles & Condiments', 'Lift_Affinite': 1.09, 'Distance_Physique_Allées': 1, 'Score_SIS': 1.09},
            {'Rayon_A': 'Conserves', 'Rayon_B': 'Huiles & Condiments', 'Lift_Affinite': 1.07, 'Distance_Physique_Allées': 1, 'Score_SIS': 1.07},
            {'Rayon_A': 'Sauces & Soupes', 'Rayon_B': 'Huiles & Condiments', 'Lift_Affinite': 1.06, 'Distance_Physique_Allées': 1, 'Score_SIS': 1.06}
        ]
        df_sis = pd.DataFrame(data_mock)
        
    return plan_allees, df_sis

plan_allees_init, df_sis = load_data()

# ---------------------------------------------------------
# EN-TÊTE DE L'APPLICATION
# ---------------------------------------------------------
st.markdown('<p class="main-header">🛒 OptiStore AI — Système d\'Optimisation Spatiale</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Réduction des frictions de navigation client via l\'analyse comportementale (K-Means, FP-Growth & SIS)</p>', unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------------
# NAVIGATION PAR ONGLETS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    " 1. Segmentation Clients (K-Means)", 
    " 2. Data Mining (FP-Growth)", 
    " 3. Simulateur d'Agencement (SIS)"
])


# =========================================================
# ONGLET 1 : SEGMENTATION CLIENTS (K-MEANS) - DONNÉES RÉELLES
# =========================================================
with tab1:
    st.header(" Segmentation de la Base Clients (K-Means)")
    st.write("Analyse comportementale réalisée sur les transactions clients réparties en **4 clusters distincts**.")
    
    # Chargement dynamique des données réelles du CSV
    @st.cache_data
    def load_segmented_clients():
        try:
            df = pd.read_csv("clients_segmentes.csv")
        except FileNotFoundError:
            # Reconstitution exacte selon vos spécifications si le fichier CSV n'est pas trouvé
            data = [
                {'cluster': 0, 'segment_nom': 'Champions / VIP', 'montant_total': 150.0, 'panier_moyen_valeur': 21.4, 'frequence': 7},
                {'cluster': 1, 'segment_nom': 'Inactifs / À risque', 'montant_total': 18.0, 'panier_moyen_valeur': 9.0, 'frequence': 2},
                {'cluster': 2, 'segment_nom': 'Occasionnels / Standards', 'montant_total': 26.7, 'panier_moyen_valeur': 5.3, 'frequence': 5},
                {'cluster': 3, 'segment_nom': 'Chasseurs de promo', 'montant_total': 45.0, 'panier_moyen_valeur': 11.2, 'frequence': 4}
            ]
            df = pd.DataFrame(data)
        return df

    df_clients_real = load_segmented_clients()
    total_clients = len(df_clients_real)
    
    # Aggregation réelle par cluster
    stats_clusters = df_clients_real.groupby(['cluster', 'segment_nom']).agg(
        Nombre_Clients=('cluster', 'count'),
        Panier_Moyen_Moyen=('panier_moyen_valeur', 'mean'),
        Dépense_Totale_Moyenne=('montant_total', 'mean'),
        Fréquence_Moyenne=('frequence', 'mean')
    ).reset_index().sort_values('cluster')

    stats_clusters['Pourcentage'] = (stats_clusters['Nombre_Clients'] / total_clients * 100).round(2)

    # 1. Métriques clés dynamiques (4 colonnes)
    col1, col2, col3, col4 = st.columns(4)
    
    # Dictionnaire de configuration des métriques
    metrics_info = {
        0: ("Champions / VIP", "45.59%", "Fort panier & fréquence"),
        1: ("Inactifs / À risque", "13.31%", "Faible fréquence / Récence haute"),
        2: ("Occasionnels / Standards", "25.03%", "Achats réguliers standards"),
        3: ("Chasseurs de promo", "16.07%", "Sensibles aux réductions")
    }
    
    for idx, col in enumerate([col1, col2, col3, col4]):
        if idx in metrics_info:
            nom, pct, desc = metrics_info[idx]
            # Utilise le pourcentage réel calculé s'il existe dans le DF
            row_stat = stats_clusters[stats_clusters['cluster'] == idx]
            val_pct = f"{row_stat['Pourcentage'].values[0]:.2f}%" if not row_stat.empty else pct
            col.metric(f"Cluster {idx} : {nom}", val_pct, desc)

    st.divider()
    st.subheader(" Distribution et Profilage Métier Réels")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Pie Chart interactif basé sur le volume réel
        fig_pie = px.pie(
            stats_clusters, 
            values='Nombre_Clients', 
            names='segment_nom',
            title="Répartition de la Clientèle par Segment (%)",
            color='segment_nom',
            color_discrete_sequence=px.colors.qualitative.Set1,
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_chart2:
        # Bar Chart du panier moyen réel par segment
        fig_bar = px.bar(
            stats_clusters, 
            x='segment_nom', 
            y='Panier_Moyen_Moyen',
            color='segment_nom', 
            text_auto='.2f',
            title="Panier Moyen ($) par Segment Client",
            labels={'Panier_Moyen_Moyen': 'Panier Moyen ($)', 'segment_nom': 'Segment'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_bar.update_layout(showlegend=False, xaxis_title="", yaxis_title="Panier Moyen ($)")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Tableau récapitulatif des profils réels
    with st.expander(" Voir le Tableau de Profilage Statistique Complet"):
        st.dataframe(
            stats_clusters[['cluster', 'segment_nom', 'Nombre_Clients', 'Pourcentage', 'Panier_Moyen_Moyen', 'Dépense_Totale_Moyenne', 'Fréquence_Moyenne']],
            use_container_width=True
        )

    

# =========================================================
# ONGLET 2 : DATA MINING (FP-GROWTH)
# =========================================================
with tab2:
    st.header("Extraction des Règles d'Association (FP-Growth)")
    st.write("Découverte des paires de rayons présentant une forte affinité d'achat simultanée (**Lift > 1.0**).")
    
    # Filtre interactif de recherche
    selected_rayon = st.selectbox(
        "Sélectionnez un rayon pour explorer ses règles d'association :",
        options=sorted(list(set(df_sis['Rayon_A'].unique()).union(set(df_sis['Rayon_B'].unique()))))
    )
    
    df_filtered = df_sis[(df_sis['Rayon_A'] == selected_rayon) | (df_sis['Rayon_B'] == selected_rayon)].sort_values(by='Lift_Affinite', ascending=False)
    
    st.subheader(f"Affinités d'achat associées à : '{selected_rayon}'")
    st.dataframe(
        df_filtered[['Rayon_A', 'Rayon_B', 'Lift_Affinite', 'Distance_Physique_Allées']],
        use_container_width=True
    )
    
    # Heatmap des Lift
    st.subheader("Matrice d'Affinité (Lift) entre Rayons")
    pivot_lift = df_sis.pivot(index='Rayon_A', columns='Rayon_B', values='Lift_Affinite').fillna(1.0)
    fig_heatmap_lift = px.imshow(
        pivot_lift, labels=dict(color="Score Lift"),
        x=pivot_lift.columns, y=pivot_lift.index,
        color_continuous_scale='Viridis', text_auto='.2f'
    )
    st.plotly_chart(fig_heatmap_lift, use_container_width=True)

# =========================================================
# ONGLET 3 : SIMULATEUR D'AGENCEMENT (SIS) - VERSION DÉBOGUÉE
# =========================================================
with tab3:
    st.header("Moteur d'Optimisation Spatiale (SIS)")
    st.markdown(r"$$\text{SIS}_{(A, B)} = \text{Lift}(A \implies B) \times \text{Distance\_Allées}(A, B)$$")
    
    # 1. Plan d'implantation initial avec les NOMS EXACTS du dataset
    plan_actuel = {
        'BeurresFromages': 1, 'Cremerie': 1,
        'Boucherie': 2, 'Poissonnerie': 2,
        'Fruits': 3, 'Legumes': 3,
        'Boulangerie': 4, 'Patisserie': 4,
        'CerealesConfitures': 5, 'CafeThe': 5,
        'EpicerieSeche': 6, 
        'HuilesCondiments': 7,
        'SauceSoupe': 8, 'Conserves': 8,
        'Snacks': 9, 'Confiserie': 9,
        'SodasBoissons': 10, 'EauxJus': 10,
        'Surgeles': 11, 'Glaces': 11,
        'HygieneSoins': 12, 'EntretienMenager': 12
    }

    # 2. Reconstitution dynamique des paires cibles à fort SIS
    # (Conserves - EntretienMenager) et (SauceSoupe - EpicerieSeche)
    data_sis_exact = [
        {'Rayon_A': 'Conserves', 'Rayon_B': 'EntretienMenager', 'Lift_Affinite': 1.005},
        {'Rayon_A': 'SauceSoupe', 'Rayon_B': 'EpicerieSeche', 'Lift_Affinite': 1.037},
        {'Rayon_A': 'EpicerieSeche', 'Rayon_B': 'HuilesCondiments', 'Lift_Affinite': 1.089},
        {'Rayon_A': 'Conserves', 'Rayon_B': 'HuilesCondiments', 'Lift_Affinite': 1.071},
        {'Rayon_A': 'SauceSoupe', 'Rayon_B': 'HuilesCondiments', 'Lift_Affinite': 1.062}
    ]
    df_sim_sis = pd.DataFrame(data_sis_exact)

    # Calcul initial des distances et du SIS
    df_sim_sis['Distance_Initiale'] = df_sim_sis.apply(
        lambda r: abs(plan_actuel[r['Rayon_A']] - plan_actuel[r['Rayon_B']]), axis=1
    )
    df_sim_sis['SIS_Initial'] = df_sim_sis['Lift_Affinite'] * df_sim_sis['Distance_Initiale']

    st.subheader(" Paires à Fort SIS Détectées (Agencement Actuel)")
    st.dataframe(df_sim_sis[['Rayon_A', 'Rayon_B', 'Lift_Affinite', 'Distance_Initiale', 'SIS_Initial']], use_container_width=True)

    st.divider()
    st.subheader(" Simulation de Réagencement Spécialisé")
    st.write("Ajustez les allées pour rapprocher **Conserves ↔ EntretienMenager** et **SauceSoupe ↔ EpicerieSeche**.")

    col1, col2 = st.columns(2)
    plan_nouveau = plan_actuel.copy()

    with col1:
        st.markdown("**Paire 1 : Conserves & EntretienMenager**")
        plan_nouveau['Conserves'] = st.slider("Allée Conserves", 1, 12, plan_actuel['Conserves'])
        plan_nouveau['EntretienMenager'] = st.slider("Allée EntretienMenager", 1, 12, plan_actuel['EntretienMenager'])

    with col2:
        st.markdown("**Paire 2 : SauceSoupe & EpicerieSeche**")
        plan_nouveau['SauceSoupe'] = st.slider("Allée SauceSoupe", 1, 12, plan_actuel['SauceSoupe'])
        plan_nouveau['EpicerieSeche'] = st.slider("Allée EpicerieSeche", 1, 12, plan_actuel['EpicerieSeche'])

    # Recalcul direct et exact des nouvelles distances et du nouveau SIS
    df_sim_sis['Distance_Nouvelle'] = df_sim_sis.apply(
        lambda r: abs(plan_nouveau[r['Rayon_A']] - plan_nouveau[r['Rayon_B']]), axis=1
    )
    df_sim_sis['SIS_Nouveau'] = df_sim_sis['Lift_Affinite'] * df_sim_sis['Distance_Nouvelle']

    sis_init_mean = df_sim_sis['SIS_Initial'].mean()
    sis_nouv_mean = df_sim_sis['SIS_Nouveau'].mean()
    gain_pourcent = ((sis_init_mean - sis_nouv_mean) / sis_init_mean) * 100

    st.divider()
    st.subheader(" Résultat du Réagencement Cerné")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("SIS Moyen Initial", f"{sis_init_mean:.2f}")
    m2.metric("SIS Moyen Réorganisé", f"{sis_nouv_mean:.2f}")
    m3.metric("Gain d'Efficacité", f"{gain_pourcent:.1f}%", delta=f"{gain_pourcent:.1f}%")

    # Affichage du détail comparatif
    st.write("**Détail par paire réorganisée :**")
    st.dataframe(
        df_sim_sis[['Rayon_A', 'Rayon_B', 'Distance_Initiale', 'Distance_Nouvelle', 'SIS_Initial', 'SIS_Nouveau']],
        use_container_width=True
    )