import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration du style des graphiques
sns.set_theme(style="whitegrid")
plt.rcParams['font.size'] = 10

print("1. Chargement du dataset...")
# Modifiez le nom du fichier si nécessaire
df = pd.read_csv("supermarket_sales_100k.csv")

# Conversions de types pour l'analyse temporelle
df['date_achat'] = pd.to_datetime(df['date_achat'])
df['heure_num'] = df['heure_achat'].astype(str).str.split(':').str[0].astype(int)
print("\n==================================================")
print("             METRIQUES GLOBALES ENSEIGNE          ")
print("==================================================")
ca_total = df['prix_total'].sum()
nb_tickets = df['ticket_id'].nunique()
nb_clients = df['client_id'].nunique()
nb_produits = df['produit_id'].nunique()
nb_rayons = df['rayon_id'].nunique()

print(f"Chiffre d'Affaires Total : {ca_total:,.2f} $")
print(f"Nombre de Clients Unique  : {nb_clients:,}")
print(f"Nombre de Tickets/Sessions : {nb_tickets:,}")
print(f"Nombre de Lignes d'Achat  : {len(df):,}")
print(f"Panier Moyen (par ticket) : {ca_total / nb_tickets:.2f} $")
print(f"Nombre de Produits/Rayons : {nb_produits} produits répartis dans {nb_rayons} rayons")

# Calcul de la taille de chaque panier (ticket)
panier_stats = df.groupby('ticket_id').agg(
    nb_articles=('quantite', 'sum'),
    nb_references=('produit_id', 'count'),
    montant=('prix_total', 'sum')
)

print(f"Taille moyenne du panier  : {panier_stats['nb_articles'].mean():.2f} articles (Médiane : {panier_stats['nb_articles'].median():.0f})")

print("\n2. Génération du tableau de bord visuel (EDA)...")
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
plt.subplots_adjust(hspace=0.35, wspace=0.25)

# --- Graphique 1 : Distribution de la taille des paniers ---
counts_panier = panier_stats['nb_references'].value_counts(normalize=True).sort_index() * 100
# Regrouper les paniers de 7 articles et plus
counts_panier_grouped = counts_panier.head(6)
counts_panier_grouped[7] = counts_panier[counts_panier.index >= 7].sum()
labels = [str(i) for i in range(1, 7)] + ['7+']

sns.barplot(x=labels, y=counts_panier_grouped.values, ax=axes[0, 0], palette="mako")
axes[0, 0].set_title("Distribution de la taille des paniers (% de tickets)", fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel("Nombre d'articles par ticket")
axes[0, 0].set_ylabel("Pourcentage (%)")
for p in axes[0, 0].patches:
    axes[0, 0].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# --- Graphique 2 : Top 10 des Catégories par CA ---
cat_ca = df.groupby('categorie')['prix_total'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=cat_ca.values, y=cat_ca.index, ax=axes[0, 1], palette="viridis")
axes[0, 1].set_title("Top 10 des Catégories générant le plus de CA ($)", fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel("Chiffre d'affaires ($)")

# --- Graphique 3 : Profil Horaire des Achats ---
hourly_sales = df.groupby('heure_num')['prix_total'].sum()
sns.lineplot(x=hourly_sales.index, y=hourly_sales.values, ax=axes[1, 0], marker='o', color='crimson', linewidth=2.5)
axes[1, 0].set_title("Répartition du Chiffre d'Affaires par Heure de la journée", fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel("Heure de la journée")
axes[1, 0].set_ylabel("CA Total ($)")
axes[1, 0].set_xticks(range(7, 22))

# --- Graphique 4 : Répartition des Achats par Jour de Semaine ---
jours_ordre = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
jour_sales = df.groupby('jour_semaine')['prix_total'].sum().reindex(jours_ordre)
sns.barplot(x=jour_sales.index, y=jour_sales.values, ax=axes[1, 1], palette="Blues_d")
axes[1, 1].set_title("Chiffre d'Affaires par Jour de la Semaine", fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel("Jour")
axes[1, 1].set_ylabel("CA Total ($)")
axes[1, 1].tick_params(axis='x', rotation=30)

# --- Graphique 5 : Impact des Promotions ---
promo_impact = df.groupby('promotion').agg(
    prix_unitaire_moyen=('prix_unitaire', 'mean'),
    quantite_moyenne=('quantite', 'mean'),
    volume_lignes=('produit_id', 'count')
).reset_index()
promo_impact['promotion'] = promo_impact['promotion'].map({0: 'Hors Promo', 1: 'En Promo'})

sns.barplot(data=promo_impact, x='promotion', y='quantite_moyenne', ax=axes[2, 0], palette="Set2")
axes[2, 0].set_title("Quantité moyenne achetée : Hors Promo vs En Promo", fontsize=12, fontweight='bold')
axes[2, 0].set_xlabel("Statut Promotionnel")
axes[2, 0].set_ylabel("Quantité unitaire moyenne")
for p in axes[2, 0].patches:
    axes[2, 0].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# --- Graphique 6 : Fréquentation des Rayons (Nombre d'articles vendus) ---
rayon_villes = df.groupby('rayon')['quantite'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=rayon_villes.values, y=rayon_villes.index, ax=axes[2, 1], palette="cubehelix")
axes[2, 1].set_title("Top 10 des Rayons les plus fréquentés (Volume d'unités)", fontsize=12, fontweight='bold')
axes[2, 1].set_xlabel("Quantités vendues")

plt.tight_layout()
plt.savefig("eda_supermarket_dashboard.png", dpi=300)
print("\n Graphiques enregistrés sous le nom 'eda_supermarket_dashboard.png'")
plt.show()