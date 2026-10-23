import pandas as pd

# Charger votre dataset (ajustez le nom du fichier)
df = pd.read_csv("supermarket_sales_100k.csv")

# Afficher la structure des colonnes
print("--- INFORMATIONS ET TYPES ---")
print(df.info())

# Afficher les 5 premières lignes
print("\n--- APERÇU DES 5 PREMIÈRES LIGNES ---")
print(df.head())