import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)


# Adapter le chemin si besoin
df = pd.read_csv("/Users/jeannegautier/Projets/AI-projet/data/base-etablissements.csv")

print("Shape initiale :", df.shape)
print(df.head())
print(df.info())


df_clean = df.copy()


empty_cols = df_clean.columns[df_clean.isna().mean() == 1].tolist()

print("Colonnes 100% vides supprimées :")
print(empty_cols)

df_clean = df_clean.drop(columns=empty_cols)

print("Shape après suppression des colonnes vides :", df_clean.shape)


date_cols = [
    "updatedAt",
    "ehpadPrice.updatedAt",
    "raPrice.updatedAt"
]

for col in date_cols:
    if col in df_clean.columns:
        df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")

print("Conversion des dates terminée.")


id_cols = [
    "noFinesset",
    "coordinates.postcode",
    "coordinates.deptcode"
]

for col in id_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].astype("string")

print("Identifiants et codes géographiques convertis en chaînes.")


target = "IsHAB_AIDE_SOC"

print("Répartition de la cible :")
print(df_clean[target].value_counts(dropna=False))

print("Répartition normalisée de la cible :")
print(df_clean[target].value_counts(normalize=True, dropna=False))

# Conversion de la cible en 0/1
df_clean[target] = df_clean[target].astype(int)


# Plot : distribution de la cible
plt.figure(figsize=(6, 4))
df_clean[target].value_counts().sort_index().plot(kind="bar")
plt.title("Répartition de la cible : habilitation à l'aide sociale")
plt.xlabel("IsHAB_AIDE_SOC")
plt.ylabel("Nombre d'établissements")
plt.xticks([0, 1], ["Non habilité", "Habilité"], rotation=0)
plt.tight_layout()
plt.show()


if "noFinesset" in df_clean.columns:
    nb_duplicates = df_clean["noFinesset"].duplicated().sum()
    print("Nombre de doublons sur noFinesset :", nb_duplicates)

    if nb_duplicates > 0:
        print(
            df_clean[df_clean["noFinesset"].duplicated(keep=False)]
            [["noFinesset", "title", "coordinates.city", "IsEHPAD", "IsRA", "IsESLD", target]]
            .sort_values("noFinesset")
            .head(30)
        )


def get_main_type(row):
    if row.get("IsEHPAD", False):
        return "EHPAD"
    elif row.get("IsESLD", False):
        return "ESLD"
    elif row.get("IsRA", False):
        return "Résidence autonomie"
    elif row.get("IsAJA", False):
        return "Accueil de jour"
    elif row.get("IsEHPA", False):
        return "EHPA"
    else:
        return "Autre"

df_clean["type_principal"] = df_clean.apply(get_main_type, axis=1)

print("Répartition par type principal :")
print(df_clean["type_principal"].value_counts())


# Plot : type principal
plt.figure(figsize=(8, 5))
df_clean["type_principal"].value_counts().plot(kind="bar")
plt.title("Répartition des établissements par type principal")
plt.xlabel("Type d'établissement")
plt.ylabel("Nombre d'établissements")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()



missing_rate = df_clean.isna().mean().sort_values(ascending=False)

print("Colonnes avec le plus de valeurs manquantes :")
print(missing_rate.head(30))


# Plot : top 20 valeurs manquantes
plt.figure(figsize=(10, 6))
missing_rate.head(20).sort_values().plot(kind="barh")
plt.title("Top 20 des colonnes avec le plus de valeurs manquantes")
plt.xlabel("Taux de valeurs manquantes")
plt.ylabel("Colonnes")
plt.tight_layout()
plt.show()


type_cols = ["IsEHPAD", "IsEHPA", "IsESLD", "IsRA", "IsAJA"]
type_cols = [col for col in type_cols if col in df_clean.columns]

print("Nombre d'établissements par type booléen :")
print(df_clean[type_cols].sum().sort_values(ascending=False))


# Missing prix EHPAD chez les EHPAD
if "IsEHPAD" in df_clean.columns and "ehpadPrice.prixHebPermCs" in df_clean.columns:
    ehpad = df_clean[df_clean["IsEHPAD"] == True]
    print("Taux de prix chambre simple permanent manquant chez les EHPAD :")
    print(ehpad["ehpadPrice.prixHebPermCs"].isna().mean())


# Missing prix RA chez les résidences autonomie
ra_price_base_cols = ["raPrice.PrixF1", "raPrice.PrixF1Bis", "raPrice.PrixF2"]
ra_price_base_cols = [col for col in ra_price_base_cols if col in df_clean.columns]

if "IsRA" in df_clean.columns and len(ra_price_base_cols) > 0:
    ra = df_clean[df_clean["IsRA"] == True]
    print("Taux de valeurs manquantes des prix RA chez les résidences autonomie :")
    print(ra[ra_price_base_cols].isna().mean())



# Capacité
if "capacity" in df_clean.columns:
    plt.figure(figsize=(8, 5))
    df_clean["capacity"].dropna().plot(kind="hist", bins=40)
    plt.title("Distribution de la capacité des établissements")
    plt.xlabel("Capacité")
    plt.ylabel("Nombre d'établissements")
    plt.tight_layout()
    plt.show()


# Statut juridique
if "legal_status" in df_clean.columns:
    plt.figure(figsize=(8, 5))
    df_clean["legal_status"].value_counts(dropna=False).plot(kind="bar")
    plt.title("Répartition des établissements par statut juridique")
    plt.xlabel("Statut juridique")
    plt.ylabel("Nombre d'établissements")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# Prix minimum
if "prixMin" in df_clean.columns:
    plt.figure(figsize=(8, 5))
    df_clean["prixMin"].dropna().plot(kind="hist", bins=40)
    plt.title("Distribution du prix minimum")
    plt.xlabel("Prix minimum")
    plt.ylabel("Nombre d'établissements")
    plt.tight_layout()
    plt.show()


# Prix minimum par type d'établissement
if "prixMin" in df_clean.columns:
    plt.figure(figsize=(10, 6))
    df_clean.dropna(subset=["prixMin"]).boxplot(
        column="prixMin",
        by="type_principal",
        rot=45
    )
    plt.title("Prix minimum par type d'établissement")
    plt.suptitle("")
    plt.xlabel("Type principal")
    plt.ylabel("Prix minimum")
    plt.tight_layout()
    plt.show()


# Taux d'habilitation par type principal
habilitation_by_type = (
    df_clean.groupby("type_principal")[target]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))
habilitation_by_type.plot(kind="bar")
plt.title("Taux d'habilitation à l'aide sociale par type d'établissement")
plt.xlabel("Type d'établissement")
plt.ylabel("Taux d'habilitation")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# Taux d'habilitation par statut juridique
if "legal_status" in df_clean.columns:
    habilitation_by_status = (
        df_clean.groupby("legal_status")[target]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))
    habilitation_by_status.plot(kind="bar")
    plt.title("Taux d'habilitation à l'aide sociale par statut juridique")
    plt.xlabel("Statut juridique")
    plt.ylabel("Taux d'habilitation")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# Nombre de services spécialisés
service_cols = [
    "IsALZH", "IsUHR", "IsPASA", "IsPUV",
    "IsACC_JOUR", "IsACC_NUIT",
    "IsHCOMPL", "IsHTEMPO"
]

service_cols = [col for col in service_cols if col in df_clean.columns]

df_clean["nb_services_specialises"] = df_clean[service_cols].sum(axis=1)


# Indicateur APL uniquement
# Attention : on ne met pas IsHAB_AIDE_SOC dans ce score car c'est la cible.
if "IsCONV_APL" in df_clean.columns:
    df_clean["aide_logement_apl"] = df_clean["IsCONV_APL"].astype(int)


# Complétude des prix EHPAD
ehpad_price_cols = [
    "ehpadPrice.prixHebPermCs",
    "ehpadPrice.prixHebPermCd",
    "ehpadPrice.prixHebPermCsa",
    "ehpadPrice.prixHebPermCda",
    "ehpadPrice.prixHebTempCs",
    "ehpadPrice.prixHebTempCd",
    "ehpadPrice.prixHebTempCsa",
    "ehpadPrice.prixHebTempCda",
    "ehpadPrice.tarifGir12",
    "ehpadPrice.tarifGir34",
    "ehpadPrice.tarifGir56"
]

ehpad_price_cols = [col for col in ehpad_price_cols if col in df_clean.columns]

if len(ehpad_price_cols) > 0:
    df_clean["nb_prix_ehpad_renseignes"] = df_clean[ehpad_price_cols].notna().sum(axis=1)


# Complétude des prix RA
ra_price_cols = [
    "raPrice.PrixF1",
    "raPrice.PrixF1ASH",
    "raPrice.PrixF1Bis",
    "raPrice.PrixF1BisASH",
    "raPrice.PrixF2",
    "raPrice.PrixF2ASH"
]

ra_price_cols = [col for col in ra_price_cols if col in df_clean.columns]

if len(ra_price_cols) > 0:
    df_clean["nb_prix_ra_renseignes"] = df_clean[ra_price_cols].notna().sum(axis=1)


# Prix relatif au département
if "prixMin" in df_clean.columns and "coordinates.deptcode" in df_clean.columns:
    df_clean["prix_median_dept"] = (
        df_clean.groupby("coordinates.deptcode")["prixMin"]
        .transform("median")
    )

    df_clean["ecart_prix_dept"] = df_clean["prixMin"] - df_clean["prix_median_dept"]

    df_clean["ratio_prix_dept"] = (
        df_clean["prixMin"] / df_clean["prix_median_dept"]
    )

    df_clean["ratio_prix_dept"] = df_clean["ratio_prix_dept"].replace(
        [np.inf, -np.inf],
        np.nan
    )



cols_to_drop = [
    "_id",
    "title",
    "noFinesset",
    "coordinates.street",
    "coordinates.phone",
    "coordinates.emailContact",
    "coordinates.website",
    "coordinates.gestionnaire",
    "ehpadPrice.autrePrestation",
    "ehpadPrice.autreTarifPrest",
    "raPrice.autreTarifPrest",
    "raPrice.prestObligatoire"
]

cols_to_drop = [col for col in cols_to_drop if col in df_clean.columns]

df_model = df_clean.drop(columns=cols_to_drop)

print("Colonnes supprimées avant modélisation :")
print(cols_to_drop)

print("Shape du dataset nettoyé :", df_model.shape)



print("Infos finales :")
print(df_model.info())

print("Valeurs manquantes restantes :")
print(df_model.isna().mean().sort_values(ascending=False).head(30))

print("Répartition finale de la cible :")
print(df_model[target].value_counts(normalize=True))


# Croisements utiles pour l'interprétation métier
if "legal_status" in df_model.columns:
    print("Taux d'habilitation par statut juridique :")
    print(pd.crosstab(df_model["legal_status"], df_model[target], normalize="index"))

print("Taux d'habilitation par type principal :")
print(pd.crosstab(df_model["type_principal"], df_model[target], normalize="index"))



X = df_model.drop(columns=[target])
y = df_model[target]

print("Shape X :", X.shape)
print("Shape y :", y.shape)



df_model.to_csv("/Users/jeannegautier/Projets/AI-projet/data/etablissements_clean_supervised.csv", index=False)

print("Dataset nettoyé sauvegardé sous : etablissements_clean_supervised.csv")