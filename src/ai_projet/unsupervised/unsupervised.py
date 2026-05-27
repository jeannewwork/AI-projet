import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


TARGET = "IsHAB_AIDE_SOC"

INPUT_PATH = "/Users/jeannegautier/Projets/AI-projet/data/etablissements_clean_supervised.csv"
OUTPUT_DIR = Path("/Users/jeannegautier/Projets/AI-projet/data/unsupervised")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("Chargement du dataset...")
df = pd.read_csv(INPUT_PATH)

print("Shape :", df.shape)

y = df[TARGET].astype(int)
X = df.drop(columns=[TARGET])


# =========================
# Suppression des dates
# =========================

date_like_cols = [
    col for col in X.columns
    if "date" in col.lower()
    or "updatedat" in col.lower()
    or "createdat" in col.lower()
]

print("Colonnes de dates supprimées :", date_like_cols)
X = X.drop(columns=date_like_cols)


# =========================
# Séparation types colonnes
# =========================

numeric_cols = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_cols = [
    col for col in X.columns
    if col not in numeric_cols
]

print("Colonnes numériques :", len(numeric_cols))
print("Colonnes catégorielles :", len(categorical_cols))

for col in categorical_cols:
    X[col] = X[col].astype(str)


# =========================
# Preprocessing
# =========================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        ),
    ),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols),
])

print("\nPreprocessing...")
X_processed = preprocessor.fit_transform(X)
X_processed = np.asarray(X_processed)

print("Shape après preprocessing :", X_processed.shape)


# =========================
# PCA
# =========================

print("\nRéduction PCA...")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_processed)

print("Variance expliquée :", pca.explained_variance_ratio_)
print("Variance totale :", pca.explained_variance_ratio_.sum())


plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    alpha=0.6,
)

plt.title("Projection PCA du dataset colorée par la cible")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(scatter, label=TARGET)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pca_projection_labels.png", dpi=300)
plt.close()


# =========================
# Fonctions
# =========================

def evaluate_clustering(name, labels, X_data, y_true):
    labels = np.asarray(labels)

    unique_labels = set(labels)
    n_clusters = len(unique_labels)

    if -1 in unique_labels:
        n_clusters -= 1

    result = {
        "algorithm": name,
        "n_clusters": n_clusters,
        "n_noise": int(np.sum(labels == -1)) if -1 in unique_labels else 0,
    }

    if n_clusters > 1:
        result["silhouette"] = silhouette_score(X_data, labels)
        result["calinski_harabasz"] = calinski_harabasz_score(X_data, labels)
        result["davies_bouldin"] = davies_bouldin_score(X_data, labels)
    else:
        result["silhouette"] = np.nan
        result["calinski_harabasz"] = np.nan
        result["davies_bouldin"] = np.nan

    result["adjusted_rand_index"] = adjusted_rand_score(y_true, labels)
    result["normalized_mutual_info"] = normalized_mutual_info_score(y_true, labels)

    return result


def plot_clusters(X_pca, labels, title, output_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        alpha=0.6,
    )

    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


results = []


# =========================
# KMeans
# =========================

print("\nKMeans...")

for k in [2, 3, 5, 10]:
    print(f"KMeans avec k={k}")

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init="auto",
    )

    labels = kmeans.fit_predict(X_pca)

    results.append(
        evaluate_clustering(
            f"KMeans_k{k}",
            labels,
            X_pca,
            y,
        )
    )

    plot_clusters(
        X_pca,
        labels,
        f"Clusters KMeans - k={k}",
        OUTPUT_DIR / f"kmeans_k{k}_clusters.png",
    )


# =========================
# Agglomerative Clustering
# =========================

print("\nAgglomerative clustering...")

for k in [2, 3, 5]:
    print(f"Agglomerative avec k={k}")

    agglo = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward",
    )

    labels = agglo.fit_predict(X_pca)

    results.append(
        evaluate_clustering(
            f"Agglomerative_k{k}",
            labels,
            X_pca,
            y,
        )
    )

    plot_clusters(
        X_pca,
        labels,
        f"Clusters Agglomerative - k={k}",
        OUTPUT_DIR / f"agglomerative_k{k}_clusters.png",
    )


# =========================
# DBSCAN
# =========================

print("\nDBSCAN...")

dbscan_configs = [
    {"eps": 0.5, "min_samples": 10},
    {"eps": 1.0, "min_samples": 10},
    {"eps": 1.5, "min_samples": 10},
]

for config in dbscan_configs:
    eps = config["eps"]
    min_samples = config["min_samples"]

    print(f"DBSCAN avec eps={eps}, min_samples={min_samples}")

    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
    )

    labels = dbscan.fit_predict(X_pca)

    results.append(
        evaluate_clustering(
            f"DBSCAN_eps{eps}_min{min_samples}",
            labels,
            X_pca,
            y,
        )
    )

    plot_clusters(
        X_pca,
        labels,
        f"Clusters DBSCAN - eps={eps}",
        OUTPUT_DIR / f"dbscan_eps{eps}_clusters.png",
    )


# =========================
# Tableau final
# =========================

results_df = pd.DataFrame(results)

print("\nRésultats clustering :")
print(results_df)

results_df.to_csv(
    OUTPUT_DIR / "clustering_results.csv",
    index=False,
)

print("\nFichiers sauvegardés dans :", OUTPUT_DIR)
print("- pca_projection_labels.png")
print("- kmeans_k*_clusters.png")
print("- agglomerative_k*_clusters.png")
print("- dbscan_eps*_clusters.png")
print("- clustering_results.csv")