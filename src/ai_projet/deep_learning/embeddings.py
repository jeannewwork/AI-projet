import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


TARGET = "IsHAB_AIDE_SOC"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_text_dataset(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    if "text" not in df.columns:
        raise ValueError("Le fichier doit contenir une colonne 'text'.")

    if TARGET not in df.columns:
        raise ValueError(f"Le fichier doit contenir la cible '{TARGET}'.")

    df = df.dropna(subset=["text", TARGET]).copy()
    df["text"] = df["text"].astype(str)
    df[TARGET] = df[TARGET].astype(int)

    return df


def compute_embeddings(texts, model_name: str, batch_size: int):
    model = SentenceTransformer(model_name, device="cpu")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings


def save_embeddings(output_dir, X_train_emb, X_test_emb, y_train, y_test, texts_train, texts_test):
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X_train_embeddings.npy", X_train_emb)
    np.save(output_dir / "X_test_embeddings.npy", X_test_emb)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_test.npy", y_test)

    pd.DataFrame({"text": texts_train, TARGET: y_train}).to_csv(
        output_dir / "train_texts.csv",
        index=False,
    )

    pd.DataFrame({"text": texts_test, TARGET: y_test}).to_csv(
        output_dir / "test_texts.csv",
        index=False,
    )


def plot_embeddings_pca(X_train_emb, X_test_emb, y_train, y_test, output_dir):
    X_all = np.vstack([X_train_emb, X_test_emb])
    y_all = np.concatenate([y_train, y_test])

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_all)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_all, alpha=0.6)
    plt.title("Projection PCA des embeddings textuels")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(label=TARGET)
    plt.tight_layout()

    output_path = output_dir / "embeddings_pca_projection.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    return pca.explained_variance_ratio_


def get_models():
    return {
        "logistic_regression_embeddings": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest_embeddings": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "svm_rbf_embeddings": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42,
        ),
    }


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "model": name,
        "feature_type": "text_embeddings",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_0": precision_score(y_test, y_pred, pos_label=0),
        "recall_0": recall_score(y_test, y_pred, pos_label=0),
        "f1_0": f1_score(y_test, y_pred, pos_label=0),
        "precision_1": precision_score(y_test, y_pred, pos_label=1),
        "recall_1": recall_score(y_test, y_pred, pos_label=1),
        "f1_1": f1_score(y_test, y_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def train_and_evaluate_models(X_train_emb, X_test_emb, y_train, y_test):
    results = []
    trained_models = {}

    for name, model in get_models().items():
        print(f"\nTraining: {name}")

        model.fit(X_train_emb, y_train)

        result = evaluate_model(name, model, X_test_emb, y_test)
        results.append(result)
        trained_models[name] = model

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("roc_auc", ascending=False)

    return results_df, trained_models


def plot_best_model(best_model, best_model_name, X_test_emb, y_test, output_dir):
    y_pred = best_model.predict(X_test_emb)

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non habilité", "Habilité"],
    )

    disp.plot()
    plt.title(f"Matrice de confusion - {best_model_name}")
    plt.tight_layout()
    plt.savefig(output_dir / "best_embedding_confusion_matrix.png", dpi=300)
    plt.close()

    RocCurveDisplay.from_estimator(best_model, X_test_emb, y_test)
    plt.title(f"Courbe ROC - {best_model_name}")
    plt.tight_layout()
    plt.savefig(output_dir / "best_embedding_roc_curve.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default="/users/local/j22gauti/AI-projet/data/etablissements_texts.csv",
        help="Chemin vers le dataset texte.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/users/local/j22gauti/AI-projet/data/embeddings",
        help="Dossier où sauvegarder les embeddings et résultats.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size pour l'encodage des textes.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    print("Chargement du dataset texte...")
    df = load_text_dataset(input_path)

    texts = df["text"].tolist()
    y = df[TARGET].values

    print("Shape dataset :", df.shape)
    print("Répartition cible :")
    print(pd.Series(y).value_counts(normalize=True))

    texts_train, texts_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nExtraction des embeddings...")

    X_train_emb = compute_embeddings(
        texts_train,
        model_name=MODEL_NAME,
        batch_size=args.batch_size,
    )

    X_test_emb = compute_embeddings(
        texts_test,
        model_name=MODEL_NAME,
        batch_size=args.batch_size,
    )

    print("Shape X_train embeddings :", X_train_emb.shape)
    print("Shape X_test embeddings :", X_test_emb.shape)

    print("\nSauvegarde des embeddings...")
    save_embeddings(
        output_dir,
        X_train_emb,
        X_test_emb,
        y_train,
        y_test,
        texts_train,
        texts_test,
    )

    print("\nVisualisation PCA des embeddings...")
    explained_variance = plot_embeddings_pca(
        X_train_emb,
        X_test_emb,
        y_train,
        y_test,
        output_dir,
    )

    print("Variance expliquée par PCA :", explained_variance)

    print("\nEntraînement des modèles sur embeddings...")
    results_df, trained_models = train_and_evaluate_models(
        X_train_emb,
        X_test_emb,
        y_train,
        y_test,
    )

    results_path = output_dir / "embedding_model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    print("\nRésultats embeddings :")
    print(results_df)

    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    print("\nMeilleur modèle embeddings :", best_model_name)
    print("\nClassification report :")
    print(classification_report(y_test, best_model.predict(X_test_emb)))

    plot_best_model(
        best_model,
        best_model_name,
        X_test_emb,
        y_test,
        output_dir,
    )

    print("\nFichiers sauvegardés dans :", output_dir)
    print("- X_train_embeddings.npy")
    print("- X_test_embeddings.npy")
    print("- embedding_model_comparison.csv")
    print("- embeddings_pca_projection.png")
    print("- best_embedding_confusion_matrix.png")
    print("- best_embedding_roc_curve.png")


if __name__ == "__main__":
    main()