import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


path = "/Users/jeannegautier/Projets/AI-projet/data/etablissements_clean_supervised.csv"

df = pd.read_csv(path, low_memory=False)

print("Shape du dataset :", df.shape)



target = "IsHAB_AIDE_SOC"

y = df[target]
X = df.drop(columns=[target])

print("Répartition de la cible :")
print(y.value_counts(normalize=True))


cols_to_drop = [
    "updatedAt",
    "ehpadPrice.updatedAt",
    "raPrice.updatedAt",
    "ehpadPrice._id",
    "raPrice._id",
    "coordinates.postcode",
    "coordinates.city",
    "coordinates.deptname"
]

cols_to_drop = [col for col in cols_to_drop if col in X.columns]
X = X.drop(columns=cols_to_drop)

print("Colonnes supprimées :")
print(cols_to_drop)

print("Shape X :", X.shape)


bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()

for col in bool_cols:
    X[col] = X[col].astype(int)

print("Colonnes booléennes converties :", bool_cols)



numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()

print("Nombre de variables numériques :", len(numeric_features))
print("Nombre de variables catégorielles :", len(categorical_features))
print("Variables catégorielles :", categorical_features)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Shape X_train :", X_train.shape)
print("Shape X_test :", X_test.shape)



numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)



configs = [
    {
        "name": "rf_current",
        "n_estimators": 150,
        "max_depth": 14,
        "min_samples_leaf": 5,
        "min_samples_split": 10,
        "class_weight": "balanced"
    },
    {
        "name": "rf_deeper",
        "n_estimators": 150,
        "max_depth": 18,
        "min_samples_leaf": 4,
        "min_samples_split": 8,
        "class_weight": "balanced"
    },
    {
        "name": "rf_no_class_weight",
        "n_estimators": 150,
        "max_depth": 14,
        "min_samples_leaf": 5,
        "min_samples_split": 10,
        "class_weight": None
    },
    {
        "name": "rf_small_leaf",
        "n_estimators": 180,
        "max_depth": 16,
        "min_samples_leaf": 3,
        "min_samples_split": 8,
        "class_weight": "balanced"
    }
]



results = []
trained_models = {}

for config in configs:
    print("\n====================================")
    print("Training :", config["name"])
    print("====================================")

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                min_samples_leaf=config["min_samples_leaf"],
                min_samples_split=config["min_samples_split"],
                max_features="sqrt",
                class_weight=config["class_weight"],
                random_state=42,
                n_jobs=-1
            ))
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "model": config["name"],
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_0": precision_score(y_test, y_pred, pos_label=0),
        "recall_0": recall_score(y_test, y_pred, pos_label=0),
        "f1_0": f1_score(y_test, y_pred, pos_label=0),
        "precision_1": precision_score(y_test, y_pred, pos_label=1),
        "recall_1": recall_score(y_test, y_pred, pos_label=1),
        "f1_1": f1_score(y_test, y_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_test, y_proba)
    })

    trained_models[config["name"]] = model


results_df = pd.DataFrame(results)
results_df = results_df.sort_values("roc_auc", ascending=False)

print("\nRésultats comparés :")
print(results_df)



best_model_name = results_df.iloc[0]["model"]
best_model = trained_models[best_model_name]

print("\nMeilleur modèle selon ROC AUC :", best_model_name)


y_pred_best = best_model.predict(X_test)
y_proba_best = best_model.predict_proba(X_test)[:, 1]

print("\nClassification report du meilleur modèle :")
print(classification_report(y_test, y_pred_best))

print("ROC AUC du meilleur modèle :", round(roc_auc_score(y_test, y_proba_best), 4))



cm = confusion_matrix(y_test, y_pred_best)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non habilité", "Habilité"]
)

disp.plot()
plt.title(f"Matrice de confusion - {best_model_name}")
plt.tight_layout()
plt.show()



RocCurveDisplay.from_estimator(
    best_model,
    X_test,
    y_test
)

plt.title(f"Courbe ROC - {best_model_name}")
plt.tight_layout()
plt.show()



metrics_to_plot = ["accuracy", "f1_0", "f1_1", "roc_auc"]

results_plot = results_df.set_index("model")[metrics_to_plot]

results_plot.plot(kind="bar", figsize=(10, 6))
plt.title("Comparaison des configurations Random Forest")
plt.ylabel("Score")
plt.xlabel("Modèle")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


preprocessor_fitted = best_model.named_steps["preprocessor"]

num_names = numeric_features

cat_encoder = preprocessor_fitted.named_transformers_["cat"].named_steps["onehot"]
cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()

feature_names = num_names + cat_names

importances = best_model.named_steps["model"].feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

importance_df = importance_df.sort_values("importance", ascending=False)

print("\nTop 20 variables les plus importantes :")
print(importance_df.head(20))



top_importance = importance_df.head(20).sort_values("importance")

plt.figure(figsize=(10, 7))
plt.barh(top_importance["feature"], top_importance["importance"])
plt.title(f"Top 20 variables importantes - {best_model_name}")
plt.xlabel("Importance")
plt.ylabel("Variable")
plt.tight_layout()
plt.show()



results_df.to_csv(
    "/Users/jeannegautier/Projets/AI-projet/data/random_forest_config_comparison.csv",
    index=False
)

importance_df.to_csv(
    "/Users/jeannegautier/Projets/AI-projet/data/random_forest_best_importances.csv",
    index=False
)

print("\nRésultats sauvegardés :")
print("- random_forest_config_comparison.csv")
print("- random_forest_best_importances.csv")