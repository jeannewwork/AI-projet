import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay
)


path = "/Users/jeannegautier/Projets/AI-projet/data/etablissements_clean_supervised.csv"

df = pd.read_csv(path, low_memory=False)

print("Shape du dataset :", df.shape)
print(df.head())


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
    "coordinates.city"
]

cols_to_drop = [col for col in cols_to_drop if col in X.columns]

X = X.drop(columns=cols_to_drop)

print("Colonnes supprimées :")
print(cols_to_drop)

print("Shape X après suppression :", X.shape)



bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()

for col in bool_cols:
    X[col] = X[col].astype(int)

print("Colonnes booléennes converties :", bool_cols)



numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()

print("Nombre de variables numériques :", len(numeric_features))
print(numeric_features)

print("Nombre de variables catégorielles :", len(categorical_features))
print(categorical_features)



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
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
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


logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            solver="liblinear",
            random_state=42
        ))
    ]
)


logistic_model.fit(X_train, y_train)

print("Modèle entraîné.")



y_pred = logistic_model.predict(X_test)
y_proba = logistic_model.predict_proba(X_test)[:, 1]



print("Classification report :")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_proba)
print("ROC AUC :", round(auc, 4))



cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non habilité", "Habilité"]
)

disp.plot()
plt.title("Matrice de confusion - Régression logistique")
plt.tight_layout()
plt.show()



RocCurveDisplay.from_estimator(
    logistic_model,
    X_test,
    y_test
)

plt.title("Courbe ROC - Régression logistique")
plt.tight_layout()
plt.show()



# Récupération des noms de variables après OneHotEncoding
preprocessor_fitted = logistic_model.named_steps["preprocessor"]

num_names = numeric_features

cat_encoder = preprocessor_fitted.named_transformers_["cat"].named_steps["onehot"]
cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()

feature_names = num_names + cat_names

coefficients = logistic_model.named_steps["model"].coef_[0]

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
})

coef_df["abs_coefficient"] = coef_df["coefficient"].abs()

coef_df = coef_df.sort_values("abs_coefficient", ascending=False)

print("Top 20 variables les plus influentes :")
print(coef_df.head(20))



top_coef = coef_df.head(20).sort_values("coefficient")

plt.figure(figsize=(10, 7))
plt.barh(top_coef["feature"], top_coef["coefficient"])
plt.title("Top 20 coefficients - Régression logistique")
plt.xlabel("Coefficient")
plt.ylabel("Variable")
plt.tight_layout()
plt.show()



coef_df.to_csv(
    "/Users/jeannegautier/Projets/AI-projet/data/logistic_regression_coefficients.csv",
    index=False
)

print("Coefficients sauvegardés dans logistic_regression_coefficients.csv")