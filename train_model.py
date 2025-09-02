# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

# Charger dataset (mettre le vrai chemin vers application_train.csv)
df = pd.read_csv("data/application_train.csv")

# Séparer features / target
X = df.drop(columns=["TARGET"])
y = df["TARGET"]

# Identifier colonnes numériques et catégorielles
categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

# Préprocesseur
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols)
])

# Pipeline complet
pipeline = Pipeline([
    ("preproc", preprocessor),
    ("model", XGBClassifier(use_label_encoder=False, eval_metric="logloss"))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner pipeline
pipeline.fit(X_train, y_train)

# Créer dossier models si inexistant
os.makedirs("models", exist_ok=True)

# Sauvegarder pipeline complet
joblib.dump(pipeline, "models/pipeline.pkl")
print("✅ Pipeline enregistré dans models/pipeline.pkl")
