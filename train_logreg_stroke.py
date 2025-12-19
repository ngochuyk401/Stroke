# train_logreg_stroke.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
import joblib

DATA_PATH = "dataset/stroke_data.csv"   # đặt file stroke_data.csv cùng thư mục với script, hoặc chỉnh đường dẫn
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "logreg_stroke.pkl"

# Load dataset
df = pd.read_csv(DATA_PATH)
print("Loaded dataset:", df.shape)
print("Columns:", df.columns.tolist())

# Features & target (adjust nếu tên cột khác)
FEATURES = ["Age","Gender","SES","Hypertension","Heart_Disease","BMI","Avg_Glucose","Diabetes","Smoking_Status"]
TARGET = "Stroke"

for f in FEATURES + [TARGET]:
    if f not in df.columns:
        raise KeyError(f"Không thấy cột `{f}` trong dataset. Kiểm tra tên cột (in hoa/thường).")

X = df[FEATURES].copy()
y = df[TARGET].astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Preprocessing
numeric_features = ["Age","BMI","Avg_Glucose"]
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# SES ordinal (Low<Medium<High)
ordinal_features = ["SES"]
ordinal_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ord", OrdinalEncoder(categories=[['Low','Medium','High']], dtype=float))
])

# One-hot for Gender, Smoking_Status
ohe_features = ["Gender","Smoking_Status"]
ohe_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop='first', sparse=False, dtype=int))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("ord", ordinal_transformer, ordinal_features),
    ("ohe", ohe_transformer, ohe_features)
], remainder="drop")

clf = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

# Fit
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(clf, MODEL_PATH)
print(f"Saved model pipeline to {MODEL_PATH}")
