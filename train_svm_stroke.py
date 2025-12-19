# train_svm_stroke.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
import joblib
import sklearn

print("sklearn version:", sklearn.__version__)

# Paths
DATA_PATH = "dataset/stroke_data.csv"   # chỉnh nếu file ở nơi khác
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "svm_stroke.pkl"

# Load data
df = pd.read_csv(DATA_PATH)
print("Loaded dataset:", df.shape)
print("Columns:", df.columns.tolist())

# Required columns (adjust tên nếu khác)
FEATURES = ["Age","Gender","SES","Hypertension","Heart_Disease","BMI","Avg_Glucose","Diabetes","Smoking_Status"]
TARGET = "Stroke"

for c in FEATURES + [TARGET]:
    if c not in df.columns:
        raise KeyError(f"Không thấy cột `{c}` trong dataset. Có trong file: {df.columns.tolist()}")

X = df[FEATURES].copy()
y = df[TARGET].astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Numeric pipeline
numeric_features = ["Age","BMI","Avg_Glucose"]
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Ordinal for SES
ordinal_features = ["SES"]
ordinal_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ord", OrdinalEncoder(categories=[['Low','Medium','High']]))
])

# One-hot for Gender and Smoking_Status with sklearn-version compatibility
ohe_kwargs = {"drop": "first"}
# try using sparse_output for newer sklearn, otherwise sparse
try:
    OneHotEncoder(sparse_output=False)  # test signature
    ohe_kwargs["sparse_output"] = False
except TypeError:
    ohe_kwargs["sparse"] = False

ohe_features = ["Gender","Smoking_Status"]
ohe_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(**ohe_kwargs, dtype=int))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("ord", ordinal_transformer, ordinal_features),
    ("ohe", ohe_transformer, ohe_features)
], remainder="drop")

# SVM classifier (RBF) with probability True
clf = Pipeline([
    ("pre", preprocessor),
    ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42))
])

# Fit
print("Fitting SVM pipeline...")
clf.fit(X_train, y_train)

# Predict & evaluate
y_pred = clf.predict(X_test)
try:
    y_proba = clf.predict_proba(X_test)[:,1]
except Exception:
    # fallback: use decision_function and sigmoid
    try:
        from scipy.special import expit
        z = clf.decision_function(X_test)
        y_proba = expit(z)
    except Exception:
        y_proba = np.zeros_like(y_pred, dtype=float)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
try:
    auc = roc_auc_score(y_test, y_proba)
    print("ROC AUC:", auc)
except Exception:
    print("ROC AUC: cannot compute (no proba)")

print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(clf, MODEL_PATH)
print(f"\nSaved SVM pipeline to {MODEL_PATH}")

# print sample transformed features and categories for inspection
try:
    pre = clf.named_steps["pre"]
    # attempt to show onehot categories if present
    if "ohe" in pre.named_transformers_:
        ohe_pipe = pre.named_transformers_["ohe"]
        if hasattr(ohe_pipe, "named_steps") and "onehot" in ohe_pipe.named_steps:
            ohe = ohe_pipe.named_steps["onehot"]
            try:
                print("OneHot categories:", [list(c) for c in ohe.categories_])
            except Exception:
                pass
    if "ord" in pre.named_transformers_:
        ord_pipe = pre.named_transformers_["ord"]
        if hasattr(ord_pipe, "named_steps") and "ord" in ord_pipe.named_steps:
            ordenc = ord_pipe.named_steps["ord"]
            try:
                print("Ordinal categories:", [list(c) for c in ordenc.categories_])
            except Exception:
                pass
except Exception:
    pass
