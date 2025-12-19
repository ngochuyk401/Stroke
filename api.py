from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import pandas as pd
import numpy as np
import sklearn, joblib, sys

from model_util import (
    list_models, load_model, to_dataframe_one, to_dataframe_batch, predict_one, FEATURES
)

app = FastAPI(
    title="Stroke Risk API",
    description="API dự đoán nguy cơ đột quỵ (Logistic Regression / RandomForest / Pipeline).",
    version="1.0.0",
)

# CORS cho dev/local; tùy chỉnh theo môi trường triển khai
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hạn chế domain thật khi deploy
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================ Schemas (Pydantic v2) ============================

class Patient(BaseModel):
    """
    Schema cho 1 bệnh nhân dùng input cho model (không bao gồm target 'Stroke').
    Các trường khớp với file stroke_data.csv:
      - Age: float
      - Gender: str (Male/Female/Other)
      - SES: str (Low/Medium/High)
      - Hypertension: int (0/1)
      - Heart_Disease: int (0/1)
      - BMI: float
      - Avg_Glucose: float
      - Diabetes: int (0/1)
      - Smoking_Status: str (Never/Former/Current...)
    """
    Age: float
    Gender: str
    SES: str
    Hypertension: int
    Heart_Disease: int
    BMI: float
    Avg_Glucose: float
    Diabetes: int
    Smoking_Status: str

    # Basic validators
    @field_validator("Age", "BMI", "Avg_Glucose", mode="before")
    @classmethod
    def non_negative_numeric(cls, v):
        try:
            val = float(v)
        except Exception:
            raise ValueError("Giá trị phải là số.")
        if val < 0:
            raise ValueError("Các giá trị số không được âm.")
        return val

    @field_validator("Hypertension", "Heart_Disease", "Diabetes")
    @classmethod
    def binary_int(cls, v):
        try:
            vi = int(v)
        except Exception:
            raise ValueError("Giá trị phải là 0 hoặc 1.")
        if vi not in (0, 1):
            raise ValueError("Các trường nhị phân chỉ chấp nhận 0 hoặc 1.")
        return vi

    @field_validator("Gender", "SES", "Smoking_Status", mode="before")
    @classmethod
    def normalize_str(cls, v):
        if v is None:
            raise ValueError("Trường ký tự không được rỗng.")
        return str(v).strip().title()


class PredictRequest(BaseModel):
    patient: Patient
    threshold: float = Field(0.5, ge=0.01, le=0.99)
    model_name: Optional[str] = Field(
        None, description="Tên file .pkl trong thư mục models/ (vd: 'stroke_model.pkl')"
    )

class PredictResponse(BaseModel):
    status: str
    model_used: str
    threshold: float
    label: int
    probability: float

class BatchPredictItem(Patient):
    """Kế thừa cấu trúc của Patient"""
    pass

class BatchPredictRequest(BaseModel):
    items: List[BatchPredictItem] = Field(..., min_items=1)
    threshold: float = Field(0.5, ge=0.01, le=0.99)
    model_name: Optional[str] = None

class BatchPredictResponse(BaseModel):
    status: str
    model_used: str
    threshold: float
    results: List[dict]  # mỗi dict: {index, label, probability}

# =============================== Endpoints ===============================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "available_models": list_models(),
        "features": FEATURES,
    }

@app.get("/models")
def models():
    return {"status": "ok", "models": list_models()}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Dự đoán cho một bệnh nhân.
    req.patient phải có các trường giống danh sách FEATURES (model_util.FEATURES).
    """
    try:
        model, used = load_model(req.model_name)
        # to_dataframe_one bên model_util chịu trách nhiệm mapping đúng thứ tự FEATURES
        x_df = to_dataframe_one(req.patient.model_dump())
        label, p1 = predict_one(model, x_df, threshold=req.threshold)
        return PredictResponse(
            status="success",
            model_used=used,
            threshold=req.threshold,
            label=label,
            probability=round(p1, 6),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predict error: {e}")

@app.post("/batch_predict", response_model=BatchPredictResponse)
def batch_predict(req: BatchPredictRequest):
    """
    Batch predict: nhận list các items giống schema Patient.
    """
    try:
        model, used = load_model(req.model_name)
        x_df = to_dataframe_batch([it.model_dump() for it in req.items])

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x_df)[:, 1]
        else:
            yhat = model.predict(x_df)
            proba = np.where(yhat == 1, 0.9, 0.1)

        labels = (proba >= req.threshold).astype(int)
        results = [
            {"index": i, "label": int(l), "probability": float(round(p, 6))}
            for i, (l, p) in enumerate(zip(labels, proba))
        ]
        return BatchPredictResponse(
            status="success",
            model_used=used,
            threshold=req.threshold,
            results=results,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch predict error: {e}")

@app.post("/batch_predict_csv")
def batch_predict_csv(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    model_name: Optional[str] = Form(None),
):
    """
    Nhận CSV có đầy đủ các cột trong FEATURES (không cần 'target').
    Trả JSON gồm label & probability cho từng dòng.
    """
    try:
        model, used = load_model(model_name)
        df = pd.read_csv(file.file)

        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Thiếu cột: {missing}")

        X = df[FEATURES]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
        else:
            yhat = model.predict(X)
            proba = np.where(yhat == 1, 0.9, 0.1)

        labels = (proba >= float(threshold)).astype(int)
        return {
            "status": "success",
            "model_used": used,
            "threshold": float(threshold),
            "results": [
                {"index": int(i), "label": int(l), "probability": float(round(p, 6))}
                for i, (l, p) in enumerate(zip(labels, proba))
            ],
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV predict error: {e}")
