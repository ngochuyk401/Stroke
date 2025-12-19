from pathlib import Path
import joblib
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

# Thư mục chứa model .pkl
MODELS_DIR = Path(__file__).parent / "models"

# Danh sách features khớp với stroke_data.csv (không bao gồm target 'Stroke')
FEATURES: List[str] = [
    "Age",
    "Gender",
    "SES",
    "Hypertension",
    "Heart_Disease",
    "BMI",
    "Avg_Glucose",
    "Diabetes",
    "Smoking_Status",
]

# Cache để tránh load nhiều lần
_model_cache: Dict[str, Any] = {}


def list_models() -> List[str]:
    """Trả về danh sách file .pkl trong thư mục models (alphabetical)."""
    MODELS_DIR.mkdir(exist_ok=True)
    return sorted([p.name for p in MODELS_DIR.glob("*.pkl")])


def load_model(model_name: Optional[str] = None) -> Tuple[Any, str]:
    """
    Load model từ thư mục MODELS_DIR.
    Nếu model_name là None -> chọn model đầu tiên theo thứ tự alphabet.
    Trả về (model_object, model_filename).
    """
    MODELS_DIR.mkdir(exist_ok=True)

    all_models = list_models()
    if not all_models:
        raise FileNotFoundError("Thư mục models/ chưa có file .pkl")

    if model_name is None:
        model_name = all_models[0]

    # đảm bảo model_name là tên file (string)
    model_name = str(model_name)

    # cache lookup
    if model_name in _model_cache:
        return _model_cache[model_name], model_name

    path = MODELS_DIR / model_name
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {path}")

    model = joblib.load(path)
    _model_cache[model_name] = model
    return model, model_name


def _ensure_payload_has_features(payload: Dict, features: List[str]) -> None:
    """Ném KeyError nếu payload thiếu bất kỳ feature nào."""
    missing = [f for f in features if f not in payload]
    if missing:
        raise KeyError(f"Payload thiếu các trường bắt buộc: {missing}")


def to_dataframe_one(payload: Dict) -> pd.DataFrame:
    """
    Chuyển dict 1 bệnh nhân -> DataFrame đúng thứ tự FEATURES.
    Nếu payload thiếu key -> KeyError.
    """
    if not isinstance(payload, dict):
        raise TypeError("payload phải là dict")
    _ensure_payload_has_features(payload, FEATURES)

    # Lấy dữ liệu theo đúng thứ tự FEATURES
    data = {feat: [payload[feat]] for feat in FEATURES}
    df = pd.DataFrame(data, columns=FEATURES)
    return df


def to_dataframe_batch(items: List[Dict]) -> pd.DataFrame:
    """
    Chuyển list bệnh nhân -> DataFrame đúng thứ tự cột FEATURES.
    Mỗi item phải là dict chứa đầy đủ FEATURES.
    """
    if not isinstance(items, list):
        raise TypeError("items phải là list các dict")
    rows = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise TypeError(f"Item index {i} không phải dict")
        try:
            _ensure_payload_has_features(it, FEATURES)
        except KeyError as e:
            raise KeyError(f"Item index {i} lỗi: {e}")
        rows.append([it[feat] for feat in FEATURES])
    df = pd.DataFrame(rows, columns=FEATURES)
    return df


def predict_one(model, x_df: pd.DataFrame, threshold: float = 0.5) -> Tuple[int, float]:
    """
    Dự đoán cho dataframe 1 dòng x_df (ở đúng thứ tự cột/features).
    Trả về (label:int, proba:float) với label = 1 nếu proba >= threshold.
    """
    if x_df.shape[0] != 1:
        # cho phép nhưng sẽ lấy dòng đầu nếu người dùng vô tình truyền nhiều dòng
        x_input = x_df.iloc[[0]]
    else:
        x_input = x_df

    if hasattr(model, "predict_proba"):
        # predict_proba trả về mảng (n_samples, n_classes)
        p = model.predict_proba(x_input)
        # lấy proba class 1 (nếu model binary và thứ tự lớp là [0,1])
        try:
            p1 = float(p[:, 1][0])
        except Exception:
            # phòng trường hợp model trả lớp khác thứ tự -> try argmax fallback
            probs = p[0]
            class_idx = probs.argmax()
            p1 = float(probs[class_idx])
    else:
        # fallback: dùng predict nếu không có predict_proba
        yhat = model.predict(x_input)
        p1 = 0.9 if int(yhat[0]) == 1 else 0.1

    label = int(p1 >= float(threshold))
    return label, p1
