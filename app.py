# app.py - Streamlit app cho Stroke dataset
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional

load_dotenv()

# Page config
st.set_page_config(page_title="Stroke Risk • ML App", page_icon="🧠", layout="wide")

# Paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ===================== STYLE =====================
st.markdown("""
<style>
:root { --card-bg:#fff; --soft:#f6f7fb; --primary:#6c63ff; --danger:#ef476f; --ok:#06d6a0; }
.stApp { background: linear-gradient(180deg,#f8fbff 0%,#f2f3ff 100%); }
h1,h2,h3 { font-weight:800; letter-spacing:.2px; }
.card { background:var(--card-bg); padding:1.2rem 1.4rem; border-radius:18px;
        box-shadow:0 8px 24px rgba(80,72,229,.08); border:1px solid #eee; }
.badge { padding:.25rem .55rem; border-radius:999px; font-size:.75rem; background:#eef; color:#334; }
.metric-ok { color:var(--ok); font-weight:700; }
.metric-bad { color:var(--danger); font-weight:700; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ===================== Utilities: model loading & preprocess introspection =====================
@st.cache_resource
def load_model_cached(path: Path):
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Không load được model: {e}")
        return None

def list_model_files() -> List[Path]:
    return sorted([p for p in MODELS_DIR.glob("*.pkl")])

def extract_preprocessor_info(model) -> Dict[str, Any]:
    """
    Try to extract categories from pipeline preprocessor (onehot/ordinal).
    Returns dict with keys possibly: 'ohe_features','ohe_categories','ord_features','ord_categories'
    """
    info: Dict[str, Any] = {}
    try:
        if not hasattr(model, "named_steps"):
            return info
        # find likely preprocessor step name
        pre = None
        for n in ("pre", "preprocessor", "preproc", "preprocessor__ct", "transform"):
            if n in model.named_steps:
                pre = model.named_steps[n]
                break
        # fallback: take first transformer-like step
        if pre is None:
            # iterate named_steps to find a ColumnTransformer-like
            for name, step in model.named_steps.items():
                if hasattr(step, "named_transformers_") or hasattr(step, "transformers_"):
                    pre = step
                    break
        if pre is None:
            return info

        # Now inspect named_transformers_
        nt = getattr(pre, "named_transformers_", None) or getattr(pre, "transformers_", None)
        if nt is None:
            return info

        # If it's a dict (named_transformers_), iterate items
        if isinstance(nt, dict):
            items = nt.items()
        else:
            # transformer's format could be list of (name, transformer, cols)
            items = []
            try:
                for t in nt:
                    # t could be tuple (name, transformer, cols)
                    if isinstance(t, (list, tuple)) and len(t) >= 2:
                        items.append((t[0], t[1]))
            except Exception:
                items = []

        # Search for OneHotEncoder / OrdinalEncoder inside these transformers
        for key, transformer in items:
            if transformer is None:
                continue
            # If pipeline, dive into named_steps
            try:
                if hasattr(transformer, "named_steps"):
                    for subname, sub in transformer.named_steps.items():
                        clsname = sub.__class__.__name__
                        if clsname == "OneHotEncoder":
                            onehot = sub
                            try:
                                info["ohe_categories"] = [list(arr) for arr in onehot.categories_]
                                info["ohe_feature_names_in"] = list(getattr(onehot, "feature_names_in_", []))
                            except Exception:
                                pass
                        if clsname == "OrdinalEncoder":
                            ordenc = sub
                            try:
                                info["ord_categories"] = [list(arr) for arr in ordenc.categories_]
                                info["ord_feature_names_in"] = list(getattr(ordenc, "feature_names_in_", []))
                            except Exception:
                                pass
                else:
                    clsname = transformer.__class__.__name__
                    if clsname == "OneHotEncoder":
                        onehot = transformer
                        try:
                            info["ohe_categories"] = [list(arr) for arr in onehot.categories_]
                            info["ohe_feature_names_in"] = list(getattr(onehot, "feature_names_in_", []))
                        except Exception:
                            pass
                    if clsname == "OrdinalEncoder":
                        ordenc = transformer
                        try:
                            info["ord_categories"] = [list(arr) for arr in ordenc.categories_]
                            info["ord_feature_names_in"] = list(getattr(ordenc, "feature_names_in_", []))
                        except Exception:
                            pass
            except Exception:
                continue
    except Exception:
        return info
    return info

# ===================== Input / Predict helpers =====================
FEATURES = ["Age","Gender","SES","Hypertension","Heart_Disease","BMI","Avg_Glucose","Diabetes","Smoking_Status"]

def make_input_df(Age, Gender, SES, Hypertension, Heart_Disease, BMI, Avg_Glucose, Diabetes, Smoking_Status):
    return pd.DataFrame({
        "Age": [Age],
        "Gender": [Gender],
        "SES": [SES],
        "Hypertension": [Hypertension],
        "Heart_Disease": [Heart_Disease],
        "BMI": [BMI],
        "Avg_Glucose": [Avg_Glucose],
        "Diabetes": [Diabetes],
        "Smoking_Status": [Smoking_Status]
    })

def predict_pipeline(model, X: pd.DataFrame, threshold: float = 0.5):
    if hasattr(model, "predict_proba"):
        p1 = float(model.predict_proba(X)[:,1][0])
    elif hasattr(model, "decision_function"):
        z = float(model.decision_function(X))
        p1 = 1.0 / (1.0 + np.exp(-z))
    else:
        y = int(model.predict(X)[0])
        p1 = 0.9 if y==1 else 0.1
    label = int(p1 >= threshold)
    return label, p1

def nice_percent(x): return f"{x*100:.2f}%"

# ===================== EDA helpers =====================
def numeric_cols(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def describe_with_iqr(df: pd.DataFrame):
    desc = df.describe(include='all').T
    num = df.select_dtypes(include=[np.number])
    if len(num.columns):
        Q1 = num.quantile(0.25); Q3 = num.quantile(0.75); IQR = Q3 - Q1
        outlier_cnt = (((num < (Q1 - 1.5*IQR)) | (num > (Q3 + 1.5*IQR))).sum())
        desc.loc[outlier_cnt.index, "outliers_IQR"] = outlier_cnt.values
    return desc

def plot_hist(df, col, bins=30):
    fig = plt.figure()
    plt.hist(df[col].dropna().values, bins=bins)
    plt.title(f"Histogram • {col}"); plt.xlabel(col); plt.ylabel("Count")
    return fig

def plot_box(df, col):
    fig = plt.figure()
    plt.boxplot(df[col].dropna().values, vert=True, labels=[col])
    plt.title(f"Boxplot • {col}")
    return fig

def plot_corr_heatmap(df):
    num_cols = numeric_cols(df)
    if len(num_cols) < 2:
        st.info("Cần ≥2 cột số để vẽ heatmap tương quan.")
        return
    corr = df[num_cols].corr(numeric_only=True)
    fig = plt.figure(figsize=(6, 5))
    im = plt.imshow(corr.values, vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(num_cols)), num_cols, rotation=45, ha='right', fontsize=8)
    plt.yticks(range(len(num_cols)), num_cols, fontsize=8)
    plt.title("Tương quan (Pearson)"); plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ===================== Simple coach suggestions for stroke-related vars =====================
def risk_bucket(prob):
    if prob >= 0.75: return "Rất cao"
    if prob >= 0.50: return "Cao"
    if prob >= 0.30: return "Trung bình"
    return "Thấp"

def coach_suggestions_stroke(v: Dict[str, Any]):
    alerts, actions = [], []
    # Hypertension & Heart disease & Diabetes & Smoking are major risks
    if v.get("Hypertension", 0) == 1:
        alerts.append("Tiền sử tăng huyết áp.")
        actions.append("Kiểm soát huyết áp: tuân thủ thuốc, giảm muối, giảm cân nếu cần.")
    if v.get("Heart_Disease", 0) == 1:
        alerts.append("Tiền sử bệnh tim.")
        actions.append("Theo dõi tim mạch định kỳ, tuân thủ điều trị chuyên khoa.")
    if v.get("Diabetes", 0) == 1:
        alerts.append("Tiền sử đái tháo đường.")
        actions.append("Kiểm soát đường huyết, xét nghiệm HbA1c, thay đổi chế độ ăn.")
    smoke = str(v.get("Smoking_Status", "")).lower()
    if "smoke" in smoke or "formerly" in smoke or smoke in ["smokes","formerly smoked","current"]:
        alerts.append("Có tiền sử hút thuốc/từng hút thuốc.")
        actions.append("Tư vấn bỏ thuốc; giảm tiếp xúc khói thuốc.")
    bmi = v.get("BMI", None)
    if bmi is not None and bmi >= 30:
        alerts.append(f"Béo phì (BMI={bmi}).")
        actions.append("Giảm cân từng bước: ăn kiêng, tăng vận động, tư vấn dinh dưỡng.")
    ag = v.get("Age", 0)
    if ag >= 65:
        actions.append("Người cao tuổi: tăng cường tầm soát, tiêm ngừa và quản lý bệnh nền.")
    # dedupe
    def uniq(seq):
        seen, out = set(), []
        for x in seq:
            if x not in seen:
                seen.add(x); out.append(x)
        return out
    return uniq(alerts), uniq(actions)

# ===================== Sidebar =====================
with st.sidebar:
    st.markdown("### ⚙️ Tuỳ chọn & Mô hình")
    threshold = st.slider("Threshold nguy cơ", 0.01, 0.99, 0.50, 0.01)
    st.divider()
    st.markdown("#### 📁 Model (.pkl) trong `models/`")
    model_files = list_model_files()
    selected_model_name: Optional[str] = None
    if model_files:
        selected_model_name = st.selectbox("Chọn model", [p.name for p in model_files])
    else:
        st.caption("Chưa có file .pkl trong `models/`.")
    if st.button("🔁 Reload model/clear cache"):
        st.cache_resource.clear(); st.experimental_rerun()
    st.divider()
    st.markdown("### ℹ️ About")
    st.write("**Author:** You  \n**Project:** Stroke risk demo")
    st.divider()

# ===================== Load model & infer categories for UI =====================
model = None
preproc_info = {}
if selected_model_name:
    model_path = MODELS_DIR / selected_model_name
    model = load_model_cached(model_path)
    if model is None:
        st.error("Không thể load model. Kiểm tra file .pkl.")
    else:
        preproc_info = extract_preprocessor_info(model)

# Default category options
DEFAULT_GENDER = ["Female", "Male"]
DEFAULT_SES = ["Low", "Medium", "High"]
DEFAULT_SMOKING = ["Never Smoked", "Formerly Smoked", "Smokes"]

gender_opts = DEFAULT_GENDER[:]
ses_opts = DEFAULT_SES[:]
smoking_opts = DEFAULT_SMOKING[:]

# Try populate from preproc_info
try:
    if preproc_info.get("ohe_categories"):
        cats = preproc_info["ohe_categories"]
        # assume order: Gender, Smoking_Status (best-effort)
        if len(cats) >= 1 and cats[0]:
            gender_opts = cats[0]
        if len(cats) >= 2 and cats[1]:
            smoking_opts = cats[1]
    if preproc_info.get("ord_categories"):
        ordcats = preproc_info["ord_categories"]
        if len(ordcats) >= 1 and ordcats[0]:
            ses_opts = ordcats[0]
except Exception:
    pass

# Normalize string display (ensure values are strings)
gender_opts = [str(x) for x in gender_opts]
ses_opts = [str(x) for x in ses_opts]
smoking_opts = [str(x) for x in smoking_opts]

# ===================== Page layout =====================
tabs = st.tabs(["🏠 Trang chính", "📊 Phân tích dữ liệu", "📈 So sánh mô hình", "ℹ️ About"])
tab_main, tab_eda, tab_cmp, tab_about = tabs

# --------------------- TAB: Trang chính ---------------------
with tab_main:
    st.markdown("<div class='badge'>Stroke Risk • Logistic Regression</div>", unsafe_allow_html=True)
    st.title("Dự đoán nguy cơ đột quỵ (Stroke)")
    st.write("Chọn model pipeline (preprocessor + classifier) trong `models/`, nhập thông tin bệnh nhân, ấn Dự đoán.")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧪 Nhập thông tin bệnh nhân")
    c1, c2, c3 = st.columns(3)
    with c1:
        Age = st.number_input("Tuổi (Age)", min_value=0, max_value=120, value=50)
        Gender = st.selectbox("Giới tính (Gender)", options=gender_opts, index=0)
        SES = st.selectbox("SES (Low/Medium/High)", options=ses_opts, index=min(1, len(ses_opts)-1))
    with c2:
        Hypertension = st.selectbox("Tăng huyết áp", ["Không (0)", "Có (1)"], index=0)
        Hypertension = 1 if "(1)" in Hypertension else 0
        Heart_Disease = st.selectbox("Bệnh tim", ["Không (0)", "Có (1)"], index=0)
        Heart_Disease = 1 if "(1)" in Heart_Disease else 0
        Diabetes = st.selectbox("Tiểu đường", ["Không (0)", "Có (1)"], index=0)
        Diabetes = 1 if "(1)" in Diabetes else 0
    with c3:
        BMI = st.number_input("BMI", min_value=5.0, max_value=80.0, value=25.0, step=0.1)
        Avg_Glucose = st.number_input("Avg_Glucose (mg/dL)", min_value=20.0, max_value=500.0, value=110.0)
        Smoking_Status = st.selectbox("Smoking_Status", options=smoking_opts, index=0)
    st.markdown("</div>", unsafe_allow_html=True)

    if model is None:
        st.warning("Hãy chọn một model pipeline trong sidebar (models/*.pkl).")
    else:
        if st.button("🚀 Dự đoán"):
            X = make_input_df(Age, Gender, SES, Hypertension, Heart_Disease, BMI, Avg_Glucose, Diabetes, Smoking_Status)
            try:
                label, p1 = predict_pipeline(model, X, threshold=threshold)
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {type(e).__name__}: {e}")
            else:
                st.markdown("### 🔍 Kết quả")
                st.write(f"Xác suất (class=1): **{nice_percent(p1)}**")
                st.write(f"Kết luận (threshold={threshold}): **{'Nguy cơ cao' if label==1 else 'Nguy cơ thấp'}**")
                fig, ax = plt.subplots()
                ax.bar(["Risk=1"], [p1], color=['#ef476f' if label==1 else '#06d6a0'])
                ax.set_ylim(0,1)
                ax.set_ylabel("Probability")
                st.pyplot(fig, use_container_width=True)

                # Feature importance (if available)
                try:
                    # If pipeline and final estimator has coef_ or feature_importances_
                    if hasattr(model, "named_steps") and "clf" in model.named_steps:
                        estimator = model.named_steps["clf"]
                    else:
                        # try last step
                        if hasattr(model, "steps"):
                            estimator = model.steps[-1][1]
                        else:
                            estimator = None
                    if estimator is not None:
                        if hasattr(estimator, "coef_"):
                            # linear model: show absolute coef magnitudes
                            coefs = estimator.coef_.ravel()
                            # try derive feature names after preprocessing (best-effort)
                            feat_names = []
                            try:
                                pre = model.named_steps.get("pre") or model.named_steps.get("preprocessor")
                                # get transformed feature names if possible
                                if hasattr(pre, "get_feature_names_out"):
                                    feat_names = list(pre.get_feature_names_out())
                            except Exception:
                                feat_names = FEATURES.copy()
                            if len(feat_names) != len(coefs):
                                # fallback: use FEATURES (approx)
                                feat_names = FEATURES.copy()
                            order = np.argsort(np.abs(coefs))[::-1]
                            st.markdown("#### Độ quan trọng (tương đối từ hệ số)")
                            fig2, ax2 = plt.subplots()
                            ax2.barh(np.array(feat_names)[order], np.abs(coefs)[order])
                            ax2.invert_yaxis()
                            st.pyplot(fig2, use_container_width=True)
                        elif hasattr(estimator, "feature_importances_"):
                            fi = estimator.feature_importances_
                            feat_names = FEATURES.copy()
                            order = np.argsort(fi)[::-1]
                            st.markdown("#### Feature importance")
                            fig2, ax2 = plt.subplots()
                            ax2.barh(np.array(feat_names)[order], fi[order])
                            ax2.invert_yaxis()
                            st.pyplot(fig2, use_container_width=True)
                except Exception:
                    pass

                # show suggestions
                alerts, actions = coach_suggestions_stroke(X.to_dict(orient='records')[0])
                if alerts:
                    st.markdown("**⚠️ Những điểm cần lưu ý**")
                    for a in alerts: st.write(f"- {a}")
                if actions:
                    st.markdown("**✅ Hành động khuyến nghị**")
                    for a in actions: st.write(f"- {a}")
                st.caption("Gợi ý mang tính tham khảo, không thay thế tư vấn/chẩn đoán của bác sĩ.")

    st.markdown("---")
    st.markdown("#### Dữ liệu đầu vào (preview)")
    st.dataframe(make_input_df(Age, Gender, SES, Hypertension, Heart_Disease, BMI, Avg_Glucose, Diabetes, Smoking_Status), use_container_width=True)

# ---------------- TAB: EDA ----------------
with tab_eda:
    st.header("📊 Phân tích dữ liệu (EDA)")
    up = st.file_uploader("Tải dataset (.csv) để EDA (tuỳ chọn)", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Không đọc được file CSV: {e}")
            df = None
        if df is not None:
            st.success(f"Đã tải dataset: {up.name} — {df.shape[0]} dòng × {df.shape[1]} cột")
            with st.expander("👀 Xem trước dữ liệu", expanded=True):
                st.dataframe(df.head(20), use_container_width=True)

            st.subheader("📚 Thống kê mô tả & Outlier (IQR)")
            desc = describe_with_iqr(df)
            st.dataframe(desc, use_container_width=True)

            num_cols = numeric_cols(df)
            if num_cols:
                c1, c2 = st.columns(2)
                with c1:
                    col_hist = st.selectbox("Chọn cột vẽ Histogram", num_cols)
                    bins = st.slider("Số bins", 10, 100, 30, 5)
                    st.pyplot(plot_hist(df, col_hist, bins=bins), use_container_width=True)
                with c2:
                    col_box = st.selectbox("Chọn cột vẽ Boxplot", num_cols, index=min(1, len(num_cols)-1))
                    st.pyplot(plot_box(df, col_box), use_container_width=True)

                st.subheader("🔥 Heatmap tương quan")
                plot_corr_heatmap(df)
            else:
                st.info("Dataset chưa có cột dạng số để vẽ biểu đồ.")
    else:
        st.info("Bạn có thể tải lên một file CSV để phân tích EDA, nhưng không bắt buộc.")

# --- train_and_compare_models() ---

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def train_and_compare_models(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Train nhanh 3 mô hình (LogisticRegression, RandomForest, SVM) trên dataframe df.
    Yêu cầu: df phải có cột 'target' (0/1). Mặc định chỉ dùng các cột số.
    Trả về: result_df (metrics) và dict roc_curves[name] = (fpr, tpr)
    """
    assert "target" in df.columns, "Dataset cần có cột 'target'."

    # Chuẩn hoá dữ liệu: chỉ lấy cột số, bỏ NA cơ bản
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    # Keep only numeric columns (simple demo)
    X_num = X.select_dtypes(include=[np.number]).copy()
    if X_num.shape[1] == 0:
        raise ValueError("Dataset không có cột số nào để train. Hãy upload dataset chứa các cột số (Age, BMI, Avg_Glucose...).")

    X_train, X_test, y_train, y_test = train_test_split(
        X_num, y, test_size=test_size, stratify=y, random_state=random_state
    )

    configs = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=300, solver="lbfgs"))
        ]),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, kernel="rbf", C=1.0, gamma="scale", random_state=random_state))
        ]),
    }

    rows = []
    roc_curves = {}
    for name, model in configs.items():
        # Fit
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        # predict_proba
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            # fallback to decision_function -> sigmoid
            if hasattr(model, "decision_function"):
                z = model.decision_function(X_test)
                y_proba = 1.0 / (1.0 + np.exp(-z))
            else:
                # fallback constant
                y_proba = np.zeros_like(y_pred, dtype=float)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = float("nan")

        rows.append({"Model": name, "Accuracy": acc, "F1": f1, "ROC-AUC": auc})

        # ROC curve
        try:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_curves[name] = (fpr, tpr)
        except Exception:
            roc_curves[name] = (np.array([0,1]), np.array([0,1]))

    result_df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False, na_position="last").reset_index(drop=True)
    return result_df, roc_curves

def plot_roc_curves(roc_curves: dict):
    """
    Nhận dict {name: (fpr, tpr)} và vẽ lên matplotlib figure.
    """
    fig = plt.figure(figsize=(6,5))
    for name, (fpr, tpr) in roc_curves.items():
        try:
            plt.plot(fpr, tpr, label=name)
        except Exception:
            continue
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    return fig

# ---------------- TAB: Model comparison ----------------
with tab_cmp:
    st.header("📈 So sánh mô hình (train nhanh trên dataset đã upload)")
    st.caption("""
    Dataset cần có cột `target` hoặc `Stroke` (0/1). 
    Ứng dụng sẽ tự động nhận diện và chuẩn hóa về `target`.
    Mặc định chỉ dùng các cột số để train mô hình.
    """)

    up2 = st.file_uploader("Tải dataset (.csv) để so sánh", type=["csv"], key="cmp_csv")
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)

    if up2 is not None:
        # Đọc file
        try:
            df2 = pd.read_csv(up2)
        except Exception as e:
            st.error(f"Không đọc được file CSV: {e}")
            st.stop()

        # --- Tự động nhận thư mục nhãn ---
        target_col = None
        if "target" in df2.columns:
            target_col = "target"
        elif "Stroke" in df2.columns:
            target_col = "Stroke"

        if not target_col:
            st.error("❌ Dataset phải có cột 'target' hoặc 'Stroke' (0/1)!")
            st.stop()

        st.success(f"Đã phát hiện cột nhãn: **{target_col}**")

        # Chuẩn hóa về tên 'target' để train
        df2 = df2.rename(columns={target_col: "target"})

        # Train thử các mô hình
        try:
            res_df, rocs = train_and_compare_models(df2, test_size=test_size)
        except Exception as e:
            st.error(f"❌ Lỗi khi train mô hình: {e}")
            st.stop()

        # Hiển thị kết quả
        st.subheader("📋 Kết quả")
        st.dataframe(
            res_df.style.format({
                "Accuracy": "{:.3f}",
                "F1": "{:.3f}",
                "ROC-AUC": "{:.3f}"
            }),
            use_container_width=True
        )

        st.subheader("📉 ROC Curves")
        st.pyplot(plot_roc_curves(rocs), use_container_width=True)

# ---------------- TAB: ABOUT ----------------
with tab_about:
    st.header("ℹ️ About")
    st.write("""
Ứng dụng minh hoạ triển khai **ML cho dự đoán nguy cơ đột quỵ (Stroke)**:
- Nhập dữ liệu & dự đoán (model pipeline preprocessor + classifier).
- Phân tích dữ liệu (EDA): thống kê, histogram, boxplot, heatmap.
- So sánh mô hình: train nhanh LR / RF / SVM trên dataset (cột `target`).
**Lưu ý**: Ứng dụng chỉ mang tính tham khảo, không thay thế chẩn đoán y khoa.
""")
