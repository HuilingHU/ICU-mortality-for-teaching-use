import streamlit as st
import joblib
import pandas as pd

# ===============================
# 加载模型和阈值（分开）
# ===============================
@st.cache_resource
def load_model_and_threshold():
    model = joblib.load("xgboost_smote.pkl")
    threshold = joblib.load("threshold_xgb.pkl")
    return model, threshold

model, threshold = load_model_and_threshold()

# ===============================
# 页面标题
# ===============================
st.title("ICU 死亡风险预测工具（XGBoost）")
st.write("请输入患者特征，系统将输出预测风险")

# ===============================
# 特征输入（必须与训练时一致）
# ===============================
FEATURES_MAP = [
    "性别": "gender",
    "入院年龄（岁）":"admission_age",
    "ICU住院时间（天）": "los_icu",
    "白细胞计数（×10⁹/L）": "wbc"
]

input_data = {}
gender_cn = st.selectbox("性别", ["男", "女"])
input_data["gender"] = 0 if gender_cn == "女" else 1


# 其他连续变量
# ===============================
input_data["admission_age"] = st.number_input("入院年龄（岁）",min_value=0.0, max_value=90.0, value=0.0)
input_data["los_icu"] = st.number_input( "ICU 住院时间（天）", min_value=0.0, value=0.0)
input_data["wbc"] = st.number_input( "白细胞计数（×10⁹/L）", min_value=0.0, value=0.0)

X_input = pd.DataFrame([input_data])[["gender", "admission_age", "los_icu", "wbc"]]

# ===============================
# 预测
# ===============================
if st.button("预测风险"):
    prob = model.predict_proba(X_input)[:, 1][0]
    pred = int(prob >= threshold)

    st.subheader("预测结果")
    st.write(f"预测死亡风险概率：{prob:.3f}")
    st.write(f"模型阈值（Youden）：{threshold:.3f}")

    if pred == 1:
        st.error("高风险患者")
    else:
        st.success("低风险患者")
