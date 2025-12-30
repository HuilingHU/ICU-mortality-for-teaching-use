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
FEATURES = [
    "gender",
    "admission_age",
    "los_icu",
    "wbc"
]

input_data = {}
for feature in FEATURES:
    input_data[feature] = st.number_input(
        feature,
        value=0.0
    )

X_input = pd.DataFrame([input_data])

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