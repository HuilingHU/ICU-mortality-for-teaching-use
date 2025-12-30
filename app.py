import streamlit as st
import joblib
import pandas as pd
import requests

# ===============================
# DeepSeek 调用函数
# ===============================
def ask_deepseek(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "deepseek-r1:7b",
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload)
    return response.json()["response"]

# ===============================
# 加载模型和阈值
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
# 初始化 session_state
# ===============================
if "predicted" not in st.session_state:
    st.session_state.predicted = False

# ===============================
# 输入区
# ===============================
gender_cn = st.selectbox("性别", ["男", "女"])
gender_code = 1 if gender_cn == "男" else 0

admission_age = st.number_input("入院年龄（岁）", min_value=0, max_value=90, value=60)
los_icu = st.number_input("ICU 住院时间（天）", min_value=0.0, value=3.0)
wbc = st.number_input("白细胞计数（×10⁹/L）", min_value=0.0, value=8.0)

X_input = pd.DataFrame([{
    "gender": gender_code,
    "admission_age": admission_age,
    "los_icu": los_icu,
    "wbc": wbc
}])

# ===============================
# 预测按钮
# ===============================
if st.button("预测风险"):
    prob = model.predict_proba(X_input)[:, 1][0]
    pred = int(prob >= threshold)

    st.session_state.predicted = True
    st.session_state.prob = prob
    st.session_state.pred = pred

# ===============================
# 预测结果显示（独立于按钮）
# ===============================
if st.session_state.predicted:
    prob = st.session_state.prob
    pred = st.session_state.pred

    st.subheader("预测结果")
    st.write(f"预测死亡风险概率：**{prob:.3f}**")
    st.write(f"模型阈值（Youden）：**{threshold:.3f}**")

    if pred == 1:
        st.error("高风险患者")
    else:
        st.success("低风险患者")

    # ===============================
    # DeepSeek 对话区
    # ===============================
    st.divider()
    st.subheader("🧠 智能解读（DeepSeek）")

    default_prompt = f"""
患者 ICU 死亡风险预测结果如下：
- 预测死亡风险概率：{prob:.3f}
- 风险分层：{"高风险" if pred==1 else "低风险"}

请用临床医生能理解的语言，对该风险结果进行解释，
不提供诊疗建议，仅做风险解读。
"""

    user_question = st.text_area(
        "你可以向模型提问（例如：如何理解这个风险？）",
        value=default_prompt,
        height=180
    )

    if st.button("向 DeepSeek 提问"):
        with st.spinner("DeepSeek 思考中..."):
            answer = ask_deepseek(user_question)
        st.markdown("### DeepSeek 回复")
        st.write(answer)
