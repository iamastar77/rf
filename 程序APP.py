import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import tempfile
import os

# 加载保存的随机森林模型
model = joblib.load('rf.pkl')

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "WBC": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 7.5},
    "BUN": {"type": "numerical", "min": 0.0, "max": 200.0, "default": 13.5},
    "MAP": {"type": "numerical", "min": 0.0, "max": 200.0, "default": 80.0},
    "Age": {"type": "numerical", "min": 18, "max": 90, "default": 54},
    "Temp": {"type": "numerical", "min": 30.0, "max": 42.0, "default": 36.5},
    "RDW": {"type": "numerical", "min": 0.0, "max": 25.0, "default": 13.0},
    "PLT": {"type": "numerical", "min": 0.0, "max": 700.0, "default": 275.0},
    "APTT": {"type": "numerical", "min": 0.0, "max": 150.0, "default": 30.0},
}

# Streamlit 界面
st.title("Sepsis Prediction Model with SHAP Visualization")
st.header("Enter the following feature values:")

# 动态生成输入项
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[predicted_class] * 100

    # 明确显示预测结果
    if predicted_class == 1:
        result_text = f"Predicted possibility of having sepsis: {probability:.2f}%"
    else:
        result_text = f"Predicted possibility of not having sepsis: {probability:.2f}%"
    st.write(result_text)

    # 计算 SHAP 值
    explainer = shap.Explainer(model)
    shap_values = explainer(pd.DataFrame(features, columns=feature_ranges.keys()))

    # 使用 SHAP 力图（HTML 交互版）
    html_output = shap.plots.force(
        explainer.expected_value,
        shap_values.values[0],
        pd.DataFrame(features, columns=feature_ranges.keys()).iloc[0],
        show=False,
        matplotlib=False,
    )
    shap_html = f"<head>{shap.getjs()}</head><body>{html_output.html()}</body>"
    st.components.v1.html(shap_html, height=300)

    # 清理临时文件
    for file in os.listdir(tempfile.gettempdir()):
        if file.startswith("shap_force_plot"):
            os.remove(os.path.join(tempfile.gettempdir(), file))
