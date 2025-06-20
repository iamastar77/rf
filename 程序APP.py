import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
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
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 明确显示预测结果，使用 Matplotlib 渲染指定字体
    if predicted_class == 1:
        text = f"Predicted possibility of having sepsis: {probability:.2f}%"
        fig, ax = plt.subplots(figsize=(8, 1))
        ax.text(
            0.5, 0.5, text,
            fontsize=16,
            ha='center', va='center',
            fontname='Times New Roman',
            transform=ax.transAxes
        )
        ax.axis('off')
        plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
        st.image("prediction_text.png")
    else:
        text = f"Predicted possibility of not having sepsis: {probability:.2f}%"
        fig, ax = plt.subplots(figsize=(8, 1))
        ax.text(
            0.5, 0.5, text,
            fontsize=16,
            ha='center', va='center',
            fontname='Times New Roman',
            transform=ax.transAxes
        )
        ax.axis('off')
        plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
        st.image("prediction_text.png")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 生成 SHAP 力图
    class_index = predicted_class  # 当前预测类别
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:, :, class_index],
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
