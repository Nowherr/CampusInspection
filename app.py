# app.py
import os
import numpy as np
import cv2
import pandas as pd
import streamlit as st
from PIL import Image

from inference.segmenter import CampusSegmenter, CLASS_NAMES
from scenarios.illegal_parking import detect_illegal_parking
from scenarios.pedestrian_intrusion import detect_pedestrian_intrusion
from scenarios.green_view import evaluate_green_view


CKPT = './checkpoint_deeplabv3plus/deeplabv3plus_resnet50_best.pth'


@st.cache_resource
def load_model():
    return CampusSegmenter(checkpoint_path=CKPT, backbone='resnet50',
                           num_classes=12, ignore_index=0)


st.set_page_config(page_title="校园场景智能巡检系统", layout="wide")
st.title("🏫 基于 DeepLabV3+ 的校园场景智能巡检系统")

seg = load_model()
miou = seg.meta.get('best_miou')
miou_str = f"{float(miou):.4f}" if miou is not None else "N/A"
st.caption(f"Backbone: {seg.meta['backbone']}  |  Best mIoU: {miou_str}  |  Device: {seg.device}")

scenario = st.sidebar.radio(
    "🧭 选择巡检任务",
    ["📸 语义分割概览", "🚗 违停检测", "🚶 行人闯入", "🌳 绿视率评估"]
)

uploaded = st.file_uploader("上传校园场景图片", type=['jpg', 'jpeg', 'png', 'bmp'])

if uploaded:
    pil = Image.open(uploaded).convert('RGB')
    image_rgb = np.array(pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    with st.spinner('🧠 模型推理中...'):
        mask = seg.predict(pil)

    if scenario == "📸 语义分割概览":
        c1, c2 = st.columns(2)
        c1.image(image_rgb, caption="原图", use_column_width=True)
        c2.image(seg.colorize(mask), caption="语义分割结果", use_column_width=True)

        ratio = seg.get_class_ratio(mask)
        df = pd.DataFrame({
            '类别': list(ratio.keys()),
            '占比(%)': [v * 100 for v in ratio.values()]
        }).sort_values('占比(%)', ascending=False)
        st.subheader("各类像素占比")
        st.bar_chart(df.set_index('类别')['占比(%)'])
        st.dataframe(df.round(2))

    elif scenario == "🚗 违停检测":
        thr = st.sidebar.slider("违规重叠阈值", 0.01, 0.5, 0.05, 0.01)
        result = detect_illegal_parking(mask, image_bgr, overlap_threshold=thr)
        st.image(cv2.cvtColor(result['visualization'], cv2.COLOR_BGR2RGB),
                 use_column_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("检测车辆", result['total_cars'])
        c2.metric("违规数量", result['illegal_count'])
        rate = result['illegal_count'] / max(result['total_cars'], 1) * 100
        c3.metric("违规率", f"{rate:.1f}%")
        if result['alerts']:
            st.subheader("⚠️ 告警详情")
            st.json(result['alerts'])

    elif scenario == "🚶 行人闯入":
        result = detect_pedestrian_intrusion(mask, image_bgr)
        st.image(cv2.cvtColor(result['visualization'], cv2.COLOR_BGR2RGB),
                 use_column_width=True)
        c1, c2 = st.columns(2)
        c1.metric("行人总数", result['total_pedestrians'])
        c2.metric("高风险行人", result['high_risk_count'])
        if result['alerts']:
            st.subheader("⚠️ 告警详情")
            st.json(result['alerts'])

    elif scenario == "🌳 绿视率评估":
        result = evaluate_green_view(mask, image_bgr)
        st.image(cv2.cvtColor(result['visualization'], cv2.COLOR_BGR2RGB),
                 use_column_width=True)
        c1, c2 = st.columns(2)
        c1.metric("整体绿视率", f"{result['overall_ratio']*100:.2f}%")
        c2.metric("评级", result['grade'])
        st.subheader("9 宫格分布")
        grid = pd.DataFrame(
            np.array(result['grid_scores']) * 100,
            columns=['左', '中', '右'],
            index=['上', '中', '下']
        ).round(2)
        st.dataframe(grid.style.background_gradient(cmap='Greens'))
else:
    st.info("👈 请上传一张图片（建议先用 CamVid test 集中的图测试）")