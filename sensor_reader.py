# web_ui.py
import streamlit as st
import requests
import base64
from PIL import Image
import io

st.set_page_config(page_title="数据中心能源调度系统", layout="wide")
st.title("🌍 数据中心综合能源调度优化")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("参数设置")
    location = st.text_input("位置（城市或经纬度）", "上海")
    mode = st.selectbox("运行模式", ["仿真模式", "硬件模式"])
    
    if mode == "硬件模式":
        st.warning("需确保 Arduino 已连接并输出传感器数据")
    
    st.subheader("优化权重")
    w1 = st.slider("经济成本权重", 0.0, 1.0, 0.5)
    w2 = st.slider("碳排放权重", 0.0, 1.0, 0.3)
    w3 = st.slider("可持续性权重", 0.0, 1.0, 0.2)
    
    if st.button("🚀 开始优化", type="primary"):
        with st.spinner("正在优化...（约10-30秒）"):
            api_mode = "hardware" if mode == "硬件模式" else "simulation"
            try:
                response = requests.post(
                    "http://localhost:5000/optimize",
                    json={
                        "location": location,
                        "mode": api_mode,
                        "weights": {"w1": w1, "w2": w2, "w3": w3}
                    },
                    timeout=60
                )
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.result = result
                else:
                    st.error(f"API 错误: {response.status_code}")
            except Exception as e:
                st.error(f"请求失败: {str(e)}")

with col2:
    st.header("优化结果")
    if 'result' in st.session_state:
        res = st.session_state.result
        st.metric("年经济成本", f"{res['annual_cost_10k_yuan']} 万元")
        st.metric("年碳排放", f"{int(res['annual_carbon_ton'])} 吨")
        st.metric("能值可持续指数 (ESI)", f"{res['ESI']:.4f}")
        
        img_data = base64.b64decode(res['plot'])
        img = Image.open(io.BytesIO(img_data))
        st.image(img, use_container_width=True)
    else:
        st.info("点击「开始优化」查看结果")