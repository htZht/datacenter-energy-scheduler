# web_ui.py
import streamlit as st
import requests
import base64
from PIL import Image
import io

st.set_page_config(page_title="数据中心能源调度系统", layout="wide")
st.title("🌍 数据中心综合能源调度优化")

if 'result' not in st.session_state:
    st.session_state.result = None

col1, col2 = st.columns([1, 2])

with col1:
    st.header("📍 位置与模式")
    location = st.text_input("位置（城市或经纬度）", "上海")
    mode = st.selectbox("运行模式", ["仿真模式", "硬件模式"])
    
    if mode == "硬件模式":
        st.warning("需确保传感器已连接（当前为模拟数据）")

    st.markdown("---")
    st.header("⚙️ 设备参数配置")
    
    pv_area = st.number_input("光伏面积 (m²)", min_value=0.0, value=400.0, step=10.0)
    pv_eff = st.slider("光伏效率", 0.0, 1.0, 0.175, 0.005)
    wind_capacity = st.number_input("风机额定功率 (kW)", min_value=0.0, value=100.0, step=10.0)
    boiler_max = st.number_input("燃气锅炉最大功率 (kW)", min_value=0.0, value=200.0, step=10.0)
    chiller_elec_max = st.number_input("电制冷机最大功率 (kW)", min_value=0.0, value=150.0, step=10.0)
    chiller_abs_max = st.number_input("吸收式制冷机最大功率 (kW)", min_value=0.0, value=100.0, step=10.0)
    bess_capacity = st.number_input("电池容量 (kWh)", min_value=0.0, value=500.0, step=50.0)
    bess_max_power = st.number_input("电池最大充放电功率 (kW)", min_value=0.0, value=100.0, step=10.0)
    tes_capacity = st.number_input("蓄冷罐容量 (kWh)", min_value=0.0, value=2000.0, step=100.0)

    st.markdown("---")
    st.header("📊 负荷需求（典型值）")
    base_el = st.number_input("基础电负荷 (kW)", min_value=0.0, value=200.0, step=10.0)
    peak_cool = st.number_input("峰值冷负荷 (kW)", min_value=0.0, value=150.0, step=10.0)
    heat_load = st.number_input("热负荷 (kW)", min_value=0.0, value=20.0, step=5.0)

    st.markdown("---")
    st.header("🎯 优化权重")
    w1 = st.slider("经济成本权重", 0.0, 1.0, 0.5, 0.05)
    w2 = st.slider("碳排放权重", 0.0, 1.0, 0.3, 0.05)
    w3 = st.slider("可持续性权重", 0.0, 1.0, 0.2, 0.05)
    
    if st.button("🚀 开始优化", type="primary"):
        with st.spinner("正在优化...（约10-30秒）"):
            api_mode = "hardware" if mode == "硬件模式" else "simulation"
            payload = {
                "location": location,
                "mode": api_mode,
                "weights": {"w1": w1, "w2": w2, "w3": w3},
                "device_config": {
                    "pv_area": pv_area,
                    "pv_efficiency": pv_eff,
                    "wind_capacity": wind_capacity,
                    "boiler_max": boiler_max,
                    "chiller_elec_max": chiller_elec_max,
                    "chiller_abs_max": chiller_abs_max,
                    "bess_capacity": bess_capacity,
                    "bess_max_power": bess_max_power,
                    "tes_capacity": tes_capacity
                },
                "load_profile": {
                    "base_el": base_el,
                    "peak_cool": peak_cool,
                    "heat_load": heat_load
                }
            }
            try:
                response = requests.post(
                    "http://localhost:5000/optimize",
                    json=payload,
                    timeout=60
                )
                if response.status_code == 200:
                    st.session_state.result = response.json()
                else:
                    st.error(f"API 错误: {response.status_code}")
            except Exception as e:
                st.error(f"请求失败: {str(e)}")

with col2:
    st.header("📈 优化结果")
    if st.session_state.result:
        res = st.session_state.result
        st.metric("年经济成本", f"{res['annual_cost_10k_yuan']} 万元")
        st.metric("年碳排放", f"{int(res['annual_carbon_ton'])} 吨")
        st.metric("能值可持续指数 (ESI)", f"{res['ESI']:.4f}")
        
        img_data = base64.b64decode(res['plot'])
        img = Image.open(io.BytesIO(img_data))
        st.image(img, use_container_width=True)
    else:
        st.info("点击「开始优化」查看结果")