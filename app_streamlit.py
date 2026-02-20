# app_streamlit.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image

# ==============================
# 自定义模块（必须和此文件在同一目录！）
# ==============================
try:
    from location_utils import parse_location_input, get_regional_config
    from config import build_config
    from pv_model import pv_forecast_from_location, pv_forecast_default
    from wind_model import wind_forecast_default
    from objectives import economic_cost, carbon_emission, weighted_objective
    from emergy_analysis import calculate_ESI
    from optimizer import optimize_single_objective
    from plot_results import plot_scheduling
except Exception as e:
    st.error("❌ 缺少必要模块，请确保所有 .py 文件都在同一文件夹！")
    st.exception(e)
    st.stop()

def generate_load_from_input(load_params):
    """根据用户输入生成24小时负荷曲线"""
    base_el = load_params.get('base_el', 200.0)
    peak_cool = load_params.get('peak_cool', 150.0)
    heat_load = load_params.get('heat_load', 20.0)
    hours = np.arange(24)
    P_el = base_el + 10 * np.sin((hours - 6) * np.pi / 12)
    Q_cool = np.maximum(peak_cool * (0.7 + 0.3 * np.sin((hours - 14) * np.pi / 12)), peak_cool * 0.5)
    Q_heat = np.full(24, heat_load)
    return P_el, Q_cool, Q_heat

# ==============================
# 🌐 Streamlit 界面
# ==============================
st.set_page_config(page_title="数据中心能源调度系统", layout="wide")
st.title("🌍 数据中心综合能源调度优化")

if 'result' not in st.session_state:
    st.session_state.result = None

col1, col2 = st.columns([1, 2])

with col1:
    st.header("📍 位置设置")
    location = st.text_input("城市或经纬度", "上海")
    
    st.markdown("---")
    st.header("⚙️ 设备参数")
    pv_area = st.number_input("光伏面积 (m²)", min_value=0.0, value=400.0, step=10.0)
    pv_eff = st.slider("光伏效率", 0.0, 1.0, 0.175, 0.005)
    boiler_max = st.number_input("燃气锅炉最大功率 (kW)", min_value=0.0, value=200.0, step=10.0)
    chiller_elec_max = st.number_input("电制冷机最大功率 (kW)", min_value=0.0, value=150.0, step=10.0)
    chiller_abs_max = st.number_input("吸收式制冷机最大功率 (kW)", min_value=0.0, value=100.0, step=10.0)
    bess_capacity = st.number_input("电池容量 (kWh)", min_value=0.0, value=500.0, step=50.0)
    bess_max_power = st.number_input("电池最大充放电功率 (kW)", min_value=0.0, value=100.0, step=10.0)
    tes_capacity = st.number_input("蓄冷罐容量 (kWh)", min_value=0.0, value=2000.0, step=100.0)

    st.markdown("---")
    st.header("📊 负荷需求")
    base_el = st.number_input("基础电负荷 (kW)", min_value=0.0, value=200.0, step=10.0)
    peak_cool = st.number_input("峰值冷负荷 (kW)", min_value=0.0, value=150.0, step=10.0)
    heat_load = st.number_input("热负荷 (kW)", min_value=0.0, value=20.0, step=5.0)

    st.markdown("---")
    st.header("🎯 优化权重")
    w1 = st.slider("经济成本权重", 0.0, 1.0, 0.5, 0.05)
    w2 = st.slider("碳排放权重", 0.0, 1.0, 0.3, 0.05)
    w3 = st.slider("可持续性权重", 0.0, 1.0, 0.2, 0.05)
    
    if st.button("🚀 开始优化", type="primary"):
        with st.spinner("正在优化...（约15-30秒）"):
            try:
                # 构建配置
                lat, lon, _ = parse_location_input(location)
                regional_config = get_regional_config(lat, lon)
                device_config = {
                    'pv_area': pv_area,
                    'pv_efficiency': pv_eff,
                    'boiler_max': boiler_max,
                    'chiller_elec_max': chiller_elec_max,
                    'chiller_abs_max': chiller_abs_max,
                    'bess_capacity': bess_capacity,
                    'bess_max_power': bess_max_power,
                    'tes_capacity': tes_capacity
                }
                global_config = build_config(device_config, regional_config)

                # 风光数据
                try:
                    P_pv = pv_forecast_from_location(lat, lon, pv_area=pv_area, pv_eff=pv_eff)
                except:
                    P_pv = pv_forecast_default(global_config)
                P_wind = wind_forecast_default()

                # 负荷
                P_el, Q_cool, Q_heat = generate_load_from_input({
                    'base_el': base_el,
                    'peak_cool': peak_cool,
                    'heat_load': heat_load
                })

                # 优化
                T = 24
                n_vars = 9 * T
                obj_func = lambda x: weighted_objective(x, P_pv, P_wind, P_el, Q_cool, Q_heat, global_config, w1=w1, w2=w2, w3=w3)
                x_opt = optimize_single_objective(obj_func, n_vars, bounds=(0, 500), n_gen=30)

                # 计算指标
                cost = economic_cost(x_opt, P_pv, P_wind, P_el, Q_cool, Q_heat, global_config)[0] * 365 / 1e4
                carbon = carbon_emission(x_opt, P_pv, P_wind, P_el, Q_cool, Q_heat, global_config)[0] * 365 / 1000
                ESI, _, _ = calculate_ESI(x_opt, P_pv, P_wind, P_el, Q_cool, Q_heat, global_config)

                # 生成图表
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_scheduling(x_opt, P_pv, P_wind, P_el, Q_cool, Q_heat, "优化结果", global_config, ax=ax)
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)

                st.session_state.result = {
                    'annual_cost_10k_yuan': round(cost, 2),
                    'annual_carbon_ton': round(carbon, 0),
                    'ESI': round(ESI, 4),
                    'plot': img_base64
                }
            except Exception as e:
                st.error(f"优化失败: {str(e)}")

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