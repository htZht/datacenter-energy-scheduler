# app_streamlit.py
import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 关键：避免云端绘图冲突
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# ==============================
# 安全导入模块（仅定义，不执行）
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
except ImportError as e:
    st.error("❌ 缺少必要模块，请确保所有 .py 文件都在仓库根目录！")
    st.exception(e)
    st.stop()

# ==============================
# 辅助函数：带超时的安全光伏预测
# ==============================
def safe_pv_forecast(lat, lon, pv_area, pv_eff, global_config, timeout=15):
    """安全获取光伏数据，超时则回退到默认值"""
    def _fetch():
        return pv_forecast_from_location(lat, lon, pv_area=pv_area, pv_eff=pv_eff)
    
    try:
        with ThreadPoolExecutor() as executor:
            future = executor.submit(_fetch)
            return future.result(timeout=timeout)
    except (TimeoutError, Exception) as e:
        st.warning(f"⚠️ 光伏数据获取失败（{str(e)}），使用默认天气数据")
        return pv_forecast_default(global_config)

# ==============================
# 初始化会话状态
# ==============================
if 'result' not in st.session_state:
    st.session_state.result = None
if 'is_optimizing' not in st.session_state:
    st.session_state.is_optimizing = False

# ==============================
# UI 布局（轻量级，快速加载）
# ==============================
st.set_page_config(page_title="数据中心能源调度系统", layout="wide")
st.title("🌍 数据中心综合能源调度优化")

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
    
    # 防重复点击 + 状态管理
    if st.button("🚀 开始优化", type="primary", disabled=st.session_state.is_optimizing):
        st.session_state.is_optimizing = True
        st.session_state.result = None
        try:
            with st.spinner("正在优化...（约10-20秒，请勿刷新）"):
                # --- 构建配置 ---
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

                # --- 获取风光数据（带超时保护）---
                P_pv = safe_pv_forecast(lat, lon, pv_area, pv_eff, global_config)
                P_wind = wind_forecast_default()

                # --- 生成负荷 ---
                hours = np.arange(24)
                P_el = base_el + 10 * np.sin((hours - 6) * np.pi / 12)
                Q_cool = np.maximum(peak_cool * (0.7 + 0.3 * np.sin((hours - 14) * np.pi / 12)), peak_cool * 0.5)
                Q_heat = np.full(24, heat_load)

                # --- 执行优化（降低代数以加速）---
                T = 24
                n_vars = 9 * T
                obj_func = lambda x: weighted_objective(x, P_pv, P_wind, P_el, Q_cool, Q_heat, global_config, w1=w1, w2=w2, w3=w3)
                x_opt = optimize_single_objective(obj_func, n_vars, bounds=(0, 500), n_gen=20)  # 关键：从30降到20

                # --- 计算指标 ---
                cost = economic_cost(x_opt, P_pv, P_wind, P_el, Q_cool, Q_heat, global_config)[0] * 365 / 1e4
                carbon = carbon_emission(x_opt, P_pv, P_wind, P_el, Q_cool, Q_heat, global_config)[0] * 365 / 1000
                ESI, _, _ = calculate_ESI(x_opt, P_pv, P_wind, P_el, Q_cool, Q_heat, global_config)

                # --- 保存结果 ---
                st.session_state.result = {
                    'x_opt': x_opt,
                    'P_pv': P_pv,
                    'P_wind': P_wind,
                    'P_el': P_el,
                    'Q_cool': Q_cool,
                    'Q_heat': Q_heat,
                    'global_config': global_config,
                    'annual_cost_10k_yuan': round(cost, 2),
                    'annual_carbon_ton': round(carbon, 0),
                    'ESI': round(ESI, 4)
                }
        except Exception as e:
            st.error(f"❌ 优化过程中出错: {str(e)}")
        finally:
            st.session_state.is_optimizing = False

# ==============================
# 结果展示区
# ==============================
with col2:
    st.header("📈 优化结果")
    if st.session_state.result:
        res = st.session_state.result
        st.metric("年经济成本", f"{res['annual_cost_10k_yuan']} 万元")
        st.metric("年碳排放", f"{int(res['annual_carbon_ton'])} 吨")
        st.metric("能值可持续指数 (ESI)", f"{res['ESI']:.4f}")
        
        # ✅ 使用 st.pyplot 直接渲染，无前端错误
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_scheduling(
            res['x_opt'], 
            res['P_pv'], 
            res['P_wind'], 
            res['P_el'], 
            res['Q_cool'], 
            res['Q_heat'], 
            "24小时优化调度结果", 
            res['global_config'], 
            ax=ax
        )
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("点击左侧「开始优化」按钮以查看结果")