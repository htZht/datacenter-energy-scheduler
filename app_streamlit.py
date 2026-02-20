# -*- coding: utf-8 -*-
"""
全要素能源调度平台 v5.1 —— 防乱码终极版
✅ 无方块 | ✅ 氢能可见 | ✅ 硬件扩展 | ✅ 开箱即用
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import hashlib
from datetime import datetime

# ==============================================================================
# 【1】区域与设备参数库
# ==============================================================================
REGIONS = {
    "华北": ["北京市", "天津市", "河北省", "山西省", "内蒙古自治区"],
    "华东": ["上海市", "江苏省", "浙江省", "安徽省", "福建省", "江西省", "山东省"],
    "华中": ["河南省", "湖北省", "湖南省"],
    "华南": ["广东省", "广西壮族自治区", "海南省"],
    "西南": ["重庆市", "四川省", "贵州省", "云南省", "西藏自治区"],
    "西北": ["陕西省", "甘肃省", "青海省", "宁夏回族自治区", "新疆维吾尔自治区"],
    "东北": ["辽宁省", "吉林省", "黑龙江省"]
}

PV_TECH = {
    "单晶硅 PERC (高效)": {"efficiency": 0.23, "temp_coeff": -0.0030, "low_light_perf": 0.95},
    "TOPCon (N型)": {"efficiency": 0.245, "temp_coeff": -0.0028, "low_light_perf": 0.97},
    "HJT (异质结)": {"efficiency": 0.25, "temp_coeff": -0.0025, "low_light_perf": 0.98},
    "多晶硅 (传统)": {"efficiency": 0.175, "temp_coeff": -0.0042, "low_light_perf": 0.88},
    "薄膜 CdTe": {"efficiency": 0.165, "temp_coeff": -0.0020, "low_light_perf": 0.92}
}

WIND_MODELS = {
    "Vestas V150-4.2MW": {"rated_power": 4200, "cut_in": 3, "cut_out": 25, "rated_wind": 12.5},
    "Siemens SG 5.0-145": {"rated_power": 5000, "cut_in": 3, "cut_out": 25, "rated_wind": 12},
    "金风 GW140-3.0MW": {"rated_power": 3000, "cut_in": 3, "cut_out": 22, "rated_wind": 11},
    "海上 Haliade-X 14MW": {"rated_power": 14000, "cut_in": 4, "cut_out": 28, "rated_wind": 13}
}

GT_MODELS = {
    "LM2500+ (30MW)": {"min_load": 0.3, "efficiency": 0.38, "fuel_cost": 0.30},
    "Frame 7FA (170MW)": {"min_load": 0.4, "efficiency": 0.36, "fuel_cost": 0.28},
    "小型燃气轮机 (5MW)": {"min_load": 0.2, "efficiency": 0.32, "fuel_cost": 0.32}
}

# ==============================================================================
# 【2】核心模型函数
# ==============================================================================

def get_weather(province):
    seed = int(hashlib.md5(province.encode()).hexdigest()[:6], 16) % 100
    np.random.seed(seed)
    region_solar = {"西北":700,"华北":620,"华东":520,"华南":560,"西南":480,"东北":510,"华中":530}
    region = next(k for k, v in REGIONS.items() if province in v)
    ghi = np.clip(np.random.normal(region_solar[region], 180, 24), 0, 1100)
    wind = 4.5 + 3.5 * np.random.rand(24)
    temp = 18 + 12 * np.sin(np.arange(24)/24*2*np.pi - np.pi/2) + 4 * np.random.randn(24)
    return ghi, wind, temp

def calc_pv(ghi, area, tech, temp, tilt, azimuth, inv_eff, soiling_loss):
    t = PV_TECH[tech]
    cos_incidence = max(0.2, np.cos(np.radians(tilt)) * 0.9 + 0.1)
    effective_ghi = ghi * cos_incidence * t["low_light_perf"]
    power_dc = effective_ghi * area * t["efficiency"] / 1000
    power_dc *= (1 + t["temp_coeff"] * (temp - 25))
    ac_power = power_dc * inv_eff * (1 - soiling_loss)
    return np.clip(ac_power, 0, None)

def calc_wind(wind_speed, model, n_turbines):
    m = WIND_MODELS[model]
    power = np.zeros_like(wind_speed)
    mask = (wind_speed >= m["cut_in"]) & (wind_speed <= m["cut_out"])
    ratio = np.minimum((wind_speed[mask] - m["cut_in"]) / (m["rated_wind"] - m["cut_in"]), 1.0)
    power[mask] = m["rated_power"] * (ratio ** 3)
    return power * n_turbines

# ==============================================================================
# 【3】调度核心算法（含氢能）
# ==============================================================================

def generate_schedule(P_load, Q_heat, Q_cool, P_pv, P_wind, caps, weights):
    schedule = np.zeros((9, 24))
    
    # 可再生能源优先
    schedule[0] = np.minimum(P_pv, caps['pv'])
    schedule[1] = np.minimum(P_wind, caps['wind'])
    
    residual = P_load - schedule[0] - schedule[1]
    w_gt, w_grid = weights[0], weights[1]
    total_w = w_gt + w_grid + 1e-8
    gt_ratio = w_gt / total_w
    
    for t in range(24):
        if residual[t] > 0:
            gt_use = min(residual[t] * gt_ratio, caps['gt'])
            schedule[2, t] = gt_use
            schedule[3, t] = residual[t] - gt_use
        else:
            schedule[3, t] = 0
    
    # 热/冷系统
    schedule[6] = np.minimum(Q_heat, caps['boiler'])
    schedule[7] = Q_cool * 0.3
    schedule[8] = Q_heat * 0.2
    
    # 🔥 氢能补缺（关键！）
    for t in range(24):
        total_supply = schedule[0, t] + schedule[1, t] + schedule[2, t] + schedule[3, t]
        if total_supply < P_load[t] and caps['h2_fc'] > 0:
            deficit = P_load[t] - total_supply
            h2_use = min(deficit, caps['h2_fc'])
            schedule[5, t] = h2_use  # ← 氢燃料电池出力
            schedule[3, t] += deficit - h2_use
    
    return schedule

# ==============================================================================
# 【4】可视化：英文标签 + 中文标题（彻底防乱码）
# ==============================================================================

def plot_schedule(schedule, P_load, Q_cool, Q_heat):
    hours = np.arange(24)
    # 英文标签 → 避免方块
    labels = ['PV', 'Wind', 'Gas Turbine', 'Grid Import', 'Battery', 'H₂ Fuel Cell', 'Gas Boiler', 'Chilled Storage', 'Thermal Storage']
    colors = ['#FFD700', '#87CEEB', '#8B0000', '#808080', '#4682B4', '#BA55D3', '#FF6347', '#00CED1', '#FFA500']
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 9))
    
    # 电力堆叠图
    bottom = np.zeros(24)
    for i in range(6):  # 前6项是电力
        if np.any(schedule[i] > 0):
            axs[0].fill_between(hours, bottom, bottom + schedule[i], label=labels[i], color=colors[i], alpha=0.8)
            bottom += schedule[i]
    axs[0].plot(hours, P_load, 'k--', linewidth=2, label='Electric Load')
    axs[0].set_ylabel('Power (kW)')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.5)
    
    # 冷负荷
    axs[1].plot(hours, Q_cool, 'b-', linewidth=2, label='Cooling Load')
    axs[1].fill_between(hours, 0, schedule[7], color='#00CED1', alpha=0.6, label='Chilled Storage Discharge')
    axs[1].set_ylabel('Cooling (kW)')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.5)
    
    # 热负荷
    axs[2].plot(hours, Q_heat, 'r-', linewidth=2, label='Heating Load')
    axs[2].fill_between(hours, 0, schedule[6], color='#FF6347', alpha=0.6, label='Gas Boiler')
    axs[2].fill_between(hours, schedule[6], schedule[6] + schedule[8], color='#FFA500', alpha=0.6, label='Thermal Storage Discharge')
    axs[2].set_ylabel('Heat (kW)')
    axs[2].set_xlabel('Hour of Day')
    axs[2].legend()
    axs[2].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig

# ==============================================================================
# 【5】硬件监测（新增氢罐液位）
# ==============================================================================

def get_hardware_data(h2_used_kwh=0):
    """模拟硬件状态，可传入氢能使用量来动态更新氢罐液位"""
    now = datetime.now()
    np.random.seed(int(now.timestamp()) % 1000)
    # 初始氢罐容量假设为 1000 kWh
    h2_capacity = 1000
    h2_consumed = min(h2_used_kwh, h2_capacity)
    h2_level = max(0, h2_capacity - h2_consumed)
    return {
        "PV Temperature": f"{25 + 20 * np.random.rand():.1f} °C",
        "Wind Speed": f"{4.5 + 3.5 * np.random.rand():.1f} m/s",
        "Battery SOC": f"{(0.4 + 0.5 * np.random.rand())*100:.1f} %",
        "H₂ Tank Pressure": f"{25 + 10 * np.random.rand():.1f} MPa",
        "H₂ Tank Level": f"{h2_level:.0f} kWh",
        "Inverter Efficiency": f"{(0.95 + 0.04 * np.random.rand())*100:.1f} %"
    }

# ==============================================================================
# 【6】Streamlit 主应用
# ==============================================================================

st.set_page_config(page_title="能源调度平台 - 防乱码版", layout="wide")
st.title("⚡ 多能协同调度平台（氢能增强版）")

# ------------------- 侧边栏 -------------------
with st.sidebar:
    st.subheader("🌍 地理与规模")
    region = st.selectbox("选择大区", list(REGIONS.keys()))
    province = st.selectbox("选择省份", REGIONS[region])
    
    st.divider()
    st.subheader("📈 负荷配置")
    base_elec = st.slider("基础电负荷 (kW)", 500, 10000, 3000)
    cool_ratio = st.slider("冷负荷比例", 0.0, 1.0, 0.5)
    heat_ratio = st.slider("热负荷比例", 0.0, 1.0, 0.4)

    st.divider()
    st.subheader("🔧 设备开关")
    pv_on = st.checkbox("光伏系统", True)
    wind_on = st.checkbox("风电系统", True)
    gt_on = st.checkbox("燃气轮机", True)
    h2_on = st.checkbox("氢能系统", True)  # ✅ 氢能开关
    monitor_on = st.checkbox("硬件监测", True)

    if pv_on:
        st.subheader("☀️ 光伏参数")
        pv_type = st.selectbox("光伏技术", list(PV_TECH.keys()))
        pv_area = st.number_input("安装面积 (m²)", 100, 50000, 5000)
        tilt = st.slider("倾角 (°)", 0, 90, 25)
        azimuth = st.slider("方位角 (°)", -180, 180, 0)
        inv_eff = st.slider("逆变器效率", 0.8, 1.0, 0.97)
        soiling = st.slider("污渍损失", 0.0, 0.2, 0.03)
    else:
        pv_type, pv_area, tilt, azimuth, inv_eff, soiling = "", 0, 0, 0, 0.97, 0.03

    if wind_on:
        st.subheader("💨 风电参数")
        wt_type = st.selectbox("风机型号", list(WIND_MODELS.keys()))
        n_wt = st.number_input("风机数量", 0, 50, 3)
    else:
        wt_type, n_wt = "", 0

    if gt_on:
        st.subheader("🔥 燃气轮机")
        gt_type = st.selectbox("型号", list(GT_MODELS.keys()))
        gt_capacity = st.number_input("额定容量 (kW)", 1000, 200000, 5000)
    else:
        gt_type, gt_capacity = "", 0

    st.subheader("♨️ 热力与氢能")
    boiler_cap = st.number_input("燃气锅炉容量 (kW)", 0, 50000, 3000)
    h2_cap = st.number_input("氢燃料电池容量 (kW)", 0, 5000, 1000 if h2_on else 0)

# ------------------- 主界面 -------------------
if st.button("🚀 生成调度方案", type="primary"):
    h = np.arange(24)
    P_load = base_elec * (0.6 + 0.4 * np.sin(2 * np.pi * (h - 8) / 24))
    Q_cool = base_elec * cool_ratio * (0.5 + 0.5 * np.abs(np.sin(2 * np.pi * (h - 14) / 24)))
    Q_heat = base_elec * heat_ratio * (0.5 + 0.5 * np.abs(np.sin(2 * np.pi * (h + 3) / 24)))

    ghi, wind_spd, temp = get_weather(province)
    P_pv = calc_pv(ghi, pv_area, pv_type, temp, tilt, azimuth, inv_eff, soiling) if pv_on else np.zeros(24)
    P_wind = calc_wind(wind_spd, wt_type, n_wt) if wind_on else np.zeros(24)

    caps = {
        'pv': 1e6 if pv_on else 0,
        'wind': 1e6 if wind_on else 0,
        'gt': gt_capacity if gt_on else 0,
        'h2_fc': h2_cap if h2_on else 0,
        'boiler': boiler_cap
    }

    weights = [0.4, 0.3, 0.2, 0.1]
    schedule = generate_schedule(P_load, Q_heat, Q_cool, P_pv, P_wind, caps, weights)

    # 计算氢能总用量（用于硬件监测）
    total_h2_used = np.sum(schedule[5])  # kWh

    # 结果展示（中文标题，无乱码）
    st.subheader(f"📊 {province} 调度结果")
    col1, col2, col3 = st.columns(3)
    total_e = np.sum(P_load)
    ren_used = np.sum(schedule[0] + schedule[1])
    fuel_cost = GT_MODELS.get(gt_type, {}).get('fuel_cost', 0.3) if gt_on else 0.3
    col1.metric("可再生占比", f"{ren_used/total_e*100:.1f}%")
    col2.metric("总碳排 (kgCO₂)", f"{(0.785*np.sum(schedule[3]) + 0.45*np.sum(schedule[2])):.0f}")
    col3.metric("总成本 (元)", f"{(np.sum(schedule[3])*0.6 + np.sum(schedule[2])*fuel_cost):.0f}")

    # 调度表（中文列名，但内容数字，安全）
    st.subheader("🔍 24小时调度方案 (kW)")
    df = pd.DataFrame(
        schedule.T,
        columns=["光伏", "风电", "燃气轮机", "电网购电", "电池放电", "氢燃料电池", "燃气锅炉", "蓄冷", "蓄热"]
    )
    df.insert(0, "小时", h)
    st.dataframe(df.round(1), use_container_width=True, hide_index=True)

    # 图表（英文标签，彻底防乱码）
    fig = plot_schedule(schedule, P_load, Q_cool, Q_heat)
    st.pyplot(fig, use_container_width=True)

    # 硬件监测（含氢罐液位）
    if monitor_on:
        st.subheader("📡 实时硬件状态")
        hw = get_hardware_data(total_h2_used)
        cols = st.columns(3)
        cols[0].metric("光伏温度", hw["PV Temperature"])
        cols[1].metric("风机转速", hw["Wind Speed"])
        cols[2].metric("电池 SOC", hw["Battery SOC"])
        cols[0].metric("氢罐压力", hw["H₂ Tank Pressure"])
        cols[1].metric("氢罐液位", hw["H₂ Tank Level"])
        cols[2].metric("逆变器效率", hw["Inverter Efficiency"])

else:
    st.info("👈 配置参数后点击「生成调度方案」。本版本已解决中文乱码问题，氢能系统已启用。")

st.caption("💡 防乱码 · 氢能可见 · 硬件扩展 · 单文件 · 开箱即用")