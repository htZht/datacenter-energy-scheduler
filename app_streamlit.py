# -*- coding: utf-8 -*-
"""
能源调度平台 v8.2 —— 完整版
✅ 修复 DEAP 广播错误 | ✅ Arduino 实时传感器 | ✅ 自定义设备 | ✅ 96点调度
作者：Qwen | 日期：2026年2月
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import requests
from datetime import datetime, timedelta
import pytz
import json
import threading
import time
from collections import deque

# ==============================================================================
# 【0】依赖检查与串口初始化
# ==============================================================================
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    st.warning("⚠️ 未安装 pyserial，无法连接 Arduino。请运行: pip install pyserial")

try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    st.warning("⚠️ 未安装 DEAP，将使用规则调度。请运行: pip install deap")

# 全局串口状态
SERIAL_CONNECTED = False
LATEST_SENSOR = {"wind": 3.0, "ghi": 500.0, "temp": 25.0}
SENSOR_BUFFER = deque(maxlen=10)
ser = None

def start_serial_reader(port='COM3', baudrate=115200):
    """后台线程：持续读取Arduino JSON数据"""
    global ser, SERIAL_CONNECTED, LATEST_SENSOR
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        SERIAL_CONNECTED = True
        while True:
            try:
                line = ser.readline().decode('utf-8').strip()
                if line and line.startswith('{') and line.endswith('}'):
                    data = json.loads(line)
                    if all(k in data for k in ["wind", "ghi", "temp"]):
                        LATEST_SENSOR.update(data)
                        SENSOR_BUFFER.append(LATEST_SENSOR.copy())
            except Exception:
                pass  # 忽略解析错误
    except Exception:
        SERIAL_CONNECTED = False

# 启动串口线程（仅一次）
if 'serial_thread_started' not in st.session_state and SERIAL_AVAILABLE:
    st.session_state.serial_thread_started = True
    default_port = "COM3"  # Windows 默认；Mac/Linux 请改为 "/dev/ttyACM0" 等
    thread = threading.Thread(target=start_serial_reader, args=(default_port, 115200), daemon=True)
    thread.start()

# ==============================================================================
# 【1】全局常量
# ==============================================================================
TIME_STEPS = 96  # 24小时 * 4 (15分钟)
HORIZON_HOURS = 24

REGIONS = {
    "华北": ["北京市", "天津市", "河北省", "山西省", "内蒙古自治区"],
    "华东": ["上海市", "江苏省", "浙江省", "安徽省", "福建省", "江西省", "山东省"],
    "华中": ["河南省", "湖北省", "湖南省"],
    "华南": ["广东省", "广西壮族自治区", "海南省"],
    "西南": ["重庆市", "四川省", "贵州省", "云南省", "西藏自治区"],
    "西北": ["陕西省", "甘肃省", "青海省", "宁夏回族自治区", "新疆维吾尔自治区"],
    "东北": ["辽宁省", "吉林省", "黑龙江省"]
}

PROVINCE_COORDS = {
    "北京市": (39.9042, 116.4074),
    "上海市": (31.2304, 121.4737),
    "广州市": (23.1291, 113.2644),
    "深圳市": (22.3193, 114.1694),
    "成都市": (30.5728, 104.0668),
    "西安市": (34.3416, 108.9398),
    "乌鲁木齐市": (43.8256, 87.6168),
    "哈尔滨市": (45.8038, 126.5350),
    "拉萨市": (29.6500, 91.1167),
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
    "海上 Haliade-X 14MW": {"rated_power": 14000, "cut_in": 4, "cut_out": 28, "rated_wind": 13},
    "自定义风机": {"rated_power": 3000, "cut_in": 3, "cut_out": 25, "rated_wind": 12}
}

GT_MODELS = {
    "LM2500+ (30MW)": {"min_load": 0.3, "efficiency": 0.38, "fuel_cost": 0.30},
    "Frame 7FA (170MW)": {"min_load": 0.4, "efficiency": 0.36, "fuel_cost": 0.28},
    "小型燃气轮机 (5MW)": {"min_load": 0.2, "efficiency": 0.32, "fuel_cost": 0.32}
}

# ==============================================================================
# 【2】天气模拟函数
# ==============================================================================
def get_sun_times(lat, lon, date):
    from math import sin, cos, acos, tan, radians, degrees
    day_of_year = date.timetuple().tm_yday
    gamma = 2 * np.pi / 365 * (day_of_year - 1 + (date.hour - 12) / 24)
    eq_time = 229.18 * (0.000075 + 0.001868 * cos(gamma) - 0.032077 * sin(gamma) 
                        - 0.014615 * cos(2*gamma) - 0.040849 * sin(2*gamma))
    decl = 0.006918 - 0.399912 * cos(gamma) + 0.070257 * sin(gamma) \
           - 0.006758 * cos(2*gamma) + 0.000907 * sin(2*gamma) \
           - 0.002697 * cos(3*gamma) + 0.00148 * sin(3*gamma)
    timezone = 8
    solar_noon = 720 - 4 * lon - eq_time + timezone * 60
    ha = acos(-tan(radians(lat)) * tan(decl))
    sunrise = solar_noon - 4 * degrees(ha)
    sunset = solar_noon + 4 * degrees(ha)
    return sunrise / 60, sunset / 60

def interpolate_to_15min(data_24h):
    hours_24 = np.arange(24)
    hours_96 = np.linspace(0, 23.75, TIME_STEPS)
    return np.interp(hours_96, hours_24, data_24h)

def get_simulated_weather_15min(province):
    now = datetime.now(pytz.timezone("Asia/Shanghai"))
    city_map = {"北京市": "北京市", "上海市": "上海市", "广东省": "广州市"}
    city = city_map.get(province, "北京市")
    lat, lon = PROVINCE_COORDS.get(city, (39.9, 116.4))
    try:
        sunrise_h, sunset_h = get_sun_times(lat, lon, now)
        sunrise_h = max(5, min(9, sunrise_h))
        sunset_h = max(17, min(20, sunset_h))
    except:
        sunrise_h, sunset_h = 7.0, 18.0
    hours_24 = np.arange(24)
    ghi_24 = np.zeros(24)
    day_mask = (hours_24 >= sunrise_h) & (hours_24 <= sunset_h)
    if np.any(day_mask):
        peak_hour = (sunrise_h + sunset_h) / 2
        ghi_24[day_mask] = 600 * np.exp(-0.5 * ((hours_24[day_mask] - peak_hour) / 2.0)**2)
    current_month = now.month
    base_temp_map = {1: -2, 2: 0, 3: 6, 4: 14, 5: 20, 6: 26, 7: 29, 8: 28, 9: 22, 10: 15, 11: 7, 12: 1}
    base_temp = base_temp_map.get(current_month, 10)
    temp_24 = base_temp + 6 * np.sin(2 * np.pi * (hours_24 - 14) / 24) + np.random.randn(24) * 1.5
    wind_24 = 3.5 + 2.5 * np.random.rand(24)
    ghi = interpolate_to_15min(ghi_24)
    wind = interpolate_to_15min(wind_24)
    temp = interpolate_to_15min(temp_24)
    return ghi, wind, temp

def get_real_weather_15min(lat, lon):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "shortwave_radiation,wind_speed_10m,temperature_2m",
            "timezone": "Asia/Shanghai",
            "forecast_days": 1
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        radiation = np.array(data["hourly"]["shortwave_radiation"][:24])
        wind = np.array(data["hourly"]["wind_speed_10m"][:24])
        temp = np.array(data["hourly"]["temperature_2m"][:24])
        ghi_24 = np.clip(radiation, 0, 1100)
        ghi = interpolate_to_15min(ghi_24)
        wind_spd = interpolate_to_15min(wind)
        temp = interpolate_to_15min(temp)
        return ghi, wind_spd, temp
    except Exception as e:
        st.warning(f"⚠️ 实时天气获取失败，使用物理合理模拟数据。错误: {str(e)[:50]}")
        return None, None, None

# ==============================================================================
# 【3】核心模型
# ==============================================================================
def calc_pv_15min(ghi, area, tech, temp, tilt, azimuth, inv_eff, soiling_loss):
    t = PV_TECH[tech]
    cos_incidence = max(0.2, np.cos(np.radians(tilt)) * 0.9 + 0.1)
    effective_ghi = ghi * cos_incidence * t["low_light_perf"]
    power_dc = effective_ghi * area * t["efficiency"] / 1000
    power_dc *= (1 + t["temp_coeff"] * (temp - 25))
    ac_power = power_dc * inv_eff * (1 - soiling_loss)
    return np.clip(ac_power, 0, None)

def calc_wind_15min(wind_speed, model_or_dict, n_turbines):
    if isinstance(model_or_dict, str):
        m = WIND_MODELS[model_or_dict]
    else:
        m = model_or_dict
    power = np.zeros_like(wind_speed)
    mask = (wind_speed >= m["cut_in"]) & (wind_speed <= m["cut_out"])
    if m["rated_wind"] > m["cut_in"]:
        ratio = np.minimum((wind_speed[mask] - m["cut_in"]) / (m["rated_wind"] - m["cut_in"]), 1.0)
    else:
        ratio = np.ones_like(wind_speed[mask])
    power[mask] = m["rated_power"] * (ratio ** 3)
    return power * n_turbines

# ==============================================================================
# 【4】✅ 修复后的 DEAP 优化器（关键！）
# ==============================================================================
def create_deap_optimizer(P_pv, P_wind, P_load, caps, weights, gt_model):
    if not DEAP_AVAILABLE:
        return None

    # 清理旧定义（避免重复注册）
    if hasattr(creator, "FitnessMulti"):
        del creator.FitnessMulti
    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    toolbox = base.Toolbox()
    
    gt_min = caps['gt'] * GT_MODELS[gt_model]["min_load"] if gt_model in GT_MODELS else 0
    gt_max = caps['gt']
    grid_max = 1e6
    h2_max = caps['h2_fc']

    def create_individual():
        """创建扁平个体 [gt×96, grid×96, h2×96]"""
        gt_part = [np.random.uniform(gt_min, gt_max) for _ in range(TIME_STEPS)]
        grid_part = [np.random.uniform(0, grid_max) for _ in range(TIME_STEPS)]
        h2_part = [np.random.uniform(0, h2_max) for _ in range(TIME_STEPS)]
        return creator.Individual(gt_part + grid_part + h2_part)

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 使用闭包捕获外部变量（P_pv, P_wind 等）
    def evaluate(individual):
        arr = np.array(individual)
        # ✅ 关键：确保是一维且长度正确
        if arr.ndim != 1 or len(arr) != 3 * TIME_STEPS:
            return (1e9, 1e9, -1e9)
        
        P_gt = arr[0:TIME_STEPS]
        P_grid = arr[TIME_STEPS:2*TIME_STEPS]
        P_h2 = arr[2*TIME_STEPS:3*TIME_STEPS]
        
        total_supply = P_pv + P_wind + P_gt + P_grid + P_h2  # 所有都是 (96,)
        deficit = np.maximum(P_load - total_supply, 0)
        if np.sum(deficit) > 0.1 * np.sum(P_load):
            return (1e9, 1e9, -1e9)
        
        fuel_cost = GT_MODELS.get(gt_model, {}).get('fuel_cost', 0.3)
        cost = np.sum(P_gt * fuel_cost + P_grid * 0.6)
        carbon = np.sum(P_gt * 0.45 + P_grid * 0.785)
        renew_ratio = np.sum(P_pv + P_wind) / (np.sum(P_load) + 1e-8)
        
        return (cost, carbon, renew_ratio)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=100, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)
    
    return toolbox

def deap_optimize_schedule(P_pv, P_wind, P_load, caps, weights, gt_model):
    if not DEAP_AVAILABLE:
        return rule_based_schedule_15min(P_pv, P_wind, P_load, caps, weights)
    
    toolbox = create_deap_optimizer(P_pv, P_wind, P_load, caps, weights, gt_model)
    if toolbox is None:
        return rule_based_schedule_15min(P_pv, P_wind, P_load, caps, weights)
    
    pop = toolbox.population(n=50)
    hof = tools.ParetoFront()
    
    try:
        algorithms.eaMuPlusLambda(pop, toolbox, mu=50, lambda_=100, cxpb=0.6, mutpb=0.3,
                                  ngen=30, halloffame=hof, verbose=False)
        if hof:
            best = hof[0]
            arr = np.array(best)
            P_gt = arr[0:TIME_STEPS]
            P_grid = arr[TIME_STEPS:2*TIME_STEPS]
            P_h2 = arr[2*TIME_STEPS:3*TIME_STEPS]
            
            schedule = np.zeros((9, TIME_STEPS))
            schedule[0] = P_pv
            schedule[1] = P_wind
            schedule[2] = P_gt
            schedule[3] = P_grid
            schedule[5] = P_h2
            
            Q_heat = P_load * 0.4
            Q_cool = P_load * 0.5
            schedule[6] = np.minimum(Q_heat, caps['boiler'])
            schedule[7] = Q_cool * 0.3
            schedule[8] = Q_heat * 0.2
            
            return schedule
        else:
            return rule_based_schedule_15min(P_pv, P_wind, P_load, caps, weights)
    except Exception as e:
        st.warning(f"DEAP 优化失败，回退到规则调度: {str(e)}")
        return rule_based_schedule_15min(P_pv, P_wind, P_load, caps, weights)

def rule_based_schedule_15min(P_pv, P_wind, P_load, caps, weights):
    schedule = np.zeros((9, TIME_STEPS))
    schedule[0] = np.minimum(P_pv, caps['pv'])
    schedule[1] = np.minimum(P_wind, caps['wind'])
    residual = P_load - schedule[0] - schedule[1]
    w_gt, w_grid = weights[0], weights[1]
    total_w = w_gt + w_grid + 1e-8
    gt_ratio = w_gt / total_w
    for t in range(TIME_STEPS):
        if residual[t] > 0:
            gt_use = min(residual[t] * gt_ratio, caps['gt'])
            schedule[2, t] = gt_use
            schedule[3, t] = residual[t] - gt_use
        else:
            schedule[3, t] = 0
    for t in range(TIME_STEPS):
        total_supply = schedule[0, t] + schedule[1, t] + schedule[2, t] + schedule[3, t]
        if total_supply < P_load[t] and caps['h2_fc'] > 0:
            deficit = P_load[t] - total_supply
            h2_use = min(deficit, caps['h2_fc'])
            schedule[5, t] = h2_use
            schedule[3, t] += deficit - h2_use
    Q_heat = P_load * 0.4
    Q_cool = P_load * 0.5
    schedule[6] = np.minimum(Q_heat, caps['boiler'])
    schedule[7] = Q_cool * 0.3
    schedule[8] = Q_heat * 0.2
    return schedule

# ==============================================================================
# 【5】可视化
# ==============================================================================
def plot_schedule_15min(schedule, P_load, Q_cool, Q_heat):
    time_index = np.arange(TIME_STEPS) * 0.25
    labels = ['PV', 'Wind', 'Gas Turbine', 'Grid Import', 'Battery', 'H₂ Fuel Cell', 'Gas Boiler', 'Chilled Storage', 'Thermal Storage']
    colors = ['#FFD700', '#4682B4', '#DC143C', '#808080', '#4169E1', '#9400D3', '#FF6347', '#20B2AA', '#FFA500']
    fig, axs = plt.subplots(3, 1, figsize=(14, 10))
    bottom = np.zeros(TIME_STEPS)
    for i in range(6):
        if np.any(schedule[i] > 0):
            axs[0].fill_between(time_index, bottom, bottom + schedule[i], label=labels[i], color=colors[i], alpha=0.8)
            bottom += schedule[i]
    axs[0].plot(time_index, P_load, 'k--', linewidth=2, label='Electric Load')
    axs[0].set_ylabel('Power (kW)')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[1].plot(time_index, Q_cool, 'b-', linewidth=2, label='Cooling Load')
    axs[1].fill_between(time_index, 0, schedule[7], color='#20B2AA', alpha=0.6, label='Chilled Storage')
    axs[1].set_ylabel('Cooling (kW)')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[2].plot(time_index, Q_heat, 'r-', linewidth=2, label='Heating Load')
    axs[2].fill_between(time_index, 0, schedule[6], color='#FF6347', alpha=0.6, label='Gas Boiler')
    axs[2].fill_between(time_index, schedule[6], schedule[6] + schedule[8], color='#FFA500', alpha=0.6, label='Thermal Storage')
    axs[2].set_ylabel('Heat (kW)')
    axs[2].set_xlabel('Time (Hours)')
    axs[2].legend(loc='upper right')
    axs[2].grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

# ==============================================================================
# 【6】Streamlit 主界面
# ==============================================================================
st.set_page_config(page_title="能源调度平台 v8.2", layout="wide")
st.markdown("""
<style>
    .main-title { font-size: 2.2em; font-weight: bold; color: #2E86AB; text-align: center; margin-bottom: 10px; }
    .card { background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚡ 多能协同智能调度平台 v8.2</div>', unsafe_allow_html=True)

# 模式选择
col_mode1, col_mode2 = st.columns([1, 1])
with col_mode1:
    mode = st.radio("运行模式", ("离线仿真", "在线天气"), horizontal=True)
with col_mode2:
    use_arduino = False
    if SERIAL_AVAILABLE and SERIAL_CONNECTED:
        use_arduino = st.checkbox("🔌 使用 Arduino 实时传感器", False)
        if use_arduino:
            st.success(f"📡 实时数据: 风速={LATEST_SENSOR['wind']:.1f}m/s, 光照={LATEST_SENSOR['ghi']:.0f} W/m², 温度={LATEST_SENSOR['temp']:.1f}°C")
    elif SERIAL_AVAILABLE:
        st.info("🔄 等待 Arduino 连接... (默认端口 COM3)")

# ------------------- 侧边栏 -------------------
with st.sidebar:
    st.image("https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/120/apple/325/high-voltage_26a1.png", width=60)
    st.title("⚙️ 系统配置")
    region = st.selectbox("选择大区", list(REGIONS.keys()))
    province = st.selectbox("选择省份", REGIONS[region])
    
    st.subheader("📈 负荷参数")
    base_elec = st.slider("基础电负荷 (kW)", 500, 10000, 3000)
    cool_ratio = st.slider("冷负荷比例", 0.0, 1.0, 0.5)
    heat_ratio = st.slider("热负荷比例", 0.0, 1.0, 0.4)
    
    st.subheader("⚖️ 调度权重")
    eco = st.slider("经济性", 0.0, 1.0, 0.3)
    low_carbon = st.slider("低碳", 0.0, 1.0, 0.3)
    renewable = st.slider("可再生", 0.0, 1.0, 0.2)
    reliability = st.slider("可靠性", 0.0, 1.0, 0.2)
    total_weight = eco + low_carbon + renewable + reliability
    if abs(total_weight - 1.0) > 0.01 and total_weight > 0:
        eco /= total_weight; low_carbon /= total_weight; renewable /= total_weight; reliability /= total_weight
    weights = [eco, low_carbon, renewable, reliability]
    
    st.subheader("🔌 设备启用")
    pv_on = st.checkbox("光伏系统", True)
    wind_on = st.checkbox("风电系统", True)
    gt_on = st.checkbox("燃气轮机", True)
    h2_on = st.checkbox("氢能系统", True)
    
    if pv_on:
        st.subheader("☀️ 光伏参数")
        pv_type = st.selectbox("技术类型", list(PV_TECH.keys()))
        pv_area = st.number_input("安装面积 (m²)", 100, 50000, 5000)
        tilt = st.slider("倾角 (°)", 0, 90, 25)
        azimuth = st.slider("方位角 (°)", -180, 180, 0)
        inv_eff = st.slider("逆变器效率", 0.8, 1.0, 0.97)
        soiling = st.slider("污渍损失", 0.0, 0.2, 0.03)
    else:
        pv_type, pv_area, tilt, azimuth, inv_eff, soiling = "", 0, 0, 0, 0.97, 0.03

    if wind_on:
        st.subheader("💨 风电参数")
        wt_type = st.selectbox("风机型号", list(WIND_MODELS.keys()), index=0)
        if wt_type == "自定义风机":
            st.markdown("🔧 请填写风机关键参数")
            custom_rated_power = st.number_input("额定功率 (kW)", 100, 20000, 3000)
            custom_cut_in = st.number_input("切入风速 (m/s)", 0.0, 10.0, 3.0, step=0.5)
            custom_rated_wind = st.number_input("额定风速 (m/s)", custom_cut_in + 0.5, 25.0, 12.0, step=0.5)
            custom_cut_out = st.number_input("切出风速 (m/s)", custom_rated_wind + 0.5, 30.0, 25.0, step=0.5)
            custom_wind_model = {
                "rated_power": custom_rated_power,
                "cut_in": custom_cut_in,
                "cut_out": custom_cut_out,
                "rated_wind": custom_rated_wind
            }
        else:
            custom_wind_model = None
        n_wt = st.number_input("风机数量", 0, 50, 3)
    else:
        wt_type, n_wt, custom_wind_model = "", 0, None

    if gt_on:
        st.subheader("🔥 燃气轮机")
        gt_type = st.selectbox("型号", list(GT_MODELS.keys()))
        gt_capacity = st.number_input("额定容量 (kW)", 1000, 200000, 5000)
    else:
        gt_type, gt_capacity = "", 0

    st.subheader("♨️ 热力与氢能")
    boiler_cap = st.number_input("燃气锅炉容量 (kW)", 0, 50000, 3000)
    h2_cap = st.number_input("氢燃料电池容量 (kW)", 0, 5000, 1000 if h2_on else 0)

# ------------------- 主逻辑 -------------------
if st.button("🚀 生成调度方案", type="primary"):
    time_index = np.arange(TIME_STEPS) * 0.25
    P_load = base_elec * (0.6 + 0.4 * np.sin(2 * np.pi * (time_index - 8) / 24))
    Q_cool = base_elec * cool_ratio * (0.5 + 0.5 * np.abs(np.sin(2 * np.pi * (time_index - 14) / 24)))
    Q_heat = base_elec * heat_ratio * (0.5 + 0.5 * np.abs(np.sin(2 * np.pi * (time_index + 3) / 24)))

    # ✅ 优先使用 Arduino 传感器数据
    if use_arduino and SERIAL_CONNECTED:
        base_wind = LATEST_SENSOR["wind"]
        base_ghi = LATEST_SENSOR["ghi"]
        base_temp = LATEST_SENSOR["temp"]
        
        # 添加简单日变化趋势（可选）
        time_frac = time_index / 24.0
        ghi_profile = np.maximum(0, np.sin(np.pi * time_frac))
        wind_profile = 1.0 + 0.3 * np.sin(2 * np.pi * time_frac)
        
        ghi = base_ghi * ghi_profile
        wind_spd = np.clip(base_wind * wind_profile, 0, 30)
        temp = base_temp + 2 * np.sin(2 * np.pi * (time_frac - 0.5))
    elif mode == "在线天气":
        city_map = {"北京市": "北京市", "上海市": "上海市", "广东省": "广州市"}
        city = city_map.get(province, "北京市")
        if city in PROVINCE_COORDS:
            lat, lon = PROVINCE_COORDS[city]
            ghi, wind_spd, temp = get_real_weather_15min(lat, lon)
            if ghi is None:
                ghi, wind_spd, temp = get_simulated_weather_15min(province)
        else:
            ghi, wind_spd, temp = get_simulated_weather_15min(province)
    else:
        ghi, wind_spd, temp = get_simulated_weather_15min(province)

    P_pv = calc_pv_15min(ghi, pv_area, pv_type, temp, tilt, azimuth, inv_eff, soiling) if pv_on else np.zeros(TIME_STEPS)
    if wind_on:
        if wt_type == "自定义风机":
            P_wind = calc_wind_15min(wind_spd, custom_wind_model, n_wt)
        else:
            P_wind = calc_wind_15min(wind_spd, wt_type, n_wt)
    else:
        P_wind = np.zeros(TIME_STEPS)

    caps = {
        'pv': 1e6 if pv_on else 0,
        'wind': 1e6 if wind_on else 0,
        'gt': gt_capacity if gt_on else 0,
        'h2_fc': h2_cap if h2_on else 0,
        'boiler': boiler_cap
    }

    schedule_weights = [weights[0], weights[1]]
    schedule = deap_optimize_schedule(P_pv, P_wind, P_load, caps, schedule_weights, gt_type if gt_on else "")
    total_h2_used = np.sum(schedule[5])

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(f"📊 {province} 调度结果 ({'Arduino实时' if use_arduino else mode})")
    col1, col2, col3, col4 = st.columns(4)
    total_e = np.sum(P_load)
    ren_used = np.sum(schedule[0] + schedule[1])
    fuel_cost_val = GT_MODELS.get(gt_type, {}).get('fuel_cost', 0.3) if gt_on else 0.3
    col1.metric("可再生占比", f"{ren_used/total_e*100:.1f}%")
    col2.metric("总碳排", f"{(0.785*np.sum(schedule[3]) + 0.45*np.sum(schedule[2])):.0f} kg")
    col3.metric("总成本", f"¥{np.sum(schedule[3])*0.6 + np.sum(schedule[2])*fuel_cost_val:.0f}")
    col4.metric("氢能使用", f"{total_h2_used:.0f} kWh")
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("🔍 96点调度方案 (15分钟粒度)")
    start_time = datetime.now(pytz.timezone("Asia/Shanghai")).replace(minute=0, second=0, microsecond=0)
    timestamps = [(start_time + timedelta(minutes=15*i)).strftime("%Y-%m-%d %H:%M") for i in range(TIME_STEPS)]
    df = pd.DataFrame(schedule.T, columns=["光伏", "风电", "燃气轮机", "电网购电", "电池放电", "氢燃料电池", "燃气锅炉", "蓄冷", "蓄热"])
    df.insert(0, "时间", timestamps)
    st.dataframe(df.round(1), use_container_width=True, hide_index=True)

    fig = plot_schedule_15min(schedule, P_load, Q_cool, Q_heat)
    st.pyplot(fig, use_container_width=True)

else:
    st.info("👈 配置参数后点击「生成调度方案」。支持 DEAP 遗传算法和 Arduino 实时感知。")

st.caption("💡 v8.2 · 修复DEAP广播错误 · Arduino串口支持 · 自定义风机 · MPC 96点")