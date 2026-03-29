# -*- coding: utf-8 -*-
"""
能源调度平台 v10.0 (UI 重构版) - 修复版
✅ 修复已知 BUG | ✅ 按图片样式重构 UI | ✅ 集成 DeepSeek AI | ✅ 多轮对话记忆
✅ 修复 NameError: mode 未定义 | ✅ 增加天气模式选择 | ✅ 修复中文乱码
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ========== 修复中文乱码 ==========
# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用于正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号

import pandas as pd
import requests
from datetime import datetime, timedelta
import pytz
import json
import threading
import time
from collections import deque
import math

# 硬件配置
use_arduino = False  # 默认不使用硬件模拟数据
SERIAL_AVAILABLE = False
SERIAL_CONNECTED = False
LATEST_SENSOR = {"wind": 0, "ghi": 0, "temp": 25}

# 尝试导入 openai 库
try:
    from openai import OpenAI
    OPENAI_LIB_AVAILABLE = True
except ImportError:
    OPENAI_LIB_AVAILABLE = False

# ==============================================================================
# 【0】依赖检查与串口初始化
# ==============================================================================
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False

# 全局状态
if 'serial_port' not in st.session_state:
    st.session_state.serial_port = "COM3"

SERIAL_CONNECTED = False
LATEST_SENSOR = {"wind": 3.0, "ghi": 500.0, "temp": 25.0}
SENSOR_BUFFER = deque(maxlen=10)
ser = None

def serial_reader(port, baudrate=115200):
    global ser, SERIAL_CONNECTED, LATEST_SENSOR
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        SERIAL_CONNECTED = True
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line and line.startswith('{') and line.endswith('}'):
                try:
                    data = json.loads(line)
                    if all(k in data for k in ["wind", "ghi", "temp"]):
                        LATEST_SENSOR.update(data)
                        SENSOR_BUFFER.append(LATEST_SENSOR.copy())
                except:
                    pass
    except Exception:
        SERIAL_CONNECTED = False

if 'serial_thread_started' not in st.session_state and SERIAL_AVAILABLE:
    st.session_state.serial_thread_started = True
    thread = threading.Thread(target=serial_reader, args=(st.session_state.serial_port, 115200), daemon=True)
    thread.start()

# ==============================================================================
# 【0.5】DeepSeek AI 集成
# ==============================================================================
def get_deepseek_client(api_key):
    if not OPENAI_LIB_AVAILABLE:
        return None, "❌ 未安装 openai 库。请运行: pip install openai"
    if not api_key or api_key == "":
        return None, "⚠️ 请先在左侧侧边栏输入你的 DeepSeek API Key"
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        return client, None
    except Exception as e:
        return None, f"❌ 客户端初始化失败: {str(e)}"

def query_deepseek(messages, api_key):
    client, error = get_deepseek_client(api_key)
    if error:
        return error
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=1500,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg:
            return "❌ API Key 无效或已过期，请检查后重试。"
        elif "429" in err_msg:
            return "⚠️ 请求过于频繁或余额不足，请稍后再试。"
        elif "connection" in err_msg.lower() or "timeout" in err_msg.lower():
            return "❌ 网络连接超时，请检查网络环境。"
        else:
            return f"❌ 发生错误: {err_msg}"

# ==============================================================================
# 【1】全局常量与模型定义
# ==============================================================================
TIME_STEPS = 96
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

# 省份到代表性城市的映射（用于获取天气）
PROVINCE_TO_CITY = {
    "北京市": "北京市", "天津市": "天津市", "河北省": "石家庄市", "山西省": "太原市",
    "内蒙古自治区": "呼和浩特市", "上海市": "上海市", "江苏省": "南京市", "浙江省": "杭州市",
    "安徽省": "合肥市", "福建省": "福州市", "江西省": "南昌市", "山东省": "济南市",
    "河南省": "郑州市", "湖北省": "武汉市", "湖南省": "长沙市", "广东省": "广州市",
    "广西壮族自治区": "南宁市", "海南省": "海口市", "重庆市": "重庆市", "四川省": "成都市",
    "贵州省": "贵阳市", "云南省": "昆明市", "西藏自治区": "拉萨市", "陕西省": "西安市",
    "甘肃省": "兰州市", "青海省": "西宁市", "宁夏回族自治区": "银川市", "新疆维吾尔自治区": "乌鲁木齐市",
    "辽宁省": "沈阳市", "吉林省": "长春市", "黑龙江省": "哈尔滨市"
}

# 城市经纬度
CITY_COORDS = {
    "北京市": (39.9042, 116.4074), "天津市": (39.3434, 117.3616), "石家庄市": (38.0423, 114.5149),
    "太原市": (37.8706, 112.5624), "呼和浩特市": (40.8175, 111.6708), "上海市": (31.2304, 121.4737),
    "南京市": (32.0603, 118.7969), "杭州市": (30.2741, 120.1551), "合肥市": (31.8206, 117.2272),
    "福州市": (26.0745, 119.2965), "南昌市": (28.6765, 115.8922), "济南市": (36.6512, 117.1201),
    "郑州市": (34.7473, 113.6249), "武汉市": (30.5928, 114.3055), "长沙市": (28.2282, 112.9388),
    "广州市": (23.1291, 113.2644), "南宁市": (22.824, 108.32), "海口市": (20.044, 110.2),
    "重庆市": (29.4316, 106.9123), "成都市": (30.5728, 104.0668), "贵阳市": (26.647, 106.63),
    "昆明市": (24.880, 102.832), "拉萨市": (29.6500, 91.1167), "西安市": (34.3416, 108.9398),
    "兰州市": (36.064, 103.834), "西宁市": (36.623, 101.780), "银川市": (38.487, 106.230),
    "乌鲁木齐市": (43.8256, 87.6168), "沈阳市": (41.805, 123.431), "长春市": (43.817, 125.323),
    "哈尔滨市": (45.8038, 126.5350)
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
# 【2】天气与物理模型
# ==============================================================================
def get_sun_times(lat, lon, date):
    from math import sin, cos, acos, tan, radians, degrees
    day_of_year = date.timetuple().tm_yday
    gamma = 2 * np.pi / 365 * (day_of_year - 1 + (date.hour - 12) / 24)
    eq_time = 229.18 * (0.000075 + 0.001868 * cos(gamma) - 0.032077 * sin(gamma)
                        - 0.014615 * cos(2 * gamma) - 0.040849 * sin(2 * gamma))
    decl = 0.006918 - 0.399912 * cos(gamma) + 0.070257 * sin(gamma) \
           - 0.006758 * cos(2 * gamma) + 0.000907 * sin(2 * gamma) \
           - 0.002697 * cos(3 * gamma) + 0.00148 * sin(3 * gamma)
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
    today = now.date()
    city = PROVINCE_TO_CITY.get(province, "北京市")
    lat, lon = CITY_COORDS.get(city, (39.9, 116.4))
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
        ghi_24[day_mask] = 600 * np.exp(-0.5 * ((hours_24[day_mask] - peak_hour) / 2.0) ** 2)
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
            "latitude": lat, "longitude": lon,
            "hourly": "shortwave_radiation,wind_speed_10m,temperature_2m",
            "timezone": "Asia/Shanghai", "forecast_days": 1
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
        st.warning(f"⚠️ 实时天气获取失败，使用模拟数据。错误: {str(e)[:50]}")
        return None, None, None

# ==============================================================================
# 【3】核心发电模型
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
# 【4】DEAP 优化器
# ==============================================================================
def create_deap_optimizer(P_pv, P_wind, P_load, caps, weights, gt_model):
    if not DEAP_AVAILABLE:
        return None
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
        gt_part = [np.random.uniform(gt_min, gt_max) for _ in range(TIME_STEPS)]
        grid_part = [np.random.uniform(0, grid_max) for _ in range(TIME_STEPS)]
        h2_part = [np.random.uniform(0, h2_max) for _ in range(TIME_STEPS)]
        return creator.Individual(gt_part + grid_part + h2_part)

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        arr = np.array(individual)
        if arr.ndim != 1 or len(arr) != 3 * TIME_STEPS:
            return (1e9, 1e9, -1e9)
        P_gt = arr[0:TIME_STEPS]
        P_grid = arr[TIME_STEPS:2 * TIME_STEPS]
        P_h2 = arr[2 * TIME_STEPS:3 * TIME_STEPS]
        total_supply = P_pv + P_wind + P_gt + P_grid + P_h2
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
            arr = np.array(best).flatten()
            P_gt = arr[0:TIME_STEPS]
            P_grid = arr[TIME_STEPS:2 * TIME_STEPS]
            P_h2 = arr[2 * TIME_STEPS:3 * TIME_STEPS]
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
    if caps['gt'] == 0:
        gt_ratio = 0
    else:
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
# 【6】Streamlit 主界面（UI 重构版）
# ==============================================================================
st.set_page_config(page_title="能源调度平台 v10.0 | 多能协同智能调度", layout="wide")

# 自定义 CSS
st.markdown("""
<style>
    .main-title { font-size: 2.2em; font-weight: bold; color: #2E86AB; text-align: center; margin-bottom: 10px; }
    .card { background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .ai-container { background-color: #f0f8ff; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 5px solid #4A90E2; }
    .device-card { background-color: #ffffff; border-radius: 12px; padding: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); text-align: center; margin-bottom: 10px; }
    .device-title { font-size: 1.2em; font-weight: bold; margin-bottom: 8px; }
    .metric-value { font-size: 1.5em; font-weight: bold; color: #2c3e50; }
    .metric-label { font-size: 0.8em; color: #7f8c8d; }
    hr { margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚡ 多能协同智能调度平台 v10.0</div>', unsafe_allow_html=True)

# ------------------- 侧边栏 -------------------
with st.sidebar:
    st.image("https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/120/apple/325/high-voltage_26a1.png", width=60)
    st.title("⚙️ 系统配置")

    # AI 配置
    st.subheader("🤖 AI 助手配置")
    if not OPENAI_LIB_AVAILABLE:
        st.error("❌ 未检测到 openai 库。\n请在终端运行:\n`pip install openai -i https://mirrors.aliyun.com/pypi/simple/`")
        api_key = ""
    else:
        api_key = st.text_input("DeepSeek API Key", type="password", placeholder="sk-...", help="在 deepseek.com 获取")
        if api_key:
            st.success("✅ API Key 已设置")
        else:
            st.info("💡 输入 Key 后即可启用 AI 问答")

    st.divider()

    # 天气模式选择
    st.subheader("🌤️ 天气数据来源")
    weather_mode = st.radio(
        "选择天气模式",
        options=["模拟天气", "实时网络天气", "Arduino 串口数据"],
        index=0,
        help="模拟天气使用算法生成；实时网络天气需要联网；Arduino串口需要连接硬件"
    )
    if weather_mode == "Arduino 串口数据" and not SERIAL_AVAILABLE:
        st.warning("⚠️ 未安装 pyserial，无法使用串口数据。请运行 `pip install pyserial` 安装。")
        weather_mode = "模拟天气"

    st.divider()

    if not SERIAL_AVAILABLE:
        st.info("💡 安装 pyserial 以启用 Arduino：\n```\npip install pyserial\n```")

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
        eco /= total_weight
        low_carbon /= total_weight
        renewable /= total_weight
        reliability /= total_weight
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

# ------------------- 主界面顶部栏 -------------------
col_date, col_login, col_customer = st.columns(3)
with col_date:
    now = datetime.now(pytz.timezone("Asia/Shanghai"))
    st.write(f"📅 {now.strftime('%Y/%m/%d')}   {now.strftime('%H:%M:%S')}")
with col_login:
    st.write("✅ 已登录")
with col_customer:
    st.write(f"🏢 客户类型：{province}")

st.divider()

# ------------------- 设备卡片 -------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    with st.container(border=True):
        st.markdown('<div class="device-title">☀️ 光伏系统</div>', unsafe_allow_html=True)
        if pv_on:
            pv_capacity = pv_area * PV_TECH[pv_type]['efficiency']
            st.markdown(f'<div class="metric-value">1 个</div><div class="metric-label">运行数量</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{pv_capacity:.0f} kW</div><div class="metric-label">装机容量</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-value">未启用</div>', unsafe_allow_html=True)

with col2:
    with st.container(border=True):
        st.markdown('<div class="device-title">💨 风机系统</div>', unsafe_allow_html=True)
        if wind_on:
            if wt_type == "自定义风机":
                wind_capacity = n_wt * custom_wind_model['rated_power']
            else:
                wind_capacity = n_wt * WIND_MODELS[wt_type]['rated_power']
            st.markdown(f'<div class="metric-value">{n_wt} 个</div><div class="metric-label">运行数量</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{wind_capacity:.0f} kW</div><div class="metric-label">装机容量</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-value">未启用</div>', unsafe_allow_html=True)

with col3:
    with st.container(border=True):
        st.markdown('<div class="device-title">🔋 储能系统</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">20 个</div><div class="metric-label">运行数量</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">4850 kW</div><div class="metric-label">装机容量</div>', unsafe_allow_html=True)

with col4:
    with st.container(border=True):
        st.markdown('<div class="device-title">🔌 充电桩系统</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">快充 25<br>慢充 12</div><div class="metric-label">运行数量</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">5 个</div><div class="metric-label">故障数量</div>', unsafe_allow_html=True)

st.divider()

# ------------------- 左右两列：当月效益分析 & 实时告警 -------------------
col_left, col_right = st.columns(2)

with col_left:
    with st.container(border=True):
        st.subheader("📊 当月效益分析")
        if 'schedule' in st.session_state:
            total_gen = np.sum(st.session_state.schedule[0] + st.session_state.schedule[1] + st.session_state.schedule[2] + st.session_state.schedule[5])
            total_elec = np.sum(st.session_state.P_load)
            total_buy = np.sum(st.session_state.schedule[3])
            renew_ratio = np.sum(st.session_state.schedule[0] + st.session_state.schedule[1]) / (total_elec + 1e-8) * 100
            st.metric("发电量 (kWh)", f"{total_gen:.2f}")
            st.metric("用电量 (kWh)", f"{total_elec:.2f}")
            st.metric("购电量 (kWh)", f"{total_buy:.2f}")
            st.metric("新能源占比 (%)", f"{renew_ratio:.1f}%")
        else:
            st.info("点击下方「生成调度方案」按钮查看数据")

with col_right:
    with st.container(border=True):
        st.subheader("⚠️ 实时告警")
        st.write("**注意:** 4.9")
        st.write("**严重:** 92.00")
        st.write("**待处理:** 16")
        st.write("**已处理:** 4540")

st.divider()

# ------------------- 图表区域 -------------------
st.subheader("📈 能源趋势分析")

# 生成调度方案按钮
if st.button("🚀 生成调度方案", type="primary", use_container_width=True):
    # 计算负荷曲线
    time_index = np.arange(TIME_STEPS) * 0.25
    P_load = base_elec * (0.6 + 0.4 * np.sin(2 * np.pi * (time_index - 8) / 24))
    Q_cool = base_elec * cool_ratio * (0.5 + 0.5 * np.abs(np.sin(2 * np.pi * (time_index - 14) / 24)))
    Q_heat = base_elec * heat_ratio * (0.5 + 0.5 * np.abs(np.sin(2 * np.pi * (time_index + 3) / 24)))

    # 获取天气数据
    if weather_mode == "Arduino 串口数据" and SERIAL_AVAILABLE and SERIAL_CONNECTED:
        base_wind = LATEST_SENSOR["wind"]
        base_ghi = LATEST_SENSOR["ghi"]
        base_temp = LATEST_SENSOR["temp"]
        time_frac = np.linspace(0, 24, TIME_STEPS) / 24
        ghi_profile = np.maximum(0, np.sin(np.pi * time_frac))
        wind_profile = 1.0 + 0.3 * np.sin(2 * np.pi * time_frac)
        ghi = base_ghi * ghi_profile
        wind_spd = np.clip(base_wind * wind_profile, 0, 30)
        temp = base_temp + 2 * np.sin(2 * np.pi * (time_frac - 0.5))
    elif weather_mode == "实时网络天气":
        city = PROVINCE_TO_CITY.get(province, "北京市")
        lat, lon = CITY_COORDS.get(city, (39.9042, 116.4074))
        ghi, wind_spd, temp = get_real_weather_15min(lat, lon)
        if ghi is None:
            # 回退到模拟天气
            ghi, wind_spd, temp = get_simulated_weather_15min(province)
    else:  # 模拟天气
        ghi, wind_spd, temp = get_simulated_weather_15min(province)

    # 发电功率
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

    # 存储到 session_state
    st.session_state.schedule = schedule
    st.session_state.P_load = P_load
    st.session_state.Q_cool = Q_cool
    st.session_state.Q_heat = Q_heat
    st.session_state.total_h2_used = total_h2_used

    st.rerun()

# 如果有调度结果，显示图表
if 'schedule' in st.session_state:
    schedule = st.session_state.schedule
    P_load = st.session_state.P_load
    Q_cool = st.session_state.Q_cool
    Q_heat = st.session_state.Q_heat

    # 第一行图表：光伏发电功率、储能放电量及收益趋势
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("☀️ 光伏发电功率")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        time_axis = np.arange(TIME_STEPS) * 0.25
        ax1.plot(time_axis, schedule[0], label='光伏功率 (kW)', color='gold')
        ax1.set_xlabel('小时')
        ax1.set_ylabel('功率 (kW)')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend()
        st.pyplot(fig1)

    with col_chart2:
        st.subheader("🔋 储能放电量及收益趋势")
        discharge = schedule[5]  # 氢能放电
        revenue = discharge * 0.8  # 假设每度电收益0.8元
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(time_axis, discharge, label='放电量 (kWh)', color='blue')
        ax2.plot(time_axis, revenue, label='收益 (元)', color='green')
        ax2.set_xlabel('小时')
        ax2.set_ylabel('数值')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()
        st.pyplot(fig2)

    # 第二行图表：电网交换功率、消费端用电结构
    col_chart3, col_chart4 = st.columns(2)
    with col_chart3:
        st.subheader("🔌 电网交换功率")
        grid_power = schedule[3]  # 购电
        grid_export = np.zeros_like(grid_power)  # 卖电（示例）
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.plot(time_axis, grid_power, label='购电功率 (kW)', color='red')
        ax3.plot(time_axis, grid_export, label='售电功率 (kW)', color='orange')
        ax3.set_xlabel('小时')
        ax3.set_ylabel('功率 (kW)')
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.legend()
        st.pyplot(fig3)

    with col_chart4:
        st.subheader("📊 消费端用电结构")
        total_elec = np.sum(schedule[0] + schedule[1] + schedule[2] + schedule[3] + schedule[5])
        pv_ratio = np.sum(schedule[0]) / total_elec * 100
        wind_ratio = np.sum(schedule[1]) / total_elec * 100
        gt_ratio = np.sum(schedule[2]) / total_elec * 100
        grid_ratio = np.sum(schedule[3]) / total_elec * 100
        h2_ratio = np.sum(schedule[5]) / total_elec * 100
        labels = ['光伏', '风电', '燃机', '市电', '氢能']
        sizes = [pv_ratio, wind_ratio, gt_ratio, grid_ratio, h2_ratio]
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        ax4.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax4.axis('equal')
        st.pyplot(fig4)

    # 第三行：当月水电统计、节能减排效益分析
    col_chart5, col_chart6 = st.columns(2)
    with col_chart5:
        st.subheader("💧 当月水电统计")
        hydro_month = [80, 60, 40, 20]
        months = ['W1', 'W2', 'W3', 'W4']
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        ax5.bar(months, hydro_month, color='cyan')
        ax5.set_ylabel('吨')
        ax5.set_title('月度水电消耗')
        st.pyplot(fig5)

    with col_chart6:
        st.subheader("🌱 节能减排效益分析")
        total_renew = np.sum(schedule[0] + schedule[1])
        co2_reduction = total_renew * 0.8  # 每度电减碳0.8kg
        trees = co2_reduction / 20
        coal_saved = co2_reduction / 2.7
        st.metric("等效植树量", f"{trees:.0f} 棵")
        st.metric("节约标煤", f"{coal_saved:.2f} 吨")
        st.metric("减少CO₂", f"{co2_reduction:.2f} 吨")
        st.metric("减少SO₂", f"{co2_reduction * 0.03:.2f} 吨")

else:
    st.info("点击上方「生成调度方案」按钮，系统将自动计算最优调度并展示图表。")

# ------------------- AI 助手区域 -------------------
st.divider()
st.subheader("🤖 AI 能源顾问")

if not OPENAI_LIB_AVAILABLE:
    st.error("❌ 缺少 openai 库，无法启动 AI 助手。请在终端运行安装命令。")
elif not api_key:
    st.warning("⚠️ 请在左侧侧边栏输入 **DeepSeek API Key** 以启用 AI 助手。")
else:
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("询问关于当前调度方案的问题..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        system_context = f"""
        你是一个专业的能源调度专家。用户正在使用一个多能协同调度平台。
        当前系统配置如下：
        - 地区: {province}
        - 基础电负荷: {base_elec} kW
        - 启用的设备: {'光伏' if pv_on else ''} {'风电' if wind_on else ''} {'燃气轮机' if gt_on else ''} {'氢能' if h2_on else ''}
        - 调度偏好: 经济性({weights[0]:.1f}), 低碳({weights[1]:.1f}), 可再生({weights[2]:.1f})
        请基于以上背景回答用户的问题。如果用户问的是具体数值计算，请提醒用户先生成调度方案。
        """
        messages_for_api = [{"role": "system", "content": system_context}] + st.session_state.chat_messages

        with st.chat_message("assistant"):
            with st.spinner("🤖 AI 正在分析数据..."):
                response_text = query_deepseek(messages_for_api, api_key)
                st.markdown(response_text)

        st.session_state.chat_messages.append({"role": "assistant", "content": response_text})

st.caption("💡 v10.0 · 修复版 · 支持多轮对话 · 天气模式可选 · 修复中文乱码")