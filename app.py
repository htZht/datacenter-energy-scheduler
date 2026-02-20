# app.py
from flask import Flask, request, jsonify
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from location_utils import parse_location_input, get_regional_config
from device_config import get_device_config
from config import build_config
from pv_model import pv_forecast_from_location, pv_forecast_default
from wind_model import wind_power_forecast, wind_forecast_default
from objectives import economic_cost, carbon_emission, negative_ESI, weighted_objective
from emergy_analysis import calculate_ESI
from optimizer import optimize_single_objective
from plot_results import plot_scheduling
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def generate_load_from_input(load_params):
    """根据前端输入生成24小时负荷曲线"""
    base_el = load_params.get('base_el', 200.0)
    peak_cool = load_params.get('peak_cool', 150.0)
    heat_load = load_params.get('heat_load', 20.0)
    
    hours = np.arange(24)
    P_el = base_el + 10 * np.sin((hours - 6) * np.pi / 12)
    Q_cool = np.maximum(peak_cool * (0.7 + 0.3 * np.sin((hours - 14) * np.pi / 12)), peak_cool * 0.5)
    Q_heat = np.full(24, heat_load)
    return P_el, Q_cool, Q_heat

@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json
    mode = data.get('mode', 'simulation')  # 'simulation' or 'hardware'
    location = data.get('location', '上海')
    weights = data.get('weights', {'w1': 0.5, 'w2': 0.3, 'w3': 0.2})
    device_config_input = data.get('device_config', {})
    load_profile_input = data.get('load_profile', {})

    # === 构建设备配置 ===
    default_device = {
        'pv_area': 400,
        'pv_efficiency': 0.175,
        'boiler_max': 200,
        'chiller_elec_max': 150,
        'chiller_abs_max': 100,
        'bess_capacity': 500,
        'bess_max_power': 100,
        'tes_capacity': 2000
    }
    device_config = {k: device_config_input.get(k, v) for k, v in default_device.items()}
    
    # === 地理与全局配置 ===
    lat, lon, _ = parse_location_input(location)
    regional_config = get_regional_config(lat, lon)
    global_config = build_config(device_config, regional_config)

    # === 获取风光数据 ===
    if mode == 'hardware':
        try:
            from sensor_reader import generate_hardware_profiles
            P_pv_hw, P_wind_hw, _, _, _ = generate_hardware_profiles()
            P_pv = P_pv_hw
            P_wind = P_wind_hw
        except Exception as e:
            print(f"Hardware fallback: {e}")
            P_pv = pv_forecast_from_location(lat, lon, 
                pv_area=device_config['pv_area'],
                pv_eff=device_config['pv_efficiency']
            )
            P_wind = wind_forecast_default()
    else:
        try:
            P_pv = pv_forecast_from_location(lat, lon,
                pv_area=device_config['pv_area'],
                pv_eff=device_config['pv_efficiency']
            )
        except:
            P_pv = pv_forecast_default(global_config)
        P_wind = wind_forecast_default()

    # === 获取负荷 ===
    P_el, Q_cool, Q_heat = generate_load_from_input(load_profile_input)

    # === 优化 ===
    T = 24
    n_vars = 9 * T
    obj_func = lambda x: weighted_objective(x, P_pv, P_wind, P_el, Q_cool, Q_heat, global_config, **weights)
    x_opt = optimize_single_objective(obj_func, n_vars, bounds=(0, 500), n_gen=40)

    # === 计算指标 ===
    cost = economic_cost(x_opt, P_pv, P_wind, P_el, Q_cool, Q_heat, global_config)[0] * 365 / 1e4
    carbon = carbon_emission(x_opt, P_pv, P_wind, P_el, Q_cool, Q_heat, global_config)[0] * 365 / 1000
    ESI, _, _ = calculate_ESI(x_opt, P_pv, P_wind, P_el, Q_cool, Q_heat, global_config)

    # === 生成图表 ===
    plt.figure(figsize=(10, 8))
    plot_scheduling(x_opt, P_pv, P_wind, P_el, Q_cool, Q_heat, "优化结果", global_config)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return jsonify({
        'status': 'success',
        'annual_cost_10k_yuan': round(cost, 2),
        'annual_carbon_ton': round(carbon, 0),
        'ESI': round(ESI, 4),
        'plot': img_base64
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)