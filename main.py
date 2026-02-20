# main.py
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from location_utils import parse_location_input, get_regional_config
from device_config import get_device_config
from config import build_config
from pv_model import pv_forecast_from_location, pv_forecast_default
from wind_model import wind_power_forecast, wind_forecast_default
from sensor_reader import ArduinoSensorReader, lux_to_irradiance, estimate_pv_power, estimate_wind_power
from load_profile import generate_load_profiles
from objectives import (
    economic_cost, carbon_emission, negative_ESI, weighted_objective
)
from emergy_analysis import calculate_ESI
from optimizer import optimize_single_objective
from plot_results import plot_scheduling
import numpy as np

def main():
    print("=" * 50)
    print("  数据中心综合能源调度优化系统 v3.0")
    print("  ✅ 光伏 + 风电 + 储热 + 实时传感器支持")
    print("=" * 50)
    
    # === 步骤1: 选择数据源 ===
    print("\n📡 数据源选择:")
    print("1. 仿真模式（基于地理位置）")
    print("2. 实时传感器模式（Arduino）")
    data_mode = input("请选择 (1/2): ").strip()
    
    P_pv = None
    P_wind = None
    global_config = None
    
    if data_mode == '2':
        # === 实时传感器模式 ===
        port = input("  Arduino 串口号 (Windows: COM3, Linux: /dev/ttyACM0): ").strip() or "COM3"
        try:
            reader = ArduinoSensorReader(port=port)
            print("  正在读取传感器数据...（等待5秒）")
            time.sleep(5)
            
            lux, wind_speed = reader.read_data()
            reader.close()
            
            if lux is None:
                raise Exception("未收到有效数据")
            
            print(f"  ✅ 传感器数据: 光照={lux:.0f} lux, 风速={wind_speed:.1f} m/s")
            
            # 获取设备配置（用于功率估算）
            device_config = get_device_config()
            regional_config = {'price_buy':0.6,'price_sell':0.7,'carbon_grid':600e-6,'timezone':'Asia/Shanghai'}
            global_config = build_config(device_config, regional_config)
            
            GHI = lux_to_irradiance(lux)
            P_pv_val = estimate_pv_power(GHI, global_config['PV_AREA'], global_config['PV_EFF'])
            P_wind_val = estimate_wind_power(wind_speed)
            
            P_pv = np.full(24, P_pv_val)
            P_wind = np.full(24, P_wind_val)
            print(f"  → 光伏出力 ≈ {P_pv_val:.1f} kW, 风电出力 ≈ {P_wind_val:.1f} kW")
            
        except Exception as e:
            print(f"❌ 传感器模式失败: {e}，回退到仿真模式")
            data_mode = '1'
    
    if data_mode == '1':
        # === 仿真模式 ===
        location_input = input("📍 请输入位置（城市或经纬度）: ")
        lat, lon, city_name = parse_location_input(location_input)
        location_str = city_name if city_name else f"({lat:.2f}, {lon:.2f})"
        print(f"✅ 位置: {location_str}")
        
        regional_config = get_regional_config(lat, lon)
        device_config = get_device_config()
        global_config = build_config(device_config, regional_config)
        
        try:
            P_pv = pv_forecast_from_location(
                lat, lon,
                pv_area=global_config['PV_AREA'],
                pv_eff=global_config['PV_EFF']
            )
            print("☀️ 光伏出力已生成")
        except:
            P_pv = pv_forecast_default(global_config)
            print("⚠️ 使用默认光伏曲线")
        
        try:
            from pvlib.iotools import get_pvgis_hourly
            wind_data, _ = get_pvgis_hourly(lat, lon, start='2023-07-15', end='2023-07-15')
            wind_speed = wind_data['wind_speed'].values[:24]
            P_wind = wind_power_forecast(wind_speed)
            print("🌬️ 风电出力已生成")
        except:
            P_wind = wind_forecast_default()
            print("⚠️ 使用默认风电曲线")
    
    # === 步骤2: 生成负荷 ===
    P_el, Q_cool, Q_heat = generate_load_profiles()
    
    # === 步骤3: 选择优化模式 ===
    print("\n⚙️  请选择优化模式:")
    print("1. 单目标优化")
    print("2. 多目标加权优化（自定义权重）")
    mode = input("请选择 (1/2): ").strip()

    T = global_config['T']
    n_vars = 9 * T

    if mode == '1':
        print("   a) 经济成本最小")
        print("   b) 碳排放最小")
        print("   c) ESI 最大")
        choice = input("子选项 (a/b/c): ").strip()
        if choice == 'a':
            obj_func = lambda x: economic_cost(x, P_pv, P_wind, P_el, Q_cool, Q_heat, global_config)
            title = "经济最优调度"
        elif choice == 'b':
            obj_func = lambda x: carbon_emission(x, P_pv, P_wind, P_el, Q_cool, Q_heat, global_config)
            title = "碳排最优调度"
        else:
            obj_func = lambda x: negative_ESI(x, P_pv, P_wind, P_el, Q_cool, Q_heat, global_config)
            title = "ESI最优调度"
    else:
        w1 = float(input("   经济成本权重 (w1): ") or "0.5")
        w2 = float(input("   碳排放权重 (w2): ") or "0.3")
        w3 = float(input("   -ESI 权重 (w3): ") or "0.2")
        obj_func = lambda x: weighted_objective(x, P_pv, P_wind, P_el, Q_cool, Q_heat, global_config, w1, w2, w3)
        title = f"加权优化 (w1={w1}, w2={w2}, w3={w3})"

    # === 步骤4: 优化 ===
    print("\n⏳ 开始优化...")
    x_opt = optimize_single_objective(obj_func, n_vars, bounds=(0, 500), n_gen=80)

    # === 步骤5: 输出结果 ===
    cost_val = economic_cost(x_opt, P_pv, P_wind, P_el, Q_cool, Q_heat, global_config)[0]
    carbon_val = carbon_emission(x_opt, P_pv, P_wind, P_el, Q_cool, Q_heat, global_config)[0]
    ESI_val, EYR, ELR = calculate_ESI(x_opt, P_pv, P_wind, P_el, Q_cool, Q_heat, global_config)

    print(f"\n📊 优化结果:")
    print(f"   年经济成本 ≈ {cost_val * 365 / 1e4:.2f} 万元")
    print(f"   年碳排放 ≈ {carbon_val * 365 / 1000:,.0f} 吨")
    print(f"   ESI = {ESI_val:.4f}")

    # === 步骤6: 可视化 ===
    plot_scheduling(x_opt, P_pv, P_wind, P_el, Q_cool, Q_heat, title, global_config)

if __name__ == "__main__":
    main()