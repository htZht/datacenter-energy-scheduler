# load_profile.py —— 替换为以下内容
import numpy as np
import pandas as pd
import os

def generate_load_profiles():
    """
    Web 模式下：优先读 CSV，否则用默认曲线（不交互！）
    """
    csv_file = "datacenter_load.csv"
    
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            P_el = df['P_el'].values[:24]
            Q_cool = df['Q_cool'].values[:24]
            Q_heat = df['Q_heat'].values[:24]
            return P_el, Q_cool, Q_heat
        except:
            pass  # 失败则用默认
    
    # 默认曲线（无交互）
    hours = np.arange(24)
    P_el = 200 + 20 * np.sin((hours - 6) * np.pi / 12)
    Q_cool = np.maximum(150 + 50 * np.sin((hours - 14) * np.pi / 12), 50)
    Q_heat = np.full(24, 20.0)
    return P_el, Q_cool, Q_heat