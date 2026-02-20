# energy_system.py
import numpy as np

def check_energy_balance(x, P_pv, P_wind, P_el, Q_cool, Q_heat, config):
    T = config['T']
    x = np.array(x)
    X = x.reshape(9, T)
    
    P_buy = X[0]; P_sell = X[1]; P_boiler = X[2]
    P_chill_e = X[3]; P_chill_a = X[4]
    P_batt_ch = X[5]; P_batt_dis = X[6]
    P_tes_ch = X[7]; P_tes_dis = X[8]
    
    soc = np.zeros(T+1); soc[0] = 0.5
    tes = np.zeros(T+1); tes[0] = 0.5
    
    elec_bal = np.zeros(T)
    heat_bal = np.zeros(T)
    cool_bal = np.zeros(T)
    
    for t in range(T):
        soc[t+1] = soc[t] + (P_batt_ch[t]*config['BESS_EFF'] - P_batt_dis[t]/config['BESS_EFF']) / config['BESS_CAPACITY']
        tes[t+1] = tes[t] + (P_tes_ch[t]*config['TES_EFF'] - P_tes_dis[t]/config['TES_EFF']) / config['TES_CAPACITY']
        
        # ✅ 关键更新：加入 P_wind[t]
        elec_bal[t] = P_pv[t] + P_wind[t] + P_buy[t] - P_sell[t] + P_batt_dis[t] - P_el[t] - P_chill_e[t] - P_batt_ch[t]
        heat_bal[t] = P_boiler[t] - Q_heat[t] - P_chill_a[t] / 1.1
        cool_bal[t] = P_chill_e[t]*3.5 + P_chill_a[t] + P_tes_dis[t] - Q_cool[t] - P_tes_ch[t]
    
    soc_pen = np.sum(np.maximum(config['BESS_SOC_MIN'] - soc[1:], 0) + np.maximum(soc[1:] - config['BESS_SOC_MAX'], 0))
    tes_pen = np.sum(np.maximum(config['TES_TEMP_MIN'] - tes[1:], 0) + np.maximum(tes[1:] - config['TES_TEMP_MAX'], 0))
    
    penalty = (np.sum(elec_bal**2) + np.sum(heat_bal**2) + np.sum(cool_bal**2) +
               soc_pen * 1e3 + tes_pen * 1e3)
    return penalty

def calculate_safety_penalty(x, config):
    T = config['T']
    x = np.array(x)
    X = x.reshape(9, T)
    
    pen = 0.0
    pen += np.sum(np.maximum(X[0] - config['CAP_GRID_BUY'], 0)) * 1e4
    pen += np.sum(np.maximum(X[1] - config['CAP_GRID_SELL'], 0)) * 1e4
    pen += np.sum(np.maximum(X[2] - config['CAP_BOILER'], 0)) * 1e4
    pen += np.sum(np.maximum(X[5] - config['BESS_MAX_POWER'], 0)) * 1e4
    pen += np.sum(np.maximum(X[6] - config['BESS_MAX_POWER'], 0)) * 1e4
    return pen