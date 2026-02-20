# objectives.py
from energy_system import check_energy_balance, calculate_safety_penalty
import numpy as np

def _base_cost(x, P_pv, P_wind, P_el, Q_cool, Q_heat, config):
    penalty = check_energy_balance(x, P_pv, P_wind, P_el, Q_cool, Q_heat, config)
    safety_pen = calculate_safety_penalty(x, config)
    T = config['T']
    X = np.array(x).reshape(9, T)
    cost = np.sum(config['PRICE_GRID_BUY']*X[0] - config['PRICE_GRID_SELL']*X[1] + config['PRICE_GAS']*X[2])
    return cost + penalty * 1e3 + safety_pen

def economic_cost(x, P_pv, P_wind, P_el, Q_cool, Q_heat, config):
    return (float(_base_cost(x, P_pv, P_wind, P_el, Q_cool, Q_heat, config)),)

def carbon_emission(x, P_pv, P_wind, P_el, Q_cool, Q_heat, config):
    penalty = check_energy_balance(x, P_pv, P_wind, P_el, Q_cool, Q_heat, config)
    safety_pen = calculate_safety_penalty(x, config)
    T = config['T']
    X = np.array(x).reshape(9, T)
    carbon = np.sum(config['CARBON_GRID']*X[0] + config['CARBON_GAS']*X[2])
    return (float(carbon + penalty * 1e3 + safety_pen),)

def negative_ESI(x, P_pv, P_wind, P_el, Q_cool, Q_heat, config):
    from emergy_analysis import calculate_ESI
    penalty = check_energy_balance(x, P_pv, P_wind, P_el, Q_cool, Q_heat, config)
    safety_pen = calculate_safety_penalty(x, config)
    ESI, _, _ = calculate_ESI(x, P_pv, P_wind, P_el, Q_cool, Q_heat, config)
    return (float(-ESI + penalty * 1e3 + safety_pen),)

def weighted_objective(x, P_pv, P_wind, P_el, Q_cool, Q_heat, config, w1=1.0, w2=0.0, w3=0.0):
    cost = _base_cost(x, P_pv, P_wind, P_el, Q_cool, Q_heat, config)
    T = config['T']
    X = np.array(x).reshape(9, T)
    carbon = np.sum(config['CARBON_GRID']*X[0] + config['CARBON_GAS']*X[2])
    from emergy_analysis import calculate_ESI
    ESI, _, _ = calculate_ESI(x, P_pv, P_wind, P_el, Q_cool, Q_heat, config)
    obj = w1 * cost + w2 * carbon + w3 * (-ESI)
    return (float(obj),)