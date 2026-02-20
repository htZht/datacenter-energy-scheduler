# config.py
def build_config(device_config, regional_config):
    return {
        'T': 24,
        'timezone': regional_config['timezone'],
        'CAP_GRID_BUY': 300,
        'CAP_GRID_SELL': 100,
        'CAP_BOILER': device_config['boiler_max'],
        'CAP_CHILL_ELEC': device_config['chiller_elec_max'],
        'CAP_CHILL_ABS': device_config['chiller_abs_max'],
        'BESS_CAPACITY': device_config['bess_capacity'],
        'BESS_MAX_POWER': device_config['bess_max_power'],
        'BESS_EFF': 0.9,
        'BESS_SOC_MIN': 0.2,
        'BESS_SOC_MAX': 0.9,
        'TES_CAPACITY': device_config['tes_capacity'],
        'TES_MAX_POWER': min(device_config['tes_capacity'] / 8, 150),
        'TES_EFF': 0.95,
        'TES_TEMP_MIN': 0.2,
        'TES_TEMP_MAX': 0.9,
        'PRICE_GRID_BUY': regional_config['price_buy'],
        'PRICE_GRID_SELL': regional_config['price_sell'],
        'PRICE_GAS': 4.84 / 10.8,
        'CARBON_GRID': regional_config['carbon_grid'],
        'CARBON_GAS': 180.0e-6,
        'EMR_SOLAR': 5.89e10,
        'EMR_WIND': 1.0e11,     # 风能能值比（典型值）
        'EMR_GRID': 1.74e8,
        'EMR_GAS': 7.73e7,
        'EMR_ASSET': 4.325e15,
        'PV_AREA': device_config['pv_area'],
        'PV_EFF': device_config['pv_efficiency']
    }