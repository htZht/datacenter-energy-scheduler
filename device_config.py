# device_config.py —— 替换为以下内容
def get_device_config():
    """
    返回固定设备配置（Web 模式下不交互）
    实际项目中可从数据库/配置文件读取
    """
    return {
        'pv_area': 400,
        'pv_efficiency': 0.175,
        'boiler_max': 200,
        'chiller_elec_max': 150,
        'chiller_abs_max': 100,
        'bess_capacity': 500,
        'bess_max_power': 100,
        'tes_capacity': 2000
    }