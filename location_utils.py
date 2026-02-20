# location_utils.py
import re

def parse_location_input(location_str):
    """
    解析用户输入的位置：支持 "上海" 或 "31.2,121.5"
    返回 (lat, lon, city_name)
    """
    # 尝试解析经纬度
    coord_pattern = r'^\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*$'
    match = re.match(coord_pattern, location_str)
    if match:
        lat = float(match.group(1))
        lon = float(match.group(2))
        return lat, lon, None
    
    # 否则视为城市名（简化处理）
    city_map = {
        '北京': (39.9, 116.4),
        '上海': (31.2, 121.5),
        '广州': (23.1, 113.3),
        '深圳': (22.5, 114.1),
        '成都': (30.6, 104.1),
        '西安': (34.3, 108.9),
        '乌鲁木齐': (43.8, 87.6),
        '哈尔滨': (45.8, 126.5)
    }
    lat, lon = city_map.get(location_str, (31.2, 121.5))  # 默认上海
    return lat, lon, location_str

def get_regional_config(lat, lon):
    """根据位置返回区域参数"""
    # 简化：按纬度划分区域
    if lat > 35:  # 北方
        price_buy = 0.55
        carbon_grid = 700e-6
    elif lat < 25:  # 南方
        price_buy = 0.65
        carbon_grid = 550e-6
    else:  # 中部
        price_buy = 0.60
        carbon_grid = 600e-6
    
    return {
        'price_buy': price_buy,
        'price_sell': price_buy * 1.1,
        'carbon_grid': carbon_grid,
        'timezone': 'Asia/Shanghai'
    }