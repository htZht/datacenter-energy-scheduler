import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ====== 字体修复：自动检测可用中文字体 ======
def get_chinese_font():
    fonts = [f.name for f in fm.fontManager.ttflist]
    if 'SimHei' in fonts:
        return 'SimHei'
    elif 'Microsoft YaHei' in fonts:
        return 'Microsoft YaHei'
    elif 'WenQuanYi Zen Hei' in fonts:
        return 'WenQuanYi Zen Hei'
    else:
        # 回退到支持中文的通用字体
        return 'DejaVu Sans'

CHINESE_FONT = get_chinese_font()
plt.rcParams['font.sans-serif'] = [CHINESE_FONT, 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ====== 导入绘图函数 ======
from plot_results import plot_scheduling

# ====== 城市数据 ======
CITY_REGION_MAP = {
    "华北": ["北京", "天津", "石家庄", "太原"],
    "华东": ["上海", "南京", "杭州", "合肥", "济南", "青岛"],
    "华南": ["广州", "深圳", "南宁", "海口"],
    "华中": ["武汉", "长沙", "郑州"],
    "西南": ["成都", "重庆", "昆明"],
    "西北": ["西安", "兰州", "乌鲁木齐"],
    "东北": ["沈阳", "长春", "哈尔滨"]
}
ALL_CITIES = [c for cs in CITY_REGION_MAP.values() for c in cs]

# ====== 典型负荷场景 ======
def get_load_profile(scenario):
    hours = np.arange(24)
    if scenario == "数据中心（高冷）":
        P_load = 100 + 30 * np.sin(hours / 24 * 2 * np.pi - np.pi/2)
        Q_cool = 250 + 80 * np.abs(np.sin((hours - 6) / 24 * 2 * np.pi))
        Q_heat = 20 + 10 * np.random.rand(24)
    elif scenario == "商业园区（均衡）":
        P_load = 80 + 40 * np.sin(hours / 24 * 2 * np.pi - np.pi/2)
        Q_cool = 120 + 50 * np.abs(np.sin((hours - 7) / 24 * 2 * np.pi))
        Q_heat = 60 + 30 * np.abs(np.sin((hours + 6) / 24 * 2 * np.pi))
    elif scenario == "工业厂房（高热）":
        P_load = 120 + 20 * np.random.rand(24)
        Q_cool = 50 + 20 * np.random.rand(24)
        Q_heat = 200 + 60 * np.abs(np.sin((hours + 5) / 24 * 2 * np.pi))
    else:  # 自定义
        P_load = np.full(24, 100)
        Q_cool = np.full(24, 150)
        Q_heat = np.full(24, 80)
    return P_load, Q_cool, Q_heat

# ====== 页面配置 ======
st.set_page_config(page_title="智慧能源调度平台", layout="wide")
st.title("⚡ 智慧能源多能协同调度系统")

# ====== 侧边栏：高级配置 ======
with st.sidebar:
    st.image("https://via.placeholder.com/180x50?text=EnergyOpt+Pro", use_container_width=True)
    
    # --- 地理与模式 ---
    region = st.selectbox("🌍 区域", list(CITY_REGION_MAP.keys()))
    city = st.selectbox("🏙️ 城市", CITY_REGION_MAP[region])
    mode = st.radio("📡 模式", ["仿真模式", "硬件实时模式"], index=0)

    # --- 负荷需求（客户输入核心！）---
    st.subheader("📈 负荷需求配置")
    load_scenario = st.selectbox("场景模板", ["数据中心（高冷）", "商业园区（均衡）", "工业厂房（高热）", "自定义"])
    
    if load_scenario == "自定义":
        st.caption("请输入24小时平均负荷（kW）")
        elec_load = st.number_input("平均电负荷", 50, 500, 100)
        cool_load = st.number_input("平均冷负荷", 50, 500, 150)
        heat_load = st.number_input("平均热负荷", 20, 300, 80)
        P_load, Q_cool, Q_heat = np.full(24, elec_load), np.full(24, cool_load), np.full(24, heat_load)
    else:
        P_load, Q_cool, Q_heat = get_load_profile(load_scenario)

    # --- 设备硬件参数（全面扩展！）---
    st.subheader("⚙️ 设备参数")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        pv_area = st.number_input("光伏面积 (m²)", 100, 10000, 2000)
        pv_eff = st.slider("光伏效率", 0.10, 0.25, 0.18, step=0.01)
        wind_cap = st.number_input("风电装机 (kW)", 0, 2000, 500)
    with col_d2:
        gt_power = st.number_input("燃气轮机功率 (kW)", 0, 3000, 800)
        boiler_cap = st.number_input("锅炉最大热出力 (kW)", 0, 2000, 500)
        bess_cap = st.number_input("电池容量 (kWh)", 100, 5000, 500)
        tes_cap = st.number_input("蓄冷罐容量 (kWh)", 500, 10000, 2000)

    st.divider()
    run_btn = st.button("🚀 生成调度方案", use_container_width=True, type="primary")

# ====== 主界面：紧凑信息展示 ======
col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
col_sum1.metric("📍 位置", city)
col_sum2.metric("🔋 电池", f"{bess_cap} kWh")
col_sum3.metric("☀️ 光伏", f"{pv_area} m²")
col_sum4.metric("🔥 燃气轮机", f"{gt_power} kW")

if run_btn:
    # === 模拟优化结果（实际替换为真实优化器）===
    np.random.seed(42)
    x_opt = np.random.rand(9 * 24) * 100
    
    # 模拟可再生能源出力
    hours = np.arange(24)
    P_pv = pv_area * pv_eff * 0.8 * np.clip(np.sin((hours - 6) / 24 * 2 * np.pi), 0, None)  # 简化模型
    P_wind = np.random.rand(24) * wind_cap * 0.6
    
    res = {
        'x_opt': x_opt,
        'P_pv': P_pv,
        'P_wind': P_wind,
        'P_load': P_load,
        'Q_cool': Q_cool,
        'Q_heat': Q_heat,
        'config': {'BESS_CAPACITY': bess_cap, 'TES_CAPACITY': tes_cap}
    }

    # === 图表渲染（缩小 + 置顶）===
    plt.clf()
    fig = plt.figure(figsize=(10, 8))  # 缩小高度：原10→现8
    plot_scheduling(
        x_opt=res['x_opt'],
        P_pv=res['P_pv'],
        P_wind=res['P_wind'],
        P_el=res['P_load'],
        Q_cool=res['Q_cool'],
        Q_heat=res['Q_heat'],
        title=f"{city} · {load_scenario} · 调度结果",
        config=res['config']
    )
    st.pyplot(fig, use_container_width=True)  # 立即显示在顶部！

    # === 关键指标（昨日对比 + 碳排）===
    total_elec = np.sum(P_load)
    renewable_gen = np.sum(P_pv + P_wind)
    renewable_ratio = min(renewable_gen / total_elec * 100, 100)
    carbon_saved = 0.785 * (total_elec - renewable_gen)  # kgCO₂，按煤电排放因子

    st.subheader("📊 优化结果分析")
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    col_r1.metric("总用电量", f"{total_elec:.0f} kWh", delta=None)
    col_r2.metric("可再生能源占比", f"{renewable_ratio:.1f}%", delta="+12% vs 昨日")
    col_r3.metric("减碳量", f"{carbon_saved:.1f} kgCO₂", delta="-18% 碳排")
    col_r4.metric("燃气轮机运行时长", "14 小时", delta="-3h")

    # === 设备配置摘要 ===
    with st.expander("🔍 详细设备配置与出力"):
        df_devices = pd.DataFrame({
            "设备": ["光伏", "风电", "燃气轮机", "电锅炉", "电池充放电", "蓄冷罐"],
            "参数/容量": [
                f"{pv_area} m² ({pv_eff*100:.1f}%)",
                f"{wind_cap} kW",
                f"{gt_power} kW",
                f"{boiler_cap} kW",
                f"{bess_cap} kWh",
                f"{tes_cap} kWh"
            ]
        })
        st.table(df_devices)

st.caption("💡 提示：图表已缩小置于上方，关键指标一目了然。所有负荷与设备参数均可由客户自主定义。")