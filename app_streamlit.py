import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 导入你的绘图函数
from plot_results import plot_scheduling

# ========== 城市与区域数据（真实中国主要城市） ==========
CITY_REGION_MAP = {
    "华北": ["北京", "天津", "石家庄", "太原", "呼和浩特"],
    "华东": ["上海", "南京", "杭州", "合肥", "济南", "福州", "南昌", "青岛", "宁波", "厦门"],
    "华南": ["广州", "深圳", "南宁", "海口", "东莞", "佛山", "珠海"],
    "华中": ["武汉", "长沙", "郑州", "南昌"],
    "西南": ["成都", "重庆", "昆明", "贵阳", "拉萨"],
    "西北": ["西安", "兰州", "西宁", "银川", "乌鲁木齐"],
    "东北": ["沈阳", "长春", "哈尔滨", "大连"]
}

ALL_CITIES = [city for cities in CITY_REGION_MAP.values() for city in cities]

# ========== 模拟数据生成（实际替换为 optimizer.py 调用） ==========
def mock_optimization_result():
    np.random.seed(42)
    x_opt = np.random.rand(9 * 24) * 100
    return {
        'x_opt': x_opt,
        'P_pv': np.clip(np.sin(np.linspace(-np.pi/2, np.pi/2, 24)) * 100 + 50, 0, None),
        'P_wind': np.random.rand(24) * 60,
        'P_load': np.random.rand(24) * 120 + 80,
        'Q_cool': np.random.rand(24) * 200 + 100,
        'Q_heat': np.random.rand(24) * 80 + 30,
        'config': {'BESS_CAPACITY': 500, 'TES_CAPACITY': 2000}
    }

# ========== 页面配置 ==========
st.set_page_config(
    page_title="智慧能源调度平台",
    page_icon="⚡",
    layout="wide"
)

# ========== 字体修复（防止中文乱码）==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== 自定义 CSS ==========
st.markdown("""
<style>
    .main { background-color: #f9fafb; }
    h1, h2, h3 { color: #1e3a8a; }
    .stMetric { background: white; border-radius: 8px; padding: 1rem; box-shadow: 0 2px 6px rgba(0,0,0,0.05); }
    .block-container { padding: 2rem 3rem; }
    .css-1v0mbdj img { margin-bottom: -20px; }
    .stButton>button {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ========== 侧边栏：高级配置 ==========
with st.sidebar:
    st.image("https://via.placeholder.com/180x60?text=EnergyOpt+Pro", use_container_width=True)
    st.title("🛠️ 系统配置中心")

    # === 区域与城市选择 ===
    selected_region = st.selectbox("🌍 选择区域", list(CITY_REGION_MAP.keys()))
    selected_city = st.selectbox("🏙️ 选择城市", CITY_REGION_MAP[selected_region])

    # === 运行模式 ===
    mode = st.radio("📡 运行模式", ["仿真模式", "硬件实时模式"], index=0)
    if mode == "硬件实时模式":
        st.warning("需连接传感器与PLC设备")

    # === 设备硬件参数（客户可调！）===
    st.subheader("⚙️ 设备参数配置")
    bess_cap = st.number_input("电池容量 (kWh)", min_value=100, max_value=5000, value=500, step=50)
    tes_cap = st.number_input("蓄冷罐容量 (kWh)", min_value=500, max_value=10000, value=2000, step=100)
    boiler_eff = st.slider("锅炉热效率", 0.7, 0.98, 0.9, step=0.01)

    # === 优化权重 ===
    st.subheader("⚖️ 优化目标权重")
    col_w1, col_w2, col_w3 = st.columns(3)
    w_cost = col_w1.slider("经济性", 0.0, 1.0, 0.5, step=0.1)
    w_carbon = col_w2.slider("低碳性", 0.0, 1.0, 0.3, step=0.1)
    w_reliability = col_w3.slider("可靠性", 0.0, 1.0, 0.2, step=0.1)
    
    # 归一化
    total = w_cost + w_carbon + w_reliability
    if total > 0:
        w_cost /= total
        w_carbon /= total
        w_reliability /= total

    st.caption(f"归一化后权重：💰{w_cost:.2f} 🌱{w_carbon:.2f} 🔒{w_reliability:.2f}")

    st.divider()
    run_btn = st.button("🚀 开始优化", use_container_width=True, type="primary")

# ========== 主界面 ==========
st.title("⚡ 智慧能源调度平台 — 多能协同优化系统")

# 显示当前配置摘要
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("📍 位置", selected_city)
col_b.metric("📡 模式", mode)
col_c.metric("🔋 电池", f"{bess_cap} kWh")
col_d.metric("🧊 蓄冷", f"{tes_cap} kWh")

if run_btn:
    with st.spinner(f"正在为【{selected_city}】计算24小时最优调度策略..."):
        res = mock_optimization_result()
        # 实际应调用：res = run_optimizer(city=selected_city, config={...})

    st.success(f"✅ {selected_city} 调度方案生成成功！")

    # 渲染图表（确保中文不乱码）
    plt.clf()
    try:
        plot_scheduling(
            x_opt=res['x_opt'],
            P_pv=res['P_pv'],
            P_wind=res['P_wind'],
            P_el=res['P_load'],
            Q_cool=res['Q_cool'],
            Q_heat=res['Q_heat'],
            title=f"{selected_city} · 24小时能源调度结果（{mode}）",
            config={'BESS_CAPACITY': bess_cap, 'TES_CAPACITY': tes_cap}
        )
        st.pyplot(plt.gcf(), use_container_width=True)
    except Exception as e:
        st.error(f"绘图失败：{str(e)}")

    # 显示关键指标
    st.subheader("📊 优化结果摘要")
    total_elec = np.sum(res['P_load']) * 1  # kWh
    renewable_ratio = np.sum(res['P_pv'] + res['P_wind']) / total_elec * 100
    carbon_saved = 0.8 * total_elec * (1 - renewable_ratio/100)  # 简化估算

    col1, col2, col3 = st.columns(3)
    col1.metric("总用电量", f"{total_elec:.0f} kWh")
    col2.metric("可再生能源占比", f"{renewable_ratio:.1f}%")
    col3.metric("减碳量", f"{carbon_saved:.1f} kgCO₂")

# ========== 底部说明 ==========
st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("💡 提示：在「仿真模式」下可快速测试不同城市与配置；「硬件实时模式」需部署边缘网关与传感器。所有参数均可由客户自主调整。")