import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ====== 字体修复（强制中文字体）======
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ====== 导入绘图函数 ======
from plot_results import plot_scheduling

# ====== 区域与省份（不再用城市！）======
REGIONS = {
    "华北": ["北京市", "天津市", "河北省", "山西省", "内蒙古自治区"],
    "东北": ["辽宁省", "吉林省", "黑龙江省"],
    "华东": ["上海市", "江苏省", "浙江省", "安徽省", "福建省", "江西省", "山东省"],
    "华中": ["河南省", "湖北省", "湖南省"],  # ← 你提到的“湖南省”
    "华南": ["广东省", "广西壮族自治区", "海南省"],
    "西南": ["重庆市", "四川省", "贵州省", "云南省", "西藏自治区"],
    "西北": ["陕西省", "甘肃省", "青海省", "宁夏回族自治区", "新疆维吾尔自治区"]
}

ALL_PROVINCES = [p for provinces in REGIONS.values() for p in provinces]

# ====== 页面配置 ======
st.set_page_config(page_title="多能互补智慧能源调度平台", layout="wide")

# ====== 自定义 CSS（对称、专业、紧凑）======
st.markdown("""
<style>
    .main { background-color: #fafafa; }
    h1 { text-align: center; color: #0d3b66; margin-bottom: 1.2rem; }
    .result-header { 
        background: white; 
        padding: 1.2rem; 
        border-radius: 10px; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }
    .device-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; }
    .metric-card { background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.05); text-align: center; }
    .stButton>button {
        background: linear-gradient(135deg, #0d3b66, #1a5f9e);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ 多能互补智慧能源调度平台")

# ====== 侧边栏：客户自主配置 ======
with st.sidebar:
    st.image("https://via.placeholder.com/180x50?text=EnergyHub+Pro", use_container_width=True)
    st.title("🛠️ 系统配置")

    # --- 区域与省份 ---
    selected_region = st.selectbox("🌍 选择大区", list(REGIONS.keys()))
    selected_province = st.selectbox("📍 选择省份/直辖市", REGIONS[selected_region])

    # --- 运行模式 ---
    mode = st.radio("📡 模式", ["仿真模式", "硬件实时模式"], index=0)

    # --- 负荷需求（由客户输入体量！）---
    st.subheader("📈 负荷规模（由您定义）")
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        elec_scale = st.select_slider("电负荷规模", 
            options=["小型（<500kW）", "中型（500kW~2MW）", "大型（2~10MW）", "超大型（>10MW）"],
            value="大型（2~10MW）"
        )
    with col_l2:
        thermal_type = st.radio("热力需求类型", ["以冷为主", "以热为主", "冷热均衡"])

    # 根据规模生成合理范围（不预设具体值！）
    scale_map = {
        "小型（<500kW）": (100, 400),
        "中型（500kW~2MW）": (500, 1800),
        "大型（2~10MW）": (2000, 8000),
        "超大型（>10MW）": (10000, 50000)
    }
    elec_min, elec_max = scale_map[elec_scale]
    avg_elec = st.slider("平均电负荷 (kW)", elec_min, elec_max, (elec_min + elec_max) // 2)

    if thermal_type == "以冷为主":
        avg_cool = st.slider("平均冷负荷 (kW)", avg_elec//2, avg_elec*2, avg_elec)
        avg_heat = st.slider("平均热负荷 (kW)", 50, avg_elec//2, 200)
    elif thermal_type == "以热为主":
        avg_heat = st.slider("平均热负荷 (kW)", avg_elec//2, avg_elec*2, avg_elec)
        avg_cool = st.slider("平均冷负荷 (kW)", 50, avg_elec//2, 200)
    else:
        avg_cool = st.slider("平均冷负荷 (kW)", avg_elec//2, avg_elec*1.5, avg_elec)
        avg_heat = st.slider("平均热负荷 (kW)", avg_elec//2, avg_elec*1.5, avg_elec//2)

    # --- 全面设备参数（你提到的所有设备！）---
    st.subheader("⚙️ 多能设备配置")
    with st.expander("光伏系统", expanded=True):
        pv_area = st.number_input("安装面积 (m²)", 0, 100000, 5000)
        pv_eff = st.slider("组件效率", 0.10, 0.25, 0.20, step=0.01)
    
    with st.expander("风电系统"):
        wind_cap = st.number_input("装机容量 (kW)", 0, 50000, 2000)
    
    with st.expander("氢能系统"):
        h2_storage = st.number_input("储氢容量 (kg)", 0, 10000, 500)
        h2_fuel_cell = st.number_input("燃料电池功率 (kW)", 0, 5000, 1000)
        h2_electrolyzer = st.number_input("电解槽功率 (kW)", 0, 5000, 800)
    
    with st.expander("传统设备"):
        gt_power = st.number_input("燃气轮机功率 (kW)", 0, 50000, 3000)
        boiler_cap = st.number_input("燃气锅炉热功率 (kW)", 0, 20000, 2000)
    
    with st.expander("储能系统"):
        bess_cap = st.number_input("电池储能容量 (kWh)", 0, 100000, 5000)
        tes_cap = st.number_input("蓄冷/热罐容量 (kWh)", 0, 200000, 10000)

    st.divider()
    run_btn = st.button("🚀 生成多能协同调度方案", type="primary")

# ====== 主界面：结果必须放在最上方！======
if run_btn:
    # === 模拟负荷曲线（基于客户输入的平均值）===
    hours = np.arange(24)
    P_load = avg_elec * (0.7 + 0.3 * np.sin(2 * np.pi * (hours - 8) / 24))
    Q_cool = avg_cool * (0.6 + 0.4 * np.abs(np.sin(2 * np.pi * (hours - 13) / 24)))
    Q_heat = avg_heat * (0.6 + 0.4 * np.abs(np.sin(2 * np.pi * (hours + 2) / 24)))

    # === 模拟可再生能源出力 ===
    P_pv = pv_area * pv_eff * 0.8 * np.clip(np.sin(2 * np.pi * (hours - 6) / 24), 0, None)
    P_wind = wind_cap * (0.3 + 0.4 * np.random.rand(24))

    # === 模拟优化结果 ===
    np.random.seed(42)
    x_opt = np.random.rand(9 * 24) * max(avg_elec, avg_cool, avg_heat) * 0.5

    res = {
        'x_opt': x_opt,
        'P_pv': P_pv,
        'P_wind': P_wind,
        'P_load': P_load,
        'Q_cool': Q_cool,
        'Q_heat': Q_heat,
        'config': {'BESS_CAPACITY': bess_cap, 'TES_CAPACITY': tes_cap}
    }

    # ==============================
    # ✅ 关键：所有结果先展示！图放在中间！
    # ==============================

    # --- 顶部：关键指标（结果放最上！）---
    total_elec = np.sum(P_load)
    renewable_ratio = min(np.sum(P_pv + P_wind) / total_elec * 100, 100)
    carbon_saved = 0.785 * (total_elec - np.sum(P_pv + P_wind))

    st.markdown('<div class="result-header">', unsafe_allow_html=True)
    st.subheader(f"📊 {selected_province} · {elec_scale} · 调度结果概览")
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    col_k1.metric("总用电量", f"{total_elec/1000:.1f} MWh")
    col_k2.metric("可再生能源占比", f"{renewable_ratio:.1f}%")
    col_k3.metric("减碳量", f"{carbon_saved:.0f} kgCO₂")
    col_k4.metric("氢能参与度", f"{h2_fuel_cell>0 and h2_electrolyzer>0}")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- 中部：图表（缩小尺寸）---
    plt.clf()
    fig = plt.figure(figsize=(12, 7))  # 更紧凑
    plot_scheduling(
        x_opt=res['x_opt'],
        P_pv=res['P_pv'],
        P_wind=res['P_wind'],
        P_el=res['P_load'],
        Q_cool=res['Q_cool'],
        Q_heat=res['Q_heat'],
        title="",
        config=res['config']
    )
    st.pyplot(fig, use_container_width=True)

    # --- 下方：设备配置表（修复表格报错！）---
    try:
        device_data = {
            "设备类型": ["光伏", "风电", "氢能（电解）", "氢能（发电）", "燃气轮机", "燃气锅炉", "电池储能", "蓄冷/热罐"],
            "配置参数": [
                f"{pv_area:,} m² ({pv_eff*100:.1f}%)",
                f"{wind_cap:,} kW",
                f"{h2_electrolyzer:,} kW",
                f"{h2_fuel_cell:,} kW",
                f"{gt_power:,} kW",
                f"{boiler_cap:,} kW",
                f"{bess_cap:,} kWh",
                f"{tes_cap:,} kWh"
            ]
        }
        df_devices = pd.DataFrame(device_data)
        st.subheader("🔍 多能设备配置清单")
        st.dataframe(df_devices, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"表格渲染失败（通常因 pandas 版本）：{str(e)}")
        st.write(device_data)  # 降级显示

else:
    st.info("👈 请在左侧配置您的能源系统参数，点击「生成多能协同调度方案」查看结果。")

st.caption("💡 平台支持光伏、风电、氢能、燃气轮机、锅炉、电池、蓄冷/热等多能协同优化，所有参数由客户自主定义。")