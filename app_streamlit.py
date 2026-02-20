# app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ====== 安全导入本地模块（避免 ModuleNotFoundError）======
try:
    from china_electricity_price import get_hourly_price, get_all_provinces
    from load_profile import generate_load_profile
    from optimizer import optimize_energy_schedule
    from exergy_model import calculate_exergy_loss
    from emergy_model import calculate_emergy_indicators
    from load_flexibility import get_flexible_windows
except ImportError as e:
    st.error(f"❌ 模块导入失败: {e}")
    st.stop()

def main():
    st.set_page_config(
        page_title="🌱 数据中心能源-算力协同调度系统",
        layout="wide"
    )

    # ========== 初始化会话状态 ==========
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0
        st.session_state.results = None

    # ========== 侧边栏 ==========
    with st.sidebar:
        st.title("⚙️ 配置")
        
        # 获取省份列表（含“自定义”）
        provinces = get_all_provinces()
        province = st.selectbox("📍 选择地区", provinces, index=0)

        custom_price_profile = None
        if province == "自定义":
            st.subheader("✏️ 自定义分时电价")
            peak_p = st.number_input("峰时电价 (¥/kWh)", 0.5, 2.0, 1.2)
            flat_p = st.number_input("平时电价 (¥/kWh)", 0.3, 1.5, 0.8)
            valley_p = st.number_input("谷时电价 (¥/kWh)", 0.1, 1.0, 0.3)
            
            st.write("🕒 设置时段（24小时制）")
            col1, col2 = st.columns(2)
            with col1:
                peak_h = st.slider("峰时段", 0, 23, (10, 15))
            with col2:
                valley_h = st.slider("谷时段", 0, 23, (0, 8))
            
            # 构建24小时价格曲线
            custom_price_profile = []
            for h in range(24):
                if peak_h[0] <= h < peak_h[1]:
                    custom_price_profile.append(peak_p)
                elif valley_h[0] <= h < valley_h[1]:
                    custom_price_profile.append(valley_p)
                else:
                    custom_price_profile.append(flat_p)

        server_count = st.slider("🖥️ 服务器数量", 100, 500, 200)
        use_gt = st.toggle("🔥 启用燃气轮机", True)
        use_h2 = st.toggle("🟢 启用氢能", True)

        if st.button("🚀 开始优化", type="primary"):
            try:
                # 获取电价
                if province == "自定义":
                    price_profile = custom_price_profile
                else:
                    price_profile = get_hourly_price(province, "大工业", 24)
                
                # 生成负荷（按服务器数量缩放）
                base_load = generate_load_profile(24)
                load_profile = [l * (server_count / 200) for l in base_load]
                
                # 模拟风光出力
                pv_power = [max(0, 100 * np.sin(np.pi * i / 24)) for i in range(24)]
                wind_power = [70 + 30 * np.sin(2 * np.pi * i / 12 + 0.5) for i in range(24)]
                
                # 优化调度
                result = optimize_energy_schedule(
                    load_profile=load_profile,
                    pv_power=pv_power,
                    wind_power=wind_power,
                    price_profile=price_profile,
                    include_gas_turbine=use_gt,
                    include_hydrogen=use_h2
                )
                
                st.session_state.results = result
                st.session_state.current_step = 0
                st.success("✅ 优化完成！")
            except Exception as e:
                st.error(f"优化失败: {str(e)}")
                st.stop()

    # ========== 主界面 ==========
    st.title("🌱 数据中心智能能源-算力协同调度系统")

    if st.session_state.results is None:
        st.info("👈 请在左侧配置并点击「开始优化」")
        return

    results = st.session_state.results
    step = st.session_state.current_step

    # ========== 实时状态 ==========
    st.subheader(f"🕒 当前时刻: {step}:00")
    cols = st.columns(4)
    cols[0].metric("电网购电", f"{results['grid'][step]:.1f} kW")
    cols[1].metric("电池功率", f"{'放电' if results['battery'][step] > 0 else '充电'} {abs(results['battery'][step]):.1f} kW")
    cols[2].metric("燃气轮机", f"{results['gas_turbine'][step]:.1f} kW")
    cols[3].metric("氢能发电", f"{results['h2_fuelcell'][step]:.1f} kW")

    # ========== 关键：计算可持续性指标（确保变量定义）==========
    try:
        ex_loss = calculate_exergy_loss(
            grid_import=results["grid"],
            gt_power=results["gas_turbine"],
            h2fc_power=results["h2_fuelcell"],
            pv_power=results["pv"],
            wind_power=results["wind"],
            load=results["load"]
        )
    except Exception:
        ex_loss = 0.0

    try:
        emergy = calculate_emergy_indicators(
            pv_energy=sum(results["pv"]),
            wind_energy=sum(results["wind"]),
            grid_energy=sum(results["grid"]),
            ng_energy=sum(results["gas_turbine"]) * 0.3,
            h2_energy=sum(results["h2_fuelcell"]) * 0.4 / 33.3
        )
    except Exception:
        emergy = {"EYR": 0, "ELR": 0, "ESI": 0}

    # ========== 可持续性看板 ==========
    st.subheader("🌍 多维可持续性绩效")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("㶲损失率", f"{ex_loss:.1%}")
    c2.metric("能值产出率 (EYR)", f"{emergy['EYR']:.2f}")
    c3.metric("环境负载率 (ELR)", f"{emergy['ELR']:.2f}")
    c4.metric("可持续指数 (ESI)", f"{emergy['ESI']:.2f}")

    # ========== 调度建议 ==========
    advice = []
    flex_windows = get_flexible_windows(results["load"])
    if flex_windows:
        future_flex = [t for t in flex_windows if t > step]
        if future_flex:
            next_low = min(future_flex)
            advice.append(f"⏱️ 建议迁移算力至 {next_low}:00（低谷期）")
    
    if results["price"][step] > 1.0:
        advice.append("⚠️ 当前电价高，优先用储能")
    
    if advice:
        st.info("；".join(advice))

    # ========== 曲线图 ==========
    df = pd.DataFrame({
        "时间": [f"{i:02d}:00" for i in range(24)],
        "光伏": results["pv"],
        "风电": results["wind"],
        "负荷": results["load"],
        "电网": results["grid"],
        "燃气轮机": results["gas_turbine"],
        "氢能": results["h2_fuelcell"]
    }).set_index("时间")
    st.line_chart(df, height=400)

    # ========== 推进按钮 ==========
    if st.button("⏭️ 下一小时"):
        st.session_state.current_step = min(step + 1, 23)
        st.rerun()


# ========== 入口点 ==========
if __name__ == "__main__":
    main()