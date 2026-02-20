import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 导入你的绘图函数
from plot_results import plot_scheduling

# 模拟优化结果（实际项目中由 optimizer.py 生成）
def mock_optimization_result():
    """
    模拟一个优化结果字典，用于演示。
    实际使用时替换为真实优化器返回的结果。
    """
    np.random.seed(42)
    x_opt = np.random.rand(9 * 24) * 100  # 9设备 × 24小时
    
    return {
        'x_opt': x_opt,
        'P_pv': np.random.rand(24) * 80,      # 光伏出力
        'P_wind': np.random.rand(24) * 60,    # 风电出力
        'P_load': np.random.rand(24) * 120 + 50,  # 电负荷（对应函数中的 P_el）
        'Q_cool': np.random.rand(24) * 200 + 100, # 冷负荷
        'Q_heat': np.random.rand(24) * 80 + 30,   # 热负荷
        'config': {
            'BESS_CAPACITY': 500,
            'TES_CAPACITY': 2000
        }
    }

# ================== Streamlit 应用开始 ==================
st.set_page_config(page_title="数据中心能源调度系统", layout="wide")
st.title("🔋 数据中心多能协同调度优化系统")

st.markdown("""
本系统基于光伏、风电、电负荷、冷热负荷等数据，  
通过优化算法求解最优设备调度策略，并可视化结果。
""")

# 按钮触发优化
if st.button("🚀 运行能源调度优化"):
    with st.spinner("正在计算最优调度方案..."):
        # 这里替换成你的真实优化调用，例如：
        # from optimizer import run_optimization
        # res = run_optimization()
        
        # 目前使用模拟数据
        res = mock_optimization_result()

    st.success("✅ 优化完成！")

    # ====== 关键：正确调用 plot_scheduling ======
    # 清除之前的图形（防止内存泄漏和图表叠加）
    plt.clf()

    try:
        plot_scheduling(
            x_opt=res['x_opt'],
            P_pv=res['P_pv'],
            P_wind=res['P_wind'],
            P_el=res['P_load'],       # 注意：函数参数叫 P_el，但数据来自 P_load
            Q_cool=res['Q_cool'],
            Q_heat=res['Q_heat'],
            title="数据中心24小时能源调度结果",
            config=res.get('config', None)
        )
        # 将当前图形渲染到 Streamlit
        st.pyplot(plt.gcf())
    except Exception as e:
        st.error(f"绘图时出错: {e}")
        st.code(str(e))

# 可选：显示原始数据表格
with st.expander("📊 查看输入负荷数据"):
    if 'res' in locals():
        st.write("电负荷 (kW):", res['P_load'])
        st.write("冷负荷 (kW):", res['Q_cool'])
        st.write("热负荷 (kW):", res['Q_heat'])

st.markdown("---")
st.caption("© 2026 能源优化团队 | 基于 DEAP + pvlib + windpowerlib")