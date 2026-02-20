import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 导入你的绘图函数
from plot_results import plot_scheduling

# ========== 模拟数据（实际替换为真实优化器） ==========
def mock_optimization_result():
    np.random.seed(42)
    x_opt = np.random.rand(9 * 24) * 100
    return {
        'x_opt': x_opt,
        'P_pv': np.clip(np.sin(np.linspace(0, 3.14, 24)) * 100, 0, None),
        'P_wind': np.random.rand(24) * 60,
        'P_load': np.random.rand(24) * 120 + 50,
        'Q_cool': np.random.rand(24) * 200 + 100,
        'Q_heat': np.random.rand(24) * 80 + 30,
        'config': {'BESS_CAPACITY': 500, 'TES_CAPACITY': 2000}
    }

# ========== 页面配置 ==========
st.set_page_config(
    page_title="数据中心能源调度系统",
    page_icon="🔋",
    layout="wide",  # 关键：宽屏布局
    initial_sidebar_state="expanded"
)

# ========== 自定义 CSS 美化 ==========
st.markdown("""
<style>
    /* 主背景 */
    .main { background-color: #f8f9fa; }
    
    /* 标题样式 */
    h1 { 
        color: #1e3a8a; 
        font-weight: 700; 
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* 卡片容器 */
    .plot-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-top: 1rem;
    }
    
    /* 按钮样式 */
    .stButton>button {
        background-color: #1e40af;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(30, 64, 175, 0.3);
    }
    
    /* 脚注 */
    footer { visibility: hidden; }
    .footer { 
        text-align: center; 
        color: #64748b; 
        font-size: 0.9rem; 
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ========== 侧边栏 ==========
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=Energy+AI", use_container_width=True)
    st.title("⚙️ 控制面板")
    st.markdown("### 调度参数设置")
    location = st.selectbox("📍 地点", ["北京", "上海", "深圳"])
    season = st.selectbox("🌦️ 季节", ["夏季", "冬季", "春秋季"])
    mode = st.radio("🎯 优化目标", ["经济性优先", "碳排最低", "综合最优"])
    st.divider()
    st.info("点击下方按钮运行24小时调度优化")

# ========== 主内容区 ==========
st.title("🔋 数据中心多能协同调度优化系统")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("总用电量", "2,840 kWh", "+5% vs 昨日")
with col2:
    st.metric("可再生能源占比", "68%", "↑ 12%")
with col3:
    st.metric("碳排放", "120 kgCO₂", "↓ 18%")

st.markdown("<br>", unsafe_allow_html=True)

# 运行按钮居中
col_center = st.columns([1, 2, 1])
with col_center[1]:
    run_button = st.button("🚀 运行能源调度优化", use_container_width=True)

if run_button:
    with st.spinner("正在计算最优调度策略..."):
        res = mock_optimization_result()
    
    st.success("✅ 优化完成！调度方案已生成")
    
    # ====== 渲染图表 ======
    plt.clf()
    try:
        plot_scheduling(
            x_opt=res['x_opt'],
            P_pv=res['P_pv'],
            P_wind=res['P_wind'],
            P_el=res['P_load'],
            Q_cool=res['Q_cool'],
            Q_heat=res['Q_heat'],
            title="",
            config=res.get('config', None)
        )
        
        # 包裹在美化容器中
        with st.container():
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.pyplot(plt.gcf(), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"绘图失败: {str(e)}")

# ========== 底部信息 ==========
st.markdown('<div class="footer">© 2026 智慧能源实验室 | 支持实时调度与碳流追踪</div>', unsafe_allow_html=True)