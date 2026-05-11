"""
clinifs – Cancer Gene Expression Feature Selection Toolkit
Streamlit web application

Run locally:
    cd app
    streamlit run main.py

Deploy to Streamlit Cloud:
    Push this directory to GitHub, then connect at share.streamlit.io.
"""
import streamlit as st

st.set_page_config(
    page_title="clinifs | Feature Selection Toolkit",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page("pages/1_run.py",       title="Online Analysis / 在线分析",        icon="🔬"),
    st.Page("pages/2_browse.py",    title="Benchmark Results / 基准结果",      icon="📊"),
    st.Page("pages/3_recommend.py", title="Method Guide / 方法指南",           icon="🎯"),
    st.Page("pages/4_custom_rra.py",title="RRA & Gene Panel / RRA 与基因面板", icon="⚙️"),
    st.Page("pages/5_help.py",      title="Help / 帮助",                      icon="❓"),
    st.Page("pages/6_contact.py",   title="Contact / 联系我们",               icon="📬"),
]

pg = st.navigation(pages)
pg.run()
