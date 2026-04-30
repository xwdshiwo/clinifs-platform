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
    st.Page("pages/1_run.py",       title="Run Analysis",       icon="🔬"),
    st.Page("pages/2_browse.py",    title="Browse Results",     icon="📊"),
    st.Page("pages/3_recommend.py", title="Get Recommendation", icon="🎯"),
    st.Page("pages/4_custom_rra.py",title="Custom RRA",         icon="⚙️"),
]

pg = st.navigation(pages)
pg.run()
