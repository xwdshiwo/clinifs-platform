import streamlit as st

LANG_OPTIONS = {"English": "en", "中文": "zh"}


def init_language():
    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"
    with st.sidebar:
        choice = st.radio(
            "Language / 语言",
            options=list(LANG_OPTIONS.keys()),
            index=0 if st.session_state["lang"] == "en" else 1,
            horizontal=True,
            key="language_choice",
        )
    st.session_state["lang"] = LANG_OPTIONS[choice]
    return st.session_state["lang"]


def get_lang():
    return st.session_state.get("lang", "en")


def tr(en, zh):
    return zh if get_lang() == "zh" else en
