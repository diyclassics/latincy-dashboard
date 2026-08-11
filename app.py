import streamlit as st
from datetime import datetime, timedelta

st.set_page_config(page_title="LatinCy Dashboard", layout="centered")

deployment_date = datetime(2026, 8, 11)
sunset_date = deployment_date + timedelta(days=90)
now = datetime.now()

if now > sunset_date:
    st.error("This application has been retired.")
    st.stop()

st.title("LatinCy Dashboard Retired")
st.markdown("""
### This Streamlit Cloud deployment has been retired.

The LatinCy Dashboard is now hosted at:
### [https://dashboard.exploratoryphilology.org/](https://dashboard.exploratoryphilology.org/)

Please update your bookmarks. This notice will be removed on **November 9, 2026**.
""")
