import streamlit as st

st.set_page_config(
    page_title="LatinCy Dashboard | Home",
    page_icon="🏠",
)

st.write("# LatinCy Dashboard")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    LatinCy is a collection of Latin language models for spaCy.

    ### See the demos
    - [Get basic spaCy data from a short text](parsing_demo)
    - [Visualize a custom span label](custom_label_demo), here tokens covered by the [DCC Core Latin Vocabulary](https://dcc.dickinson.edu/latin-core-list1)
"""
)
