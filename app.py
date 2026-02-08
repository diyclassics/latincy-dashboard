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
    - [Segment a paragraph into sentences](senter_demo)
    - [Highlight named entities](ner_demo) (people, places, groups) in Latin text
    - [Visualize dependency trees](dependency_demo) showing grammatical structure
    - [Explore word similarity](similarity_demo) using Latin word vectors
    - [Analyze morphology](morphology_demo) — lemma, case, gender, tense, and more
"""
)
