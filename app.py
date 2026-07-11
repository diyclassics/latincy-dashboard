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
    - [Normalize U/V spelling](uv_normalizer_demo) with rule-based [latincy-uv](https://github.com/diyclassics/latincy-uv)
    - [Correct long-s OCR artifacts](long_s_demo) with [latincy-long-s](https://github.com/diyclassics/latincy-long-s)
    - [Restore Greek diacritics](diacritics_demo) with [latincy-diacritics](https://github.com/diyclassics/latincy-diacritics)
    - [Look up Latin words in Whitaker's Words](lexicon_lookup_demo) with [latincy-lexicon](https://github.com/latincy/latincy-lexicon)
    - [Build a vocabulary list](vocab_demo) from any Latin passage with [latincy-vocab](https://github.com/latincy/latincy-vocab)
    - [Explore macron-based morphology signal](macron_morph_demo) *(experimental)* — how vowel-length marks supplement model predictions
"""
)
