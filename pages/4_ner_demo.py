import streamlit as st
import spacy
from spacy_streamlit import visualize_ner

st.set_page_config(page_title="NER Demo", layout="wide")
st.sidebar.header("NER Demo")

default_text = """Iason et Medea e Thessalia expulsi ad urbem Corinthum venerunt, cuius urbis Creon quidam regnum tum obtinebat."""

st.title("Latin Named Entity Recognition")

st.markdown(
    """
This demo highlights **named entities** in Latin text — people, places,
and groups — using LatinCy's NER component.

**Entity types:**
- **PER** — People (*Cicero, Caesar*)
- **LOC** — Locations (*Roma, Gallia*)
- **NORP** — Nationalities / groups (*Belgae, Aquitani*)
"""
)

model_selectbox = st.sidebar.selectbox(
    "Choose model:", ("la_core_web_lg", "la_core_web_md", "la_core_web_sm")
)


@st.cache_resource
def load_model(model_name):
    return spacy.load(model_name)


nlp = load_model(model_selectbox)

tab1, tab2 = st.tabs(["Recognize", "About"])

with tab1:
    text = st.text_area(
        "Enter Latin text to analyze (max ~200 tokens):", value=default_text, height=200
    )

    if st.button("Find Entities"):
        tokens = text.split()
        if len(tokens) > 200:
            st.warning("Text trimmed to ~200 tokens.")
            text = " ".join(tokens[:200])
        doc = nlp(text)
        ner_labels = nlp.get_pipe("ner").labels
        visualize_ner(doc, labels=ner_labels, show_table=False, title="")

with tab2:
    st.markdown("""
    ## About

    LatinCy's NER component identifies **named entities** in Latin text.

    ### Entity Types

    | Label | Description | Examples |
    |-------|-------------|----------|
    | **PER** | Person names | *Cicero*, *Caesar*, *Medea* |
    | **LOC** | Locations | *Roma*, *Gallia*, *Corinthus* |
    | **NORP** | Nationalities, groups, peoples | *Belgae*, *Aquitani*, *Romani* |

    ### Training Data

    The NER model was trained on manually annotated Latin texts spanning
    multiple genres: historical prose, poetry, biblical text, and
    philosophical works. Sources include the Vulgate (Matthew), Ovid's
    Metamorphoses, and Augustine's *De Civitate Dei*.

    ### Performance (v3.8.0, lg model)

    - **Overall NER F1:** 85.0%
    - Best on PER and LOC; NORP is harder due to ambiguity with adjectives

    ### Source

    [GitHub: diyclassics/latincy](https://github.com/diyclassics/latincy)
    """)
