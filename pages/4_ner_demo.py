import streamlit as st
from spacy_streamlit import visualize_ner

from model_helpers import load_model

st.set_page_config(page_title="NER Demo", layout="wide")
st.sidebar.header("NER Demo")

default_text = """Iason et Medea e Thessalia expulsi ad urbem Corinthum venerunt, cuius urbis Creon quidam regnum tum obtinebat."""

st.title("Latin Named Entity Recognition")

st.markdown(
    """
This demo highlights **named entities** in Latin text — people, places,
and groups — using LatinCy's NER component.

**Entity types:** PERSON, LOC, NORP
"""
)

nlp = load_model("la_core_web_lg")

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

    **Named Entity Recognition (NER)** is the task of identifying and
    classifying named entities — such as people, places, and events — in
    unstructured text. LatinCy's NER component locates spans of tokens that
    refer to real-world entities and assigns each span a category label.

    ### Entity Types

    | Label | Description | Examples |
    |-------|-------------|----------|
    | **PERSON** | Named individuals | *Cicero*, *Caesar*, *Medea* |
    | **LOC** | Locations (cities, rivers, mountains, regions) | *Roma*, *Rhenus*, *Gallia* |
    | **NORP** | Nationalities, religious/political groups | *Belgae*, *Aquitani*, *Romani* |

    ### Training Data

    The NER model was trained on manually annotated Latin texts from
    multiple sources:

    - **Tesserae texts** — a broad corpus of classical Latin prose and poetry
    - **DCD annotations** — annotations derived from Augustine's
      *De Civitate Dei*
    - **Biblical texts** — annotated passages from the Vulgate

    These sources span historical prose, poetry, philosophical works, and
    biblical text, giving the model exposure to diverse entity contexts.

    ### Model Performance

    NER F1 for the deployed model, with the transformer pipeline shown for
    reference:

    | Pipeline | NER F1 |
    |----------|--------|
    | **la_core_web_lg** (deployed) | 82.6% |
    | la_core_web_trf (reference) | 84.8% |

    The transformer pipeline (`trf`) scores higher but needs a GPU or far more
    compute; `lg` gives near-transformer accuracy on CPU, which is why the
    dashboard ships it.

    ### Source

    [LatinCy on HuggingFace](https://huggingface.co/latincy)
    """)
