import streamlit as st
import spacy
from spacy_streamlit import visualize_parser

st.set_page_config(page_title="Dependency Demo", layout="wide")
st.sidebar.header("Dependency Demo")

default_text = """Haec narrantur a poetis de Perseo. Perseus filius erat Iovis, maximi deorum; avus eius Acrisius appellabatur."""

st.title("Latin Dependency Tree Visualizer")

st.markdown(
    """
This demo renders **dependency parse trees** for Latin sentences, showing
how words relate to each other grammatically.

**Common dependency labels:**
- **nsubj** — nominal subject
- **obj** — direct object
- **obl** — oblique argument (e.g. prepositional phrases)
- **amod** — adjectival modifier
- **nmod** — nominal modifier
- **advmod** — adverbial modifier
- **conj** — conjunct
"""
)

model_selectbox = st.sidebar.selectbox(
    "Choose model:", ("la_core_web_lg", "la_core_web_md", "la_core_web_sm")
)

compact = st.sidebar.checkbox("Compact mode", value=False)


@st.cache_resource
def load_model(model_name):
    return spacy.load(model_name)


nlp = load_model(model_selectbox)

tab1, tab2 = st.tabs(["Parse", "About"])

with tab1:
    text = st.text_area(
        "Enter Latin text (keep to 3–5 sentences for readable trees):",
        value=default_text,
        height=200,
    )

    if st.button("Parse"):
        doc = nlp(text)
        sents = list(doc.sents)
        if len(sents) > 10:
            st.warning("Showing first 10 sentences only.")
            sents = sents[:10]
        for i, sent in enumerate(sents):
            st.markdown(f"**Sentence {i + 1}**")
            sent_doc = nlp(sent.text)
            visualize_parser(
                sent_doc,
                title="",
                displacy_options={"compact": compact},
                key=f"dep_sent_{i}",
            )

with tab2:
    st.markdown("""
    ## About

    This demo renders **dependency parse trees** using the
    [Universal Dependencies](https://universaldependencies.org/) framework.

    ### Common Dependency Relations

    | Label | Relation | Example |
    |-------|----------|---------|
    | **nsubj** | Nominal subject | *Caesar* venit |
    | **obj** | Direct object | librum *legit* |
    | **obl** | Oblique (prep. phrases) | *in urbe* |
    | **amod** | Adjectival modifier | vir *bonus* |
    | **nmod** | Nominal modifier | *urbis* murus |
    | **advmod** | Adverbial modifier | *bene* facit |
    | **conj** | Conjunct | *Caesar et Pompeius* |
    | **cop** | Copula | bonus *est* |
    | **xcomp** | Open clausal complement | *dicere* vult |
    | **advcl** | Adverbial clause | *cum venisset* |

    ### Notes

    - Trees follow UD v2 annotation guidelines
    - The root of each sentence is the main predicate
    - Compact mode collapses the tree vertically for long sentences
    - Trained on 5 Latin UD treebanks: Perseus, PROIEL, ITTB, LLCT, UDante

    ### Reference

    [Universal Dependencies for Latin](https://universaldependencies.org/la/)
    """)
