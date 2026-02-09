import streamlit as st
import spacy
from spacy_streamlit import visualize_parser

st.set_page_config(page_title="Dependency Demo", layout="wide")
st.sidebar.header("Dependency Demo")

default_text = """Quo usque tandem abutere, Catilina, patientia nostra? Quam diu etiam furor iste tuus nos eludet? Quem ad finem sese effrenata iactabit audacia?"""

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
