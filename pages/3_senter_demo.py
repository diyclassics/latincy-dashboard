import streamlit as st
import spacy
import datetime

st.set_page_config(page_title="Sentence Segmenter", layout="wide")
st.sidebar.header("Sentence Segmenter")


# Load spaCy model (Latin large)
@st.cache_resource
def load_model():
    return spacy.load("la_core_web_lg")


nlp = load_model()

st.title("Latin Sentence Segmenter")

# Input text area
text = st.text_area(
    "Enter a paragraph of Latin text to segment into sentences:",
    value="Lucius Catilina, nobili genere natus, fuit magna vi et animi et corporis, sed ingenio malo pravoque. Huic ab adulescentia bella intestina, caedes, rapinae, discordia civilis grata fuere ibique iuventutem suam exercuit. Corpus patiens inediae, algoris, vigiliae supra quam cuiquam credibile est. Animus audax, subdolus, varius, cuius rei lubet simulator ac dissimulator, alieni adpetens, sui profusus, ardens in cupiditatibus; satis eloquentiae, sapientiae parum. Vastus animus inmoderata, incredibilia, nimis alta semper cupiebat.",
    height=200,
)

tab1, tab2 = st.tabs(["Segment", "About"])

with tab1:
    sentences = []
    if st.button("Segment Sentences"):
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        st.success(f"Found {len(sentences)} sentences.")

    if sentences:
        # Show sentences in a text area for easy copy/paste
        sentences_text = "\n".join(sentences)
        sentences_text += "\n"
        st.text_area("Sentences (one per line):", value=sentences_text, height=400)

        # Download button
        def create_timestamp():
            return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        st.download_button(
            "Download Sentences as .txt",
            sentences_text,
            f"latin-sentences-{create_timestamp()}.txt",
            "text/plain",
            key="download-txt",
        )

with tab2:
    st.markdown("""
    ## About

    LatinCy includes a dedicated **sentence segmentation** (senter) component
    trained specifically for Latin text.

    ### Why Latin Needs a Custom Senter

    Standard sentence splitters fail on Latin because:
    - Abbreviations use periods (*e.g.*, *Aug.*, *Liv.*)
    - Weak stops like semicolons and colons often mark sentence boundaries
    - Case variation in manuscripts (all-caps, no-caps)

    ### Model Details

    - **Architecture:** spaCy `senter` (token-level binary classifier)
    - **Accuracy:** 99.71% F1 on UD Latin test data
    - **Training data:** Combined UD Latin treebanks (Perseus, PROIEL, ITTB, LLCT, UDante)

    ### Source

    [GitHub: diyclassics/latincy](https://github.com/diyclassics/latincy)
    """)
