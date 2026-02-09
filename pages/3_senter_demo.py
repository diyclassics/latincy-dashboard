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
