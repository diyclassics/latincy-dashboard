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
    value="Haec narrantur a poetis de Perseo. Perseus filius erat Iovis, maximi deorum; avus eius Acrisius appellabatur. Acrisius volebat Perseum nepotem suum necare; nam propter oraculum puerum timebat. Comprehendit igitur Perseum adhuc infantem, et cum matre in arca lignea inclusit. Tum arcam ipsam in mare coniecit. Danae, Persei mater, magnopere territa est; tempestas enim magna mare turbabat. Perseus autem in sinu matris dormiebat.",
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
