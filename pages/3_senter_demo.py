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
    value="Decrevit quondam senatus, ut L. Opimius consul videret, ne quid res publica detrimenti caperet; nox nulla intercessit; interfectus est propter quasdam seditionum suspiciones C. Gracchus, clarissimo patre, avo, maioribus, occisus est cum liberis M. Fulvius consularis. Simili senatus consulto C. Mario et L. Valerio consulibus est permissa res publica; num unum diem postea L. Saturninum tribunum pl. et C. Servilium praetorem mors ac rei publicae poena remorata est? At [vero] nos vicesimum iam diem patimur hebescere aciem horum auctoritatis. Habemus enim huiusce modi senatus consultum, verum inclusum in tabulis tamquam in vagina reconditum, quo ex senatus consulto confestim te interfectum esse, Catilina, convenit. Vivis, et vivis non ad deponendam, sed ad confirmandam audaciam. Cupio, patres conscripti, me esse clementem, cupio in tantis rei publicae periculis me non dissolutum videri, sed iam me ipse inertiae nequitiaeque condemno.",
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
