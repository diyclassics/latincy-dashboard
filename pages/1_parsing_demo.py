import streamlit as st
import pandas as pd
import datetime

from model_helpers import load_model

st.set_page_config(page_title="Parsing Demo", layout="wide")
st.sidebar.header("Parsing Demo")

default_text = """Ita fac, mi Lucili; vindica te tibi, et tempus, quod adhuc aut auferebatur aut subripiebatur aut excidebat, collige et serva."""


def format_morph(morph):
    morph = morph.to_dict()
    if morph:
        return ", ".join([f"{k}={v}" for k, v in morph.items()])
    else:
        return ""


def analyze_text(text):
    doc = nlp(text)
    rows = []
    token_count = 0
    for sent_idx, sent in enumerate(doc.sents):
        sent_id = f"s{sent_idx + 1}"
        sent_start = sent.start
        for token_idx, token in enumerate(sent):
            if token_count >= 500:
                break
            token_id = token_idx + 1
            if token.head == token:
                head = 0
            else:
                head = token.head.i - sent_start + 1
            rows.append(
                (
                    sent_id,
                    token_id,
                    token.text,
                    token.lemma_,
                    token.pos_,
                    token.tag_,
                    format_morph(token.morph),
                    head,
                    token.dep_,
                    token.ent_type_,
                )
            )
            token_count += 1
        if token_count >= 500:
            break
    df = pd.DataFrame(
        rows,
        columns=[
            "sent_id",
            "token_id",
            "form",
            "lemma",
            "upos",
            "xpos",
            "feats",
            "head",
            "deprel",
            "ent_type",
        ],
    )
    return df


st.title("LatinCy Text Analyzer")

# Shared, cached lg model — warmed at startup by the home-page preloader, so
# this returns instantly on all but the very first (cold) load.
model_name = "la_core_web_lg"
nlp = load_model(model_name)

st.write(f"Loaded model: {model_name} (v{nlp.meta['version']})")

df = None

tab1, tab2 = st.tabs(["Analyze", "About"])

with tab1:
    text = st.text_area(
        "Enter some text to analyze (max 500 tokens)", value=default_text, height=200
    )
    if st.button("Analyze"):
        df = analyze_text(text)
        sent_count = df["sent_id"].nunique()
        st.text(f"Analyzed {len(df)} tokens in {sent_count} sentences with {model_name} model.")
        # Static HTML table (st.table) instead of the interactive st.dataframe:
        # the dataframe's JS grid silently fails to paint behind HF Spaces'
        # proxy (renders blank after the summary line), while a plain HTML
        # table always renders. Index hidden for a clean CoNLL-U-style view.
        st.table(df.style.hide(axis="index"))

        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False, sep="\t").encode("utf-8")

        csv = convert_df(df)

        def create_timestamp():
            return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # nb: clicking this button resets app! Open streamlit issue, as of 4.15.2023; cf. https://github.com/streamlit/streamlit/issues/4382
        st.markdown("*NB: Clicking the download button will reset the app after download!*")
        st.download_button(
            "Press to Download",
            csv,
            f"latincy-analysis-{create_timestamp()}.tsv",
            "text/csv",
            key="download-csv",
        )

with tab2:
    st.markdown("""
    ## About

    This demo produces a **Universal Dependencies (UD)** style tabular
    analysis of Latin text, showing the full linguistic annotation that
    LatinCy predicts for each token.

    ### What Is UD Parsing?

    [Universal Dependencies](https://universaldependencies.org/) is a
    framework for consistent grammatical annotation across languages.
    A UD parse assigns every token its part of speech, morphological
    features, and a labeled syntactic dependency linking it to its
    grammatical head. The result is a complete, machine-readable
    description of sentence structure.

    ### Output Columns

    The table follows the standard
    [CoNLL-U format](https://universaldependencies.org/format.html):

    | Column | CoNLL-U Field | Description |
    |--------|---------------|-------------|
    | **sent_id** | — | Sentence identifier (added by this demo) |
    | **token_id** | ID | Position of the token within its sentence (1-indexed) |
    | **form** | FORM | The surface form exactly as it appears in the text |
    | **lemma** | LEMMA | Dictionary headword (*e.g.* *tempus* for *tempora*) |
    | **upos** | UPOS | Universal part-of-speech tag (NOUN, VERB, ADJ, ADV, etc.) |
    | **xpos** | XPOS | Language-specific POS tag from the Latin tagset |
    | **feats** | FEATS | Morphological features: Case, Number, Gender, Tense, Mood, Voice, etc. |
    | **head** | HEAD | Index of the syntactic head token (0 = sentence root) |
    | **deprel** | DEPREL | Dependency relation to head (*nsubj*, *obj*, *obl*, *advmod*, *amod*, etc.) |
    | **ent_type** | — | Named entity type, if any (PER, LOC, NORP) |

    ### Model Details

    - **Model:** `la_core_web_lg` — LatinCy v3.9.6
    - **Training data:** Harmonized annotations from 6 UD Latin treebanks
      (Perseus, PROIEL, ITTB, LLCT, UDante, CIRCSE) + LASLA
    - **Framework:** [spaCy](https://spacy.io/) v3

    ### Accuracy (v3.9.6, lg model)

    | Component | Score |
    |-----------|-------|
    | **POS (UPOS)** | 97.26% |
    | **Morphology** | 90.37% |
    | **Lemma** | 95.26% |
    | **Dependency UAS** | 83.50% |
    | **Dependency LAS** | 78.53% |
    | **Sentence segmentation** | 92.64% F1 |
    | **NER** | 87.46% F1 |

    Scores evaluated on held-out UD test set; NER on dev set.

    ### Notes

    - Output is limited to 500 tokens
    - TSV export follows [CoNLL-U format](https://universaldependencies.org/format.html)

    ### References

    - [Universal Dependencies](https://universaldependencies.org/) — annotation guidelines and treebank data
    - [UD Latin treebanks](https://universaldependencies.org/la/index.html) — all Latin UD resources
    - [LatinCy on HuggingFace](https://huggingface.co/latincy) — models and documentation
    """)
