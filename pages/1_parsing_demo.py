import streamlit as st
import pandas as pd
import datetime

from model_helpers import load_model

st.set_page_config(page_title="Parsing Demo", layout="wide")
st.sidebar.header("Parsing Demo")

# Sample passages, offered as bubbles (st.pills) below the text box. Drawn from
# the latincy-pipelines smoke tests (scripts/preflight.py), but given as
# COMPLETE sentences — where the smoke test used only an incipit (the Aeneid
# opening), it is extended to the full first sentence, since the models parse
# whole sentences more reliably than fragments. Cicero rounds out the set.
SAMPLE_PASSAGES = {
    "Seneca, Ep. 1.1": "Ita fac, mi Lucili; vindica te tibi, et tempus quod adhuc aut auferebatur aut subripiebatur aut excidebat collige et serva.",
    "Vergil, Aen. 1.1": "Arma virumque cano, Troiae qui primus ab oris Italiam fato profugus Laviniaque venit litora, multum ille et terris iactatus et alto vi superum saevae memorem Iunonis ob iram, multa quoque et bello passus, dum conderet urbem inferretque deos Latio, genus unde Latinum Albanique patres atque altae moenia Romae.",
    "Caesar, B.G. 1.1": "Gallia est omnis divisa in partes tres, quarum unam incolunt Belgae, aliam Aquitani, tertiam qui ipsorum lingua Celtae, nostra Galli appellantur.",
    "Ritchie, Fab. 1": "Olim in Graecia puer erat, qui Hercules appellabatur.",
    "Cicero, Cat. 1.1": "Quo usque tandem abutere, Catilina, patientia nostra? quam diu etiam furor iste tuus nos eludet? quem ad finem sese effrenata iactabit audacia?",
}
default_text = SAMPLE_PASSAGES["Seneca, Ep. 1.1"]


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
    # The text area is keyed on session_state so the sample-passage bubbles can
    # populate it: a pill's on_change callback sets the value, which the text
    # area picks up on the following rerun.
    if "parse_input" not in st.session_state:
        st.session_state["parse_input"] = default_text

    def load_sample():
        picked = st.session_state.get("sample_pick")
        if picked:
            st.session_state["parse_input"] = SAMPLE_PASSAGES[picked]

    text = st.text_area(
        "Enter some text to analyze (max 500 tokens)", key="parse_input", height=200
    )
    st.pills(
        "Or load a sample passage:",
        list(SAMPLE_PASSAGES),
        selection_mode="single",
        key="sample_pick",
        on_change=load_sample,
    )
    if st.button("Analyze"):
        st.session_state["parse_df"] = analyze_text(text)

    # Render the results from session_state, NOT inside the `if st.button` block.
    # st.button is True only on the single rerun right after the click, so
    # rendering the table there means the async st.dataframe grid gets torn down
    # by the next rerun (e.g. the download button) before it paints — that was
    # the blank-table-on-HF bug. Rendering from session_state paints on every
    # stable rerun, matching the lexicon demo (which works on HF). It also makes
    # the download button non-destructive: the table survives the rerun.
    if "parse_df" in st.session_state:
        df = st.session_state["parse_df"]
        sent_count = df["sent_id"].nunique()
        st.text(
            f"Analyzed {len(df)} tokens in {sent_count} sentences with {model_name} model."
        )
        # Display an all-string copy with all-TextColumn config, matching the
        # lexicon demo's dataframe exactly (which renders fine on HF). The
        # original df (with int token_id/head) still feeds the TSV download.
        st.dataframe(
            df.astype(str),
            use_container_width=True,
            hide_index=True,
            column_config={
                "sent_id": st.column_config.TextColumn(width="small"),
                "token_id": st.column_config.TextColumn(width="small"),
                "form": st.column_config.TextColumn(width="small"),
                "lemma": st.column_config.TextColumn(width="small"),
                "upos": st.column_config.TextColumn(width="small"),
                "xpos": st.column_config.TextColumn(width="small"),
                "feats": st.column_config.TextColumn(width="large"),
                "head": st.column_config.TextColumn(width="small"),
                "deprel": st.column_config.TextColumn(width="small"),
                "ent_type": st.column_config.TextColumn(width="small"),
            },
        )

        csv = df.to_csv(index=False, sep="\t").encode("utf-8")
        st.download_button(
            "Press to Download",
            csv,
            f"latincy-analysis-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.tsv",
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
