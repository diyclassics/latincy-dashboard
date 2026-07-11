import streamlit as st

from dcc_helpers import is_dcc_core
from lexicon_helpers import load_lexicon_pipeline
from vocabbuilder.core.config import PipelineConfig
from vocabbuilder.core.models import VocabList
from vocabbuilder.data.gloss_provider import GlossProvider
from vocabbuilder.processors.vocab_core import build_vocab_list
from vocabbuilder.utils.normalization import to_u_form

st.set_page_config(page_title="Vocab Builder Demo", layout="wide")
st.sidebar.header("Vocab Builder Demo")

st.title("Latin Vocabulary Builder")
st.markdown(
    """
Generate a textbook-style vocabulary list from any Latin passage using
[latincy-vocab](https://github.com/latincy/latincy-vocab) —
lemmatized, glossed, and sortable.
"""
)

DEFAULT_TEXT = (
    "Haec narrantur a poetis de Perseo. Perseus filius erat Iovis, maximi "
    "deorum; avus eius Acrisius appellabatur. Acrisius volebat Perseum nepotem "
    "suum necare; nam propter oraculum puerum timebat."
)


@st.cache_resource(show_spinner="Loading vocabulary pipeline…")
def load_resources():
    # Use the shared lexicon pipeline so token._.lexicon is populated,
    # giving us citation forms (principal parts, gen+gender, -a,-um).
    nlp = load_lexicon_pipeline("la_core_web_lg")
    config = PipelineConfig(spacy_model="la_core_web_lg", spacy_disable=[])
    config.resolve_data_paths()
    gloss_provider = GlossProvider(config.glosses_path) if config.glosses_path else None
    return nlp, config, gloss_provider


nlp, _config, _gloss_provider = load_resources()


def process(text: str) -> VocabList:
    doc = nlp(text)
    vocab = build_vocab_list(doc, _config)
    if _gloss_provider:
        for entry in vocab:
            result = _gloss_provider.lookup(entry.lemma, entry.pos, _config.max_glosses)
            if result:
                entry.glosses = result.glosses
                entry.display_lemma = result.display_lemma
            else:
                entry.display_lemma = _gloss_provider.get_display_lemma(entry.lemma)
    return vocab


tab1, tab2 = st.tabs(["Vocab List", "About"])

with tab1:
    col_input, col_vocab = st.columns([2, 3])

    with col_input:
        text = st.text_area(
            "Latin text:",
            value=DEFAULT_TEXT,
            height=280,
            key="vocab_text",
        )
        run = st.button("Build Vocabulary List", key="vocab_run")

    if run:
        with st.spinner("Processing…"):
            vocab: VocabList = process(text)
        st.session_state["vocab_list"] = vocab
        st.session_state["vocab_sort"] = "alpha"

    with col_vocab:
        if "vocab_list" in st.session_state:
            vocab: VocabList = st.session_state["vocab_list"]

            s1, s2, s3 = st.columns(3)
            with s1:
                if st.button("Alphabetical", key="sort_alpha", use_container_width=True):
                    st.session_state["vocab_sort"] = "alpha"
            with s2:
                if st.button("First Occurrence", key="sort_occ", use_container_width=True):
                    st.session_state["vocab_sort"] = "occurrence"
            with s3:
                if st.button("Frequency", key="sort_freq", use_container_width=True):
                    st.session_state["vocab_sort"] = "freq"

            hide_dcc = st.checkbox(
                "Hide DCC Core words",
                value=False,
                key="hide_dcc",
                help="Remove high-frequency words from the DCC Core Latin Vocabulary",
            )

            sort_key = st.session_state.get("vocab_sort", "alpha")
            if sort_key == "alpha":
                sorted_vocab = vocab.by_alpha()
            elif sort_key == "occurrence":
                sorted_vocab = vocab.by_first_occurrence()
            else:
                sorted_vocab = vocab.by_frequency()

            visible = [e for e in sorted_vocab if e.headword and e.headword[0].isalpha()]
            dcc_count = sum(1 for e in visible if is_dcc_core(to_u_form(e.lemma)))
            new_count = len(visible) - dcc_count

            if hide_dcc:
                visible = [e for e in visible if not is_dcc_core(to_u_form(e.lemma))]

            st.caption(
                f"{len(visible)} entries shown · {new_count} new, {dcc_count} DCC core · "
                "assembled from probabilistic LatinCy pipeline annotations — verify before use"
            )

            lines = []
            for entry in visible:
                in_dcc = is_dcc_core(to_u_form(entry.lemma))
                hw_text = entry.headword.lower()
                hw = f"<strong>{hw_text}</strong>"
                pm = f" <em>{entry.pos_marker}</em>" if entry.pos_marker else ""
                gloss = f", {entry.short_gloss}" if entry.short_gloss else ""
                freq = f" <em>×{entry.frequency}</em>" if entry.frequency > 1 else ""
                dcc_tag = (
                    " <span style='color:#09a3d5;font-size:0.75em'>dcc</span>"
                    if in_dcc else ""
                )
                lines.append(
                    f"<p style='margin:0 0 2px 0;line-height:1.4'>"
                    f"{hw}{pm}{gloss}{freq}{dcc_tag}</p>"
                )

            st.markdown("".join(lines), unsafe_allow_html=True)

with tab2:
    st.markdown(
        """
### About

This demo uses **latincy-vocab** (`vocabbuilder.VocabPipeline`) to process a Latin
passage end-to-end:

1. **spaCy** tokenizes and annotates (lemma, POS, morphology)
2. **latincy-lexicon** supplies Whitaker's Words glosses, display lemmas, and
   citation forms (principal parts for verbs; nominative, genitive, and gender
   for nouns; nominative with -a, -um endings for adjectives)
3. `VocabList` deduplicates by lemma+POS and exposes three orderings

The **Hide DCC Core words** checkbox filters out the ~1,000 high-frequency lemmas
from the [DCC Core Latin Vocabulary](https://dcc.dickinson.edu/vocab/core-vocabulary),
leaving only words a student would need to look up.

> **Note:** vocabulary entries are assembled from probabilistic LatinCy pipeline
> annotations and should be verified before use in publication or teaching.

The default text is §1 of Ritchie's *Fabulae Faciles* (1902).

**Sort modes**
- *Alphabetical* — traditional glossary order
- *First Occurrence* — passage reading order (as in textbook footnotes)
- *Frequency* — most common lemmas first
"""
    )
