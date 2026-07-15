import pandas as pd
import streamlit as st

from lexicon_helpers import annotate_lexicon, sentence_picker
from model_helpers import load_model

st.set_page_config(page_title="Lexicon Lookup Demo", layout="wide")
st.sidebar.header("Lexicon Lookup Demo")

st.title("Whitaker's Words Lexicon Lookup")
st.markdown(
    """
Run Latin text through LatinCy + Whitaker's Words to get dictionary
glosses and ranked parses for each token.
"""
)

st.info(
    "This lookup is designed for **contextual analysis**: entries are "
    "ranked using upstream LatinCy annotations (POS, morphology, "
    "dependency, NER). Single-word or fragment inputs may rank poorly "
    "because those signals are missing or unreliable — give it a full "
    "sentence for best results."
)

# Shared read-only lg model; Whitaker's Words annotations are applied to each
# doc post-hoc via annotate_lexicon (weightless blank pipeline), so this demo
# reuses the single lg instance instead of loading its own copy.
nlp = load_model("la_core_web_lg")

tab1, tab2 = st.tabs(["Lookup", "About"])

with tab1:
    text = sentence_picker("lookup")

    if st.button("Analyze", key="lookup_analyze"):
        doc = annotate_lexicon(nlp(text))
        rows = []
        lex_data = []
        for token in doc:
            if token.is_punct or token.is_space:
                continue
            lex = token._.lexicon or []
            top_gloss = ""
            if lex:
                glosses = lex[0].get("glosses", [])
                if isinstance(glosses, list) and glosses:
                    top_gloss = glosses[0]
                elif isinstance(glosses, str):
                    top_gloss = glosses
            rows.append(
                {
                    "Token": token.text,
                    "Lemma": token.lemma_,
                    "POS": token.pos_,
                    "Top Gloss": top_gloss[:80],
                }
            )
            lex_data.append({"text": token.text, "entries": lex})
        st.session_state["lookup_rows"] = rows
        st.session_state["lookup_lex"] = lex_data

    if "lookup_rows" in st.session_state:
        df = pd.DataFrame(st.session_state["lookup_rows"])
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Token": st.column_config.TextColumn(width="small"),
                "Lemma": st.column_config.TextColumn(width="small"),
                "POS": st.column_config.TextColumn(width="small"),
                "Top Gloss": st.column_config.TextColumn(width="large"),
            },
        )

        st.markdown("---")
        st.markdown("**Select a token for the full dictionary entry:**")
        lex_data = st.session_state["lookup_lex"]
        options = [f"{d['text']} ({i + 1})" for i, d in enumerate(lex_data)]
        idx = st.selectbox(
            "Token:",
            range(len(options)),
            format_func=lambda i: options[i],
            key="lookup_token",
        )

        if idx is not None:
            tok = lex_data[idx]
            st.subheader(tok["text"])
            if not tok["entries"]:
                st.info("No lexicon entries for this token.")
            else:
                for j, entry in enumerate(tok["entries"], 1):
                    headword = entry.get("headword") or entry.get("lemma", "?")
                    pos = entry.get("pos", "?")
                    st.markdown(f"**Entry {j}:** `{headword}` ({pos})")

                    parts = entry.get("principal_parts")
                    if parts:
                        if isinstance(parts, list):
                            parts = ", ".join(str(p) for p in parts)
                        st.markdown(f"- Principal parts: `{parts}`")

                    glosses = entry.get("glosses")
                    if glosses:
                        if isinstance(glosses, list):
                            st.markdown(
                                "- Glosses:\n"
                                + "\n".join(f"  - {g}" for g in glosses[:6])
                            )
                        else:
                            st.markdown(f"- Gloss: {glosses}")

                    age = entry.get("age")
                    freq = entry.get("frequency") or entry.get("freq")
                    meta = []
                    if age:
                        meta.append(f"Age: `{age}`")
                    if freq:
                        meta.append(f"Frequency: `{freq}`")
                    if meta:
                        st.markdown("- " + "  |  ".join(meta))

                    st.markdown("---")

with tab2:
    st.markdown(
        """
        ## About

        This demo runs Latin text through the full LatinCy pipeline plus
        the `whitakers_words` component from
        [latincy-lexicon](https://github.com/latincy/latincy-lexicon),
        which wraps [Whitaker's Words](https://mk270.github.io/whitakers-words/)
        as a spaCy pipeline component.

        Each token gets:

        - `token._.gloss` — short definition from the top-ranked entry
        - `token._.lexicon` — full dictionary entries with principal parts,
          age/frequency metadata, and glosses
        - `token._.ww` — ranked morphological parses from the WW stem+ending
          engine

        Entries are ranked using upstream LatinCy signals (POS match,
        morphological features, dependency label, NER context) plus WW's
        built-in frequency grades.

        ### First-run note

        On first load, the page builds the WW lexicon and analyzer JSON
        artifacts in memory (~5–10s). Subsequent loads are instant —
        Streamlit caches the pipeline for the session.
        """
    )
