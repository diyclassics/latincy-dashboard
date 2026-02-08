import streamlit as st
import spacy
import pandas as pd

st.set_page_config(page_title="Morphology Demo", layout="wide")
st.sidebar.header("Morphology Demo")

default_text = """Arma virumque cano, Troiae qui primus ab oris Italiam, fato profugus, Laviniaque venit litora."""

# Human-readable labels for morphological feature values.
# Keyed by (Feature, Value) to disambiguate collisions like
# Tense=Imp (Imperfect) vs Mood=Imp (Imperative) and
# Degree=Sup (Superlative) vs VerbForm=Sup (Supine).
MORPH_LABELS = {
    # Case
    ("Case", "Nom"): "Nominative",
    ("Case", "Gen"): "Genitive",
    ("Case", "Dat"): "Dative",
    ("Case", "Acc"): "Accusative",
    ("Case", "Abl"): "Ablative",
    ("Case", "Voc"): "Vocative",
    ("Case", "Loc"): "Locative",
    # Number
    ("Number", "Sing"): "Singular",
    ("Number", "Plur"): "Plural",
    # Gender
    ("Gender", "Masc"): "Masculine",
    ("Gender", "Fem"): "Feminine",
    ("Gender", "Neut"): "Neuter",
    # Tense
    ("Tense", "Pres"): "Present",
    ("Tense", "Past"): "Past",
    ("Tense", "Fut"): "Future",
    ("Tense", "Imp"): "Imperfect",
    ("Tense", "Pqp"): "Pluperfect",
    # Mood
    ("Mood", "Ind"): "Indicative",
    ("Mood", "Sub"): "Subjunctive",
    ("Mood", "Imp"): "Imperative",
    ("Mood", "Inf"): "Infinitive",
    # Voice
    ("Voice", "Act"): "Active",
    ("Voice", "Pass"): "Passive",
    # Person
    ("Person", "1"): "1st Person",
    ("Person", "2"): "2nd Person",
    ("Person", "3"): "3rd Person",
    # Degree
    ("Degree", "Pos"): "Positive",
    ("Degree", "Cmp"): "Comparative",
    ("Degree", "Sup"): "Superlative",
    # VerbForm
    ("VerbForm", "Fin"): "Finite",
    ("VerbForm", "Ger"): "Gerund",
    ("VerbForm", "Gdv"): "Gerundive",
    ("VerbForm", "Part"): "Participle",
    ("VerbForm", "Sup"): "Supine",
}

FEATURE_LABELS = {
    "Case": "Case",
    "Number": "Number",
    "Gender": "Gender",
    "Tense": "Tense",
    "Mood": "Mood",
    "Voice": "Voice",
    "Person": "Person",
    "Degree": "Degree",
    "VerbForm": "Verb Form",
    "Aspect": "Aspect",
    "NumType": "Numeral Type",
    "PronType": "Pronoun Type",
    "Poss": "Possessive",
    "Reflex": "Reflexive",
}

POS_LABELS = {
    "NOUN": "Noun",
    "VERB": "Verb",
    "ADJ": "Adjective",
    "ADV": "Adverb",
    "PROPN": "Proper Noun",
    "PRON": "Pronoun",
    "DET": "Determiner",
    "ADP": "Adposition",
    "AUX": "Auxiliary",
    "CCONJ": "Coordinating Conjunction",
    "SCONJ": "Subordinating Conjunction",
    "NUM": "Numeral",
    "PART": "Particle",
    "INTJ": "Interjection",
    "PUNCT": "Punctuation",
    "X": "Other",
}


def _get_val_label(feat, val):
    """Look up human-readable label for a morph value, with feature context."""
    return MORPH_LABELS.get((feat, val), val)


def format_morph_readable(morph):
    """Convert spaCy morph to human-readable string."""
    morph_dict = morph.to_dict()
    if not morph_dict:
        return ""
    parts = []
    for feat, val in morph_dict.items():
        feat_label = FEATURE_LABELS.get(feat, feat)
        val_label = _get_val_label(feat, val)
        parts.append(f"{feat_label}={val_label}")
    return ", ".join(parts)


st.title("Latin Morphology Explorer")

st.markdown(
    """
Analyze the **morphological features** of each word in a Latin text —
lemma, part of speech, case, number, gender, tense, mood, and more.

Select a word from the table to see its full analysis.
"""
)

model_selectbox = st.sidebar.selectbox(
    "Choose model:", ("la_core_web_lg", "la_core_web_md", "la_core_web_sm")
)


@st.cache_resource
def load_model(model_name):
    return spacy.load(model_name)


nlp = load_model(model_selectbox)

text = st.text_area(
    "Enter Latin text to analyze (max ~200 tokens):", value=default_text, height=200
)

if st.button("Analyze Morphology"):
    tokens = text.split()
    if len(tokens) > 200:
        st.warning("Text trimmed to ~200 tokens.")
        text = " ".join(tokens[:200])

    doc = nlp(text)

    rows = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        rows.append(
            {
                "Token": token.text,
                "Lemma": token.lemma_,
                "POS": POS_LABELS.get(token.pos_, token.pos_),
                "Features": format_morph_readable(token.morph),
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**Select a word for detailed analysis:**")

    # Build options with index to disambiguate duplicate tokens
    content_tokens = [t for t in doc if not t.is_punct and not t.is_space]
    word_options = [f"{t.text} ({i + 1})" for i, t in enumerate(content_tokens)]
    selected_idx = st.selectbox(
        "Word:", range(len(word_options)), format_func=lambda i: word_options[i],
        key="morph_word",
    )

    if selected_idx is not None:
        token = content_tokens[selected_idx]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### {token.text}")
            st.markdown(f"**Lemma:** {token.lemma_}")
            st.markdown(
                f"**Part of Speech:** {POS_LABELS.get(token.pos_, token.pos_)}"
            )
            st.markdown(f"**Fine-grained Tag:** {token.tag_}")
        with col2:
            morph_dict = token.morph.to_dict()
            if morph_dict:
                st.markdown("**Morphological Features:**")
                for feat, val in morph_dict.items():
                    feat_label = FEATURE_LABELS.get(feat, feat)
                    val_label = _get_val_label(feat, val)
                    st.markdown(f"- {feat_label}: **{val_label}**")
            else:
                st.info("No morphological features available for this token.")
