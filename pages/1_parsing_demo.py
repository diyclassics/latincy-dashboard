import streamlit as st
import spacy
import pandas as pd
import datetime

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
    for token in doc[:100]:
        rows.append(
            (
                token.text,
                token.norm_,
                token.lower_,
                token.lemma_,
                token.pos_,
                token.tag_,
                token.dep_,
                format_morph(token.morph),
                token.ent_type_,
            )
        )
    df = pd.DataFrame(
        rows,
        columns=[
            "text",
            "norm",
            "lower",
            "lemma",
            "pos",
            "tag",
            "dep",
            "morph",
            "ent_type",
        ],
    )
    return df


st.title("LatinCy Text Analyzer")

# Using object notation
model_selectbox = st.sidebar.selectbox(
    "Choose model:",
    ("la_core_web_lg", "la_core_web_md", "la_core_web_sm")
)

nlp = spacy.load(model_selectbox)

df = None

text = st.text_area(
    "Enter some text to analyze (max 100 tokens)", value=default_text, height=200
)
if st.button("Analyze"):
    df = analyze_text(text)
    st.text(f"Analyzed {len(df)} tokens with {model_selectbox} model.")
    st.dataframe(df, width=1000)

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
