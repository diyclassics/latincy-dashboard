import streamlit as st
import spacy
import pandas as pd
import datetime
import gc
import subprocess
import json
from spacy.util import registry

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
    for token in doc[:500]:
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
first_model = "la_core_web_lg"
first_model_version = spacy.info(first_model)["version"]


# Function to unload the current model
@st.cache_resource
def load_model(model_name):
    gc.collect()  # Attempt to free memory
    nlp = spacy.load(model_name)
    try:
        if "trf_vectors" in nlp.pipe_names:  # Disable trf_vectors if it exists
            nlp.disable_pipe("trf_vectors")
    except Exception as e:
        st.warning(f"Failed to disable 'trf_vectors': {e}")
    return nlp


# Function to load model in a subprocess to avoid conflicts
def load_model_in_subprocess(model_name):
    script = (
        "import spacy, json; "
        'nlp = spacy.load("' + model_name + '"); '
        "print(json.dumps(nlp.meta))"
    )
    result = subprocess.run(["python", "-c", script], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to load model {model_name}: {result.stderr}")
    return json.loads(result.stdout)


model_name = "la_core_web_lg"  # Hardcoded to use only the lg model
nlp = spacy.load(model_name)  # Directly load the lg model

st.write(f"Loaded model: {model_name} (v{spacy.info(model_name)['version']})")

df = None

text = st.text_area(
    "Enter some text to analyze (max 500 tokens)", value=default_text, height=200
)
if st.button("Analyze"):
    df = analyze_text(text)
    st.text(f"Analyzed {len(df)} tokens with {model_name} model.")
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
