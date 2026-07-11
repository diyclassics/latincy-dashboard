"""Shared spaCy model loading for the dashboard.

Read-only demos import ``load_model()`` so they all share ONE cached model
instance per name across the whole process, instead of each page caching its
own copy (which multiplied a ~500 MB model across pages in RAM).
``@st.cache_resource`` already guarantees the model loads once per process and
is reused across sessions/reruns — no manual threading needed.

Demos that MUTATE their pipeline (e.g. the custom-label demo, which adds a
component with ``add_pipe``) must cache their OWN instance instead — mutating
this shared one would leak the change into every other demo.
"""

import spacy
import streamlit as st

DEFAULT_MODEL = "la_core_web_lg"


@st.cache_resource(show_spinner="Loading LatinCy model…")
def load_model(model_name=DEFAULT_MODEL):
    """Return the shared, cached spaCy model for ``model_name``.

    Read-only use only — do not ``add_pipe``/``disable`` on the result, since
    every read-only demo shares this one instance.
    """
    nlp = spacy.load(model_name)
    if "trf_vectors" in nlp.pipe_names:  # heavy, unused in these demos
        nlp.disable_pipe("trf_vectors")
    return nlp
