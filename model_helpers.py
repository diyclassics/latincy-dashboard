"""Shared spaCy model loading + background preload for the dashboard.

Read-only demos import ``load_model()`` from here so they all share ONE model
instance per name across the whole process, instead of each page caching its
own copy (which multiplied a ~500 MB model across pages in RAM).
``start_preload()`` warms the default model in a background thread from the
home page, so navigating into a demo finds the model already resident.

Demos that MUTATE their pipeline (e.g. the custom-label demo, which adds a
component with ``add_pipe``) must NOT use this shared instance — they should
cache their own copy — because mutating the shared model would leak the change
into every other demo.
"""

import threading

import spacy
import streamlit as st

DEFAULT_MODEL = "la_core_web_lg"

# Single source of truth for loaded models, shared between the background
# preloader and the synchronous loader. Guarded by _lock so a preload and a
# concurrent load_model() can never start two loads of the same model.
_models = {}
_lock = threading.Lock()


def _load(model_name):
    """Load a model with no Streamlit (st.*) calls — safe off the main thread."""
    nlp = spacy.load(model_name)
    if "trf_vectors" in nlp.pipe_names:  # heavy, unused in these demos
        nlp.disable_pipe("trf_vectors")
    return nlp


def _get(model_name):
    with _lock:
        if model_name not in _models:
            _models[model_name] = _load(model_name)
        return _models[model_name]


def start_preload(model_name=DEFAULT_MODEL):
    """Kick off a non-blocking background load of the default model.

    Called from the home page so the model warms while the user reads the demo
    list. Cheap to call repeatedly: once loaded, the worker returns at once.
    """
    if model_name in _models:
        return
    threading.Thread(target=_get, args=(model_name,), daemon=True).start()


@st.cache_resource(show_spinner="Loading LatinCy model…")
def load_model(model_name=DEFAULT_MODEL):
    """Return the shared, cached spaCy model for ``model_name``.

    Reuses the background-preloaded instance when ready; otherwise loads it once
    synchronously. Read-only use only — do not ``add_pipe``/``disable`` on the
    result, since every demo shares this instance.
    """
    return _get(model_name)
