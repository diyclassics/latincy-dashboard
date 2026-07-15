"""Shared helpers for the latincy-lexicon demo (page 11)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st


DEFAULT_SENTENCE = "Poeta bonus carmina pulchra scribit."


@st.cache_resource(show_spinner="Building Whitaker's Words data (first load only)…")
def build_lexicon_artifacts() -> tuple[Path, Path]:
    """Build lexicon.json + analyzer.json once per session.

    Artifacts are cached under a temp directory. First call takes
    ~5–10s; subsequent calls short-circuit via Streamlit's
    cache_resource and are instant.
    """
    from latincy_lexicon.build import build

    output_dir = Path(tempfile.gettempdir()) / "latincy-dashboard-lexicon"
    output_dir.mkdir(parents=True, exist_ok=True)

    lexicon_path = output_dir / "lexicon.json"
    analyzer_path = output_dir / "analyzer.json"

    if not (lexicon_path.exists() and analyzer_path.exists()):
        build(output_dir=output_dir)

    return lexicon_path, analyzer_path


@st.cache_resource(show_spinner="Loading Whitaker's Words components (first load only)…")
def load_lexicon_components():
    """A weightless ``spacy.blank("la")`` carrying whitakers_words + paradigm_generator.

    The heavy ``lg`` model is loaded once via ``model_helpers.load_model`` and
    shared across the whole dashboard. These two components are stateless — they
    only read ``token.text``/``lemma_``/``pos_`` and set token extensions — so
    they can be applied to the shared model's docs post-hoc (see
    ``annotate_lexicon``) instead of loading a second full model just to hold
    them. The blank pipeline carries no weights, so this is ~KB, not ~650 MB.
    """
    import spacy

    lexicon_path, analyzer_path = build_lexicon_artifacts()
    blank = spacy.blank("la")
    blank.add_pipe(
        "whitakers_words",
        config={
            "lexicon_path": str(lexicon_path),
            "analyzer_path": str(analyzer_path),
        },
        last=True,
    )
    blank.add_pipe(
        "paradigm_generator",
        config={"analyzer_path": str(analyzer_path)},
        last=True,
    )
    return blank


def annotate_lexicon(doc):
    """Enrich a doc from the shared lg model with Whitaker's Words data in place.

    Applies whitakers_words + paradigm_generator so ``token._.lexicon`` /
    ``._.gloss`` / ``._.ww`` / ``._.paradigm`` are populated — no second model
    load. Returns the same doc.
    """
    blank = load_lexicon_components()
    for name in ("whitakers_words", "paradigm_generator"):
        doc = blank.get_pipe(name)(doc)
    return doc


def sentence_picker(key: str) -> str:
    """Render a free-text area prefilled with a simple example sentence."""
    return st.text_area(
        "Latin text:",
        value=DEFAULT_SENTENCE,
        height=100,
        key=f"text_{key}",
    )
