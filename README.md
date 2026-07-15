---
title: Latincy Dashboard
emoji: 🌍
colorFrom: yellow
colorTo: blue
sdk: streamlit
sdk_version: 1.45.1
app_file: app.py
pinned: false
license: mit
python_version: "3.12"
---

# LatinCy Dashboard

Streamlit dashboard for exploring features of the [LatinCy](https://huggingface.co/latincy) `la_core_web_lg` model (v3.9.6). A single shared model instance backs every demo.

View dashboard [here](https://latincy.streamlit.app/)

## Demos

- **Parsing** — UD/CoNLL-U column output (form, lemma, UPOS, XPOS, feats, head, deprel) with TSV export
- **Custom Labels** — Visualize tokens covered by the [DCC Core Latin Vocabulary](https://dcc.dickinson.edu/latin-core-list1)
- **Sentence Segmentation** — Segment paragraphs into sentences with text export
- **Named Entity Recognition** — Highlight people, places, and groups (PER, LOC, NORP)
- **Dependency Trees** — Visualize grammatical structure with displaCy
- **Word Similarity** — Explore floret subword vector similarity between Latin words
- **Morphology** — Analyze lemma, POS, case, gender, tense, mood, and more per token

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Written by [diyclassics](https://github.com/diyclassics). April 2023, updated February 2026.
