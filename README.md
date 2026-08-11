# LatinCy Dashboard

FastAPI dashboard for exploring features of the [LatinCy](https://huggingface.co/latincy) `la_core_web_lg` model. A single shared model instance backs every demo.

Live at [dashboard.exploratoryphilology.org](https://dashboard.exploratoryphilology.org)

## Demos

- **Parsing** — UD/CoNLL-U column output (form, lemma, UPOS, XPOS, feats, head, deprel), with TSV/CSV/JSON/CoNLL-U export
- **Sentence Segmentation** — Segment paragraphs into sentences with text export
- **Named Entity Recognition** — Highlight people, places, and groups (PER, LOC, NORP)
- **Dependency Trees** — Visualize grammatical structure with displaCy
- **Custom Labels** — Visualize tokens covered by the [DCC Core Latin Vocabulary](https://dcc.dickinson.edu/latin-core-list1)
- **Vocab Builder** — Glossed, citation-form vocabulary lists via `latincy-vocab` + `latincy-lexicon`
- **U/V Normalizer** — Rule-based classical U/V spelling normalization

## Setup

```bash
pip install -r parser_app/requirements.txt
uvicorn parser_app.main:app --port 8501
```

Written by [diyclassics](https://github.com/diyclassics). April 2023, updated August 2026.
