"""LatinCy demos — a plain FastAPI site (no Streamlit).

Streamlit's runtime segfaults on inference on this host; plain spaCy is stable.
The lg model is loaded ONCE at startup and shared by every model-based demo
(parser/senter/NER/dependency/custom-label) — inference is serialized behind a
lock. U/V spelling is rule-based (no model). Left nav; parser is the landing
page. LatinCy house style (Lexend, #1d70c7), matching latincy-lexicon-site.
"""

import csv
import html
import io
import json
import pathlib
import tempfile
import threading
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from urllib.parse import quote

import spacy
from spacy import displacy
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, Response

from vocabbuilder.core.config import PipelineConfig
from vocabbuilder.core.models import VocabList
from vocabbuilder.processors.vocab_core import build_vocab_list
from vocabbuilder.utils.normalization import to_u_form

from dcc_helpers import DCC_CORE_LEMMAS, is_dcc_core

MODEL = "la_core_web_lg"
MODEL_VERSION = "?"  # set from the model's meta at startup
HF_URL = "https://huggingface.co/latincy/la_core_web_lg"
# Guardrail cap for EVERY text box: these are quick demos, not a batch service —
# for whole texts, install LatinCy. Counted with the model tokenizer; the later
# enclitic_splitter may add a few tokens, so the final count can run a hair over
# 150 (immaterial for a demo).
MAX_TOKENS = 150


def _pkg_ver(pkg):
    try:
        return _pkg_version(pkg)
    except PackageNotFoundError:
        return "?"


# Tool versions for the /vocab attribution line (read from installed metadata).
LEXICON_VERSION = _pkg_ver("latincy-lexicon")
VOCAB_VERSION = _pkg_ver("latincy-vocab")

# Routes that use the lg model (so we show its attribution + metrics there).
MODEL_ROUTES = {"/", "/senter", "/ner", "/dependency", "/custom-label", "/vocab"}
METRIC_LABELS = [
    ("pos_acc", "POS (UPOS)"),
    ("tag_acc", "Tag (XPOS)"),
    ("morph_acc", "Morphology"),
    ("lemma_acc", "Lemma"),
    ("dep_uas", "Dependency UAS"),
    ("dep_las", "Dependency LAS"),
    ("sents_f", "Sentence segmentation (F1)"),
    ("ents_f", "NER (F1)"),
]

# Each sample is >= 3 sentences (the demos are more useful multi-sentence).
SAMPLE_PASSAGES = {
    "Seneca, Ep. 1.1": "Ita fac, mi Lucili; vindica te tibi, et tempus quod adhuc aut auferebatur aut subripiebatur aut excidebat collige et serva. Persuade tibi hoc sic esse ut scribo: quaedam tempora eripiuntur nobis, quaedam subducuntur, quaedam effluunt. Turpissima tamen est iactura quae per neglegentiam fit.",
    "Vergil, Aen. 1.1": "Arma virumque cano, Troiae qui primus ab oris Italiam fato profugus Laviniaque venit litora, multum ille et terris iactatus et alto vi superum saevae memorem Iunonis ob iram, multa quoque et bello passus, dum conderet urbem inferretque deos Latio, genus unde Latinum Albanique patres atque altae moenia Romae. Musa, mihi causas memora, quo numine laeso quidve dolens regina deum tot volvere casus insignem pietate virum, tot adire labores impulerit. Tantaene animis caelestibus irae?",
    "Caesar, B.G. 1.1": "Gallia est omnis divisa in partes tres, quarum unam incolunt Belgae, aliam Aquitani, tertiam qui ipsorum lingua Celtae, nostra Galli appellantur. Hi omnes lingua, institutis, legibus inter se differunt. Gallos ab Aquitanis Garumna flumen, a Belgis Matrona et Sequana dividit.",
    "Ritchie, Perseus": "Haec narrantur a poetis de Perseo. Perseus filius erat Iovis, maximi deorum; avus eius Acrisius appellabatur. Acrisius volebat Perseum nepotem suum necare.",
    "Cicero, Cat. 1.1": "Quo usque tandem abutere, Catilina, patientia nostra? quam diu etiam furor iste tuus nos eludet? quem ad finem sese effrenata iactabit audacia?",
}
DEFAULT_TEXT = SAMPLE_PASSAGES["Seneca, Ep. 1.1"]
# The U/V demo strips ANY input to u-only, then restores v — so a normal v-form
# passage (Aeneid 1.1-11) works as the default and as its own gold reference.
UV_DEFAULT = "Arma virumque cano, Troiae qui primus ab oris Italiam fato profugus Laviniaque venit litora, multum ille et terris iactatus et alto vi superum saevae memorem Iunonis ob iram. Musa, mihi causas memora, quo numine laeso quidve dolens regina deum tot volvere casus insignem pietate virum, tot adire labores impulerit."
COLUMNS = ["sent_id", "token_id", "form", "lemma", "upos", "xpos", "feats", "head", "deprel", "ent_type"]

# (path, label) — order = left-nav order; first entry is the landing page.
NAV = [
    ("/", "Parser"),
    ("/senter", "Sentences"),
    ("/ner", "Entities"),
    ("/dependency", "Dependencies"),
    ("/custom-label", "DCC Core"),
    ("/vocab", "Vocab list"),
    ("/uv", "U/V spelling"),
]

# Vocab demo (latincy-vocab). Perseus opener — a teaching text — is the default.
VOCAB_DEFAULT = SAMPLE_PASSAGES["Ritchie, Perseus"]
VOCAB_SORTS = {"alpha": "Alphabetical", "first": "First occurrence", "freq": "Frequency"}
VOCAB_DCC = {"all": "All words", "new": "New (non-core)", "core": "DCC core only"}
# fmt -> (media type, VocabList method, download filename, button label)
VOCAB_DOWNLOADS = {
    "md": ("text/markdown", "to_markdown", "latincy-vocab.md", "Markdown"),
    "json": ("application/json", "to_json", "latincy-vocab.json", "JSON"),
}

_nlp = None
_lock = threading.Lock()
_uv = None
_ww = None  # blank pipe carrying whitakers_words; built lazily on first /vocab
_ww_lock = threading.Lock()
_vocab_config = None


@asynccontextmanager
async def lifespan(app):
    global _nlp, _uv, _ww, _vocab_config, MODEL_VERSION
    _nlp = spacy.load(MODEL)
    if "trf_vectors" in _nlp.pipe_names:
        _nlp.disable_pipe("trf_vectors")
    MODEL_VERSION = _nlp.meta.get("version", "?")
    from latincy_preprocess.uv import UVNormalizerRules
    _uv = UVNormalizerRules()
    # NB: the vocab demo's Whitaker's Words lexicon (~300 MB resident) is NOT
    # loaded here — it's built lazily on the first /vocab request (see
    # _get_vocab_pipe). On this 1.9 GB box, front-loading it at startup would
    # make every parser/senter/NER/dependency/U-V visitor carry 300 MB they
    # never use; lazy keeps the common path lean and only vocab users pay it.
    yield


def metrics_html():
    """Collapsible model-evaluation metrics, pulled from the model's meta.json."""
    perf = (_nlp.meta or {}).get("performance", {})
    rows = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{perf[key] * 100:.2f}%</td></tr>"
        for key, label in METRIC_LABELS if isinstance(perf.get(key), (int, float))
    )
    if not rows:
        return ""
    return (f'<details class="metrics"><summary>{MODEL} v{MODEL_VERSION} — evaluation metrics</summary>'
            f'<table class="metrics"><thead><tr><th>Component</th><th>Score</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>'
            f'<p class="uonly">From the model’s <code>meta.json</code> — held-out UD test set (NER on dev).</p>'
            f'</details>')


app = FastAPI(lifespan=lifespan, title="LatinCy demos")


def nlp(text):
    """Serialized inference on the shared model (cheap on 1 vCPU; avoids races)."""
    with _lock:
        return _nlp(text)


# --------------------------------------------------------------------------- #
# layout                                                                       #
# --------------------------------------------------------------------------- #
def _cap(text):
    """Truncate to MAX_TOKENS tokens with the model tokenizer — the single cap for
    all boxes. Tokenizer-only (no lock needed): cheap, and being off by a few
    enclitics vs the final parse is immaterial for a demo."""
    doc = _nlp.tokenizer(text)
    return doc[:MAX_TOKENS].text if len(doc) > MAX_TOKENS else text.strip()


def _cap_hint(text):
    """Token-limit note under every text box; flags truncation with an install nudge."""
    if len(_nlp.tokenizer(text)) > MAX_TOKENS:
        return (f'<p class="uonly">Limited to {MAX_TOKENS} tokens — longer input is truncated. '
                f'For whole texts, install <a href="https://huggingface.co/latincy">LatinCy</a>.</p>')
    return f'<p class="uonly">Up to {MAX_TOKENS} tokens.</p>'


def _samples_pills():
    btns = "".join(
        f'<button type="button" class="pill" data-text="{html.escape(v)}">{html.escape(k)}</button>'
        for k, v in SAMPLE_PASSAGES.items()
    )
    return f'<div class="pills" role="group" aria-label="Sample passages">{btns}</div>'


def input_form(action, text, *, samples=True, label="Enter Latin text", button="Analyze", rows=6):
    pills = _samples_pills() if samples else ""
    return f"""
    <form method="get" action="{action}">
      <label class="fieldlabel" for="text">{html.escape(label)}</label>
      <textarea name="text" id="text" rows="{rows}">{html.escape(text)}</textarea>
      {_cap_hint(text)}
      {pills}
      <button class="go" type="submit">{html.escape(button)}</button>
    </form>"""


def layout(active, intro, body):
    nav = "".join(
        f'<a href="{p}"{" class=active aria-current=page" if p == active else ""}>{html.escape(lbl)}</a>'
        for p, lbl in NAV
    )
    page_label = html.escape(dict(NAV)[active])
    if active in MODEL_ROUTES:
        model_line = f'<p class="modelline">Model: <a href="{HF_URL}">{MODEL}</a> v{MODEL_VERSION}</p>'
        if active == "/vocab":
            model_line += (
                '<p class="modelline">Lists: '
                f'<a href="https://github.com/latincy/latincy-vocab">latincy-vocab</a> v{VOCAB_VERSION} · '
                f'glosses <a href="https://github.com/latincy/latincy-lexicon">latincy-lexicon</a> '
                f'v{LEXICON_VERSION} (Whitaker’s Words)</p>'
            )
        metrics = metrics_html()
    else:
        model_line = '<p class="modelline">Rule-based normalization (<code>latincy-preprocess</code>) — no ML model.</p>'
        metrics = ""
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="LatinCy demos — {page_label}: Latin NLP with the {MODEL} model.">
<title>{page_label} — LatinCy demos</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Lexend:wght@400;500;700&display=swap">
<style>
  :root {{ font-family:"Lexend",-apple-system,system-ui,sans-serif; --accent:#1d70c7; --accent-text:#17609f; }}
  * {{ box-sizing:border-box; }}
  .visually-hidden {{ position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; border:0; }}
  .fieldlabel {{ display:block; font-size:.9rem; font-weight:500; color:#333; margin:0 0 .35rem; }}
  body {{ margin:0; line-height:1.5; color:#1a1a1a; display:flex; min-height:100vh; }}
  .skip-link {{ position:absolute; top:-40px; left:.5rem; background:var(--accent); color:#fff; padding:.4rem .75rem; border-radius:0 0 4px 4px; z-index:100; text-decoration:none; }}
  .skip-link:focus {{ top:0; }}
  :focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
  aside {{ width:210px; flex:0 0 210px; border-right:1px solid #e2e2e2; padding:1.4rem 1rem; position:sticky; top:0; height:100vh; overflow:auto; }}
  .brand {{ display:block; text-decoration:none; color:var(--accent); font-size:1.25rem; font-weight:700; letter-spacing:-0.01em; margin-bottom:1.2rem; }}
  .brand .suffix {{ color:#000; }}
  aside nav {{ display:flex; flex-direction:column; gap:.15rem; }}
  aside nav a {{ text-decoration:none; color:#333; padding:.4rem .6rem; border-radius:6px; font-size:.95rem; }}
  aside nav a:hover {{ background:#f2f6fb; color:var(--accent-text); }}
  aside nav a.active {{ background:var(--accent); color:#fff; }}
  main {{ flex:1; padding:2rem 2.2rem 4rem; max-width:60rem; }}
  h1 {{ font-size:1.6rem; margin:0 0 .3rem; }}
  .lede {{ font-size:1.02rem; color:#444; margin:0 0 1.3rem; }}
  textarea {{ width:100%; padding:.75rem 1rem; font:inherit; font-size:1.03rem; border:1px solid #767676; border-radius:4px; resize:vertical; }}
  .pills {{ margin:.7rem 0 .2rem; display:flex; flex-wrap:wrap; gap:.4rem; }}
  .pill {{ font:inherit; border:1px solid #767676; background:#fff; border-radius:999px; padding:.3rem .85rem; font-size:.83rem; color:#333; cursor:pointer; }}
  .pill:hover {{ border-color:var(--accent); color:var(--accent-text); }}
  .go {{ margin-top:.9rem; background:var(--accent); color:#fff; border:none; border-radius:4px; padding:.6rem 1.4rem; font:inherit; font-size:1.02rem; cursor:pointer; }}
  .go:hover {{ background:var(--accent-text); }}
  .summary {{ margin:1.6rem 0 .6rem; font-size:1.02rem; font-weight:600; }}
  .summary:focus {{ outline:none; }}
  .tablewrap {{ overflow:auto; max-height:62vh; border:1px solid #e2e2e2; border-radius:6px; }}
  table.parse {{ border-collapse:collapse; width:auto; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.82rem; }}
  table.parse th, table.parse td {{ text-align:left; padding:.3rem .7rem; white-space:nowrap; border-bottom:1px solid #eee; }}
  table.parse th {{ position:sticky; top:0; background:#fafafa; font-weight:600; }}
  ol.sents {{ font-size:1.05rem; }} ol.sents li {{ margin:.35rem 0; }}
  .vcontrols {{ display:flex; flex-wrap:wrap; gap:1.2rem; margin:.9rem 0 .2rem; }}
  .vselect {{ font-size:.9rem; color:#333; }}
  .vselect select {{ font:inherit; font-size:.9rem; padding:.25rem .4rem; border:1px solid #767676; border-radius:4px; margin-left:.2rem; }}
  ul.vocab {{ list-style:none; margin:1rem 0 0; padding:0; font-size:1.02rem; columns:2; column-gap:2.4rem; }}
  ul.vocab li {{ margin:0 0 .45rem; line-height:1.45; break-inside:avoid; }}
  .vhead {{ font-weight:700; }}
  .vpos {{ color:#666; font-style:italic; font-size:.9em; }}
  .vfreq {{ color:#888; font-size:.85em; }}
  .coredot {{ display:inline-block; width:.45rem; height:.45rem; border-radius:50%; background:var(--accent); vertical-align:.12em; margin-left:.4em; }}
  @media (max-width:720px) {{ ul.vocab {{ columns:1; }} }}
  .render {{ overflow:auto; border:1px solid #e2e2e2; border-radius:6px; padding:.6rem 1rem; margin-top:.6rem; }}
  mark.core {{ background:#d6ebfb; padding:0 .05em; border-radius:3px; }}
  .textout {{ font-size:1.1rem; line-height:1.7; border:1px solid #e2e2e2; border-radius:6px; padding:.8rem 1rem; margin-top:.6rem; }}
  ins {{ background:#d8f5d8; text-decoration:none; }} del {{ background:#f7d7d7; }}
  .modes {{ border:none; margin:.7rem 0 0; padding:0; display:flex; flex-direction:column; gap:.3rem; }}
  .modes label {{ font-size:.95rem; }}
  .uonly {{ color:#666; font-size:.9rem; margin:.4rem 0; word-break:break-word; }}
  .modelline {{ color:#555; font-size:.9rem; margin:-.7rem 0 1.3rem; }}
  .modelline a {{ color:var(--accent-text); }}
  details.metrics {{ margin:2.2rem 0 0; }}
  details.metrics summary {{ cursor:pointer; font-weight:500; color:var(--accent-text); }}
  table.metrics {{ border-collapse:collapse; margin:.6rem 0; font-size:.9rem; }}
  table.metrics th, table.metrics td {{ text-align:left; padding:.3rem 1.2rem .3rem 0; border-bottom:1px solid #eee; }}
  details.tsv {{ margin-top:.8rem; }} details.tsv textarea {{ width:100%; font-family:ui-monospace,monospace; font-size:.75rem; }}
  .copytsv {{ display:inline-block; margin:.5rem 0 .4rem; font:inherit; font-size:.85rem; color:#333; background:#fff; border:1px solid #767676; border-radius:4px; padding:.35rem .8rem; cursor:pointer; }}
  .copytsv:hover {{ border-color:var(--accent); color:var(--accent-text); }}
  .downloads {{ margin:.2rem 0 1rem; display:flex; align-items:center; flex-wrap:wrap; gap:.4rem; }}
  .dllabel {{ font-size:.88rem; color:#555; margin-right:.1rem; }}
  .downloads .clear {{ margin-left:0; }}
  .clear {{ display:inline-block; margin-left:.9rem; font:inherit; font-size:.88rem; color:#555; border:1px solid #767676; border-radius:4px; padding:.35rem .7rem; text-decoration:none; vertical-align:middle; }}
  .clear:hover {{ border-color:var(--accent); color:var(--accent-text); }}
  code {{ background:#f0f1f3; padding:.05rem .35rem; border-radius:4px; }}
  footer {{ margin-top:3rem; color:#666; border-top:1px solid #eee; padding-top:1rem; font-size:.9rem; }}
  footer a {{ color:var(--accent-text); }}
  @media (max-width:720px) {{ body {{ flex-direction:column; }} aside {{ width:auto; height:auto; position:static; border-right:none; border-bottom:1px solid #e2e2e2; }} aside nav {{ flex-flow:row wrap; }} }}
</style></head>
<body>
  <a class="skip-link" href="#main-content">Skip to main content</a>
  <aside>
    <a href="/" class="brand">LatinCy <span class="suffix">demos</span></a>
    <nav aria-label="Demos">{nav}</nav>
  </aside>
  <main id="main-content">
    <h1>{page_label}</h1>
    <p class="lede">{intro}</p>
    {model_line}
    {body}
    {metrics}
    <footer>Built on the <a href="https://huggingface.co/latincy">LatinCy</a> <code>{MODEL}</code> model.
    Written by <a href="https://github.com/diyclassics">P.&nbsp;J.&nbsp;Burns</a> + Claude Opus&nbsp;4.8.</footer>
  </main>
  <script>
    document.querySelectorAll('.pill').forEach(function (b) {{
      b.addEventListener('click', function () {{
        var t = document.getElementById('text');
        t.value = b.getAttribute('data-text'); t.focus();
      }});
    }});
    document.querySelectorAll('.copytsv').forEach(function (b) {{
      b.addEventListener('click', function () {{
        var ta = document.getElementById(b.getAttribute('data-target'));
        navigator.clipboard.writeText(ta.value).then(function () {{
          var old = b.textContent; b.textContent = 'Copied!';
          setTimeout(function () {{ b.textContent = old; }}, 1500);
        }});
      }});
    }});
    var r = document.getElementById('results');
    if (r) {{ r.focus(); }}
  </script>
</body></html>"""


# --------------------------------------------------------------------------- #
# demos                                                                        #
# --------------------------------------------------------------------------- #
def _format_morph(morph):
    d = morph.to_dict()
    return ", ".join(f"{k}={v}" for k, v in d.items()) if d else ""


def parse_rows(text):
    """The parse as a list of COLUMNS-ordered rows (shared by the table + CSV)."""
    doc = nlp(_cap(text))
    rows = []
    for si, sent in enumerate(doc.sents):
        start = sent.start
        for ti, tok in enumerate(sent):
            head = 0 if tok.head == tok else tok.head.i - start + 1
            rows.append([f"s{si + 1}", ti + 1, tok.text, tok.lemma_, tok.pos_, tok.tag_,
                         _format_morph(tok.morph), head, tok.dep_, tok.ent_type_])
    return rows


def rows_to_tsv(rows):
    """CoNLL-U-ordered TSV — identical bytes for the copy box and the download."""
    return "\t".join(COLUMNS) + "\n" + "\n".join("\t".join(str(v) for v in r) for r in rows) + "\n"


def rows_to_csv(rows):
    """CSV with proper quoting (the feats field carries commas)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(COLUMNS)
    writer.writerows(rows)
    return buf.getvalue()


def rows_to_json(rows):
    return json.dumps([dict(zip(COLUMNS, r)) for r in rows], ensure_ascii=False, indent=2) + "\n"


def rows_to_conllu(rows):
    """CoNLL-U: 10 columns per token, blank-line-separated sentences.

    Our sent-local token_id/head map straight onto CoNLL-U ID/HEAD; feats become
    pipe-delimited; ent_type rides in MISC as NER=<type>. DEPS is left unset (_).
    """
    lines, cur = [], None
    for sid, tid, form, lemma, upos, xpos, feats, head, deprel, ent in rows:
        if sid != cur:
            if cur is not None:
                lines.append("")
            lines.append(f"# sent_id = {sid}")
            cur = sid
        feats_c = feats.replace(", ", "|") if feats else "_"
        misc = f"NER={ent}" if ent else "_"
        lines.append("\t".join([str(tid), form or "_", lemma or "_", upos or "_",
                                xpos or "_", feats_c, str(head), deprel or "_", "_", misc]))
    return "\n".join(lines) + "\n"


# fmt -> (media type, serializer, button label)
DOWNLOAD_FORMATS = {
    "tsv": ("text/tab-separated-values", rows_to_tsv, "TSV"),
    "csv": ("text/csv", rows_to_csv, "CSV"),
    "json": ("application/json", rows_to_json, "JSON"),
    "conllu": ("text/plain", rows_to_conllu, "CoNLL-U"),
}


def parser_result(text):
    rows = parse_rows(text)
    if not rows:
        return ""
    thead = "".join(f"<th>{html.escape(c)}</th>" for c in COLUMNS)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>" for r in rows)
    tsv = rows_to_tsv(rows)
    n_sents = len({r[0] for r in rows})
    q = quote(text)
    dl_buttons = "".join(
        f'<a href="/parse.{fmt}?text={q}" class="clear" download>{lbl}</a>'
        for fmt, (_media, _fn, lbl) in DOWNLOAD_FORMATS.items()
    )
    downloads = f'<div class="downloads"><span class="dllabel">Download:</span>{dl_buttons}</div>'
    return (f'<h2 class="summary" id="results" tabindex="-1">Analyzed {len(rows)} tokens in {n_sents} sentence(s).'
            f'<a href="/" class="clear">Clear</a></h2>'
            f'{downloads}'
            f'<div class="tablewrap"><table class="parse"><thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table></div>'
            f'<details class="tsv"><summary>Copy as TSV (CoNLL-U order)</summary>'
            f'<button type="button" class="copytsv" data-target="tsvbox">Copy to clipboard</button>'
            f'<textarea id="tsvbox" readonly rows="6">{html.escape(tsv)}</textarea></details>')


def senter_result(text):
    doc = nlp(_cap(text))
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    if not sents:
        return ""
    items = "".join(f"<li>{html.escape(s)}</li>" for s in sents)
    return f'<h2 class="summary" id="results" tabindex="-1">{len(sents)} sentence(s).</h2><ol class="sents">{items}</ol>'


def ner_result(text):
    doc = nlp(_cap(text))
    n = len(doc.ents)
    svg = displacy.render(doc, style="ent", page=False, minify=True)
    return (f'<h2 class="summary" id="results" tabindex="-1">{n} entit{"y" if n == 1 else "ies"} found '
            f'(PER, LOC, NORP).</h2><div class="render">{svg}</div>')


def dep_result(text):
    doc = nlp(_cap(text))
    parts = [displacy.render(sent.as_doc(), style="dep", page=False, minify=True,
                             options={"compact": True, "distance": 90})
             for sent in doc.sents]
    if not parts:
        return ""
    note = ('<p class="uonly">Dependency diagrams are graphics; screen-reader users can read the same '
            'head/relation data as a text table in the <a href="/">Parser</a> demo.</p>')
    render = ('<div class="render" role="img" '
              'aria-label="Dependency parse diagrams. See the Parser demo for the same data as a text table.">'
              + "".join(parts) + "</div>")
    return f'<h2 class="summary" id="results" tabindex="-1">{len(parts)} sentence(s).</h2>' + note + render


def customlabel_result(text):
    normed = text.replace("v", "u").replace("V", "U").lower()
    doc = nlp(_cap(normed))
    out, total, core = [], 0, 0
    for tok in doc:
        piece = html.escape(tok.text)
        if not tok.is_punct and not tok.is_space:
            total += 1
            if tok.lemma_ in DCC_CORE_LEMMAS:
                core += 1
                piece = f'<mark class="core">{piece}<span class="visually-hidden"> (DCC core)</span></mark>'
        out.append(piece + html.escape(tok.whitespace_))
    if not total:
        return ""
    pct = round(core / total * 100, 1)
    return (f'<h2 class="summary" id="results" tabindex="-1">{core} of {total} tokens ({pct}%) are in the '
            f'<a href="https://dcc.dickinson.edu/vocab/core-vocabulary">DCC Core</a> vocabulary '
            f'(matched on lemma, u-form).</h2><div class="textout">{"".join(out)}</div>')


def _diff_html(a, b):
    """Char-level highlight of b vs a (inserts marked)."""
    import difflib
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        seg = html.escape(b[j1:j2])
        out.append(f"<ins>{seg}</ins>" if tag in ("replace", "insert") else seg if tag == "equal" else "")
    return "".join(out)


def uv_result(text):
    # Strip ANY input to u-only (as in manuscript spelling), restore v, and score
    # the restoration against the input — so it works no matter how the text is
    # spelled, which removes the "input must be u-only" confusion.
    text = _cap(text)   # same token cap as every other text box
    source = text.replace("v", "u").replace("V", "U")
    restored = _uv.normalize(source)
    needed = sum(1 for a, b in zip(text, source) if a != b)   # consonantal u's in the input
    out, correct = [], 0
    for i, ch in enumerate(restored):
        e = html.escape(ch)
        if i < len(source) and ch != source[i]:               # normalizer restored a v here
            if i < len(text) and ch == text[i]:
                correct += 1
                out.append(f"<ins>{e}</ins>")                 # matches the input
            else:
                out.append(f"<del>{e}</del>")                 # differs from the input
        else:
            out.append(e)
    if needed:
        score = f"restored {round(correct / needed * 100, 1)}% of {needed} v-spelling(s) to match your input"
    else:
        score = "your input had no consonantal u to restore"
    return (f'<h2 class="summary" id="results" tabindex="-1">Stripped to u-only, then {score}.</h2>'
            f'<p class="uonly">u-only form &rarr; <code>{html.escape(source)}</code></p>'
            f'<div class="textout">{"".join(out)}</div>')


def uv_form(text):
    return f"""
    <form method="get" action="/uv">
      <label class="fieldlabel" for="text">Latin text</label>
      <textarea name="text" id="text" rows="4">{html.escape(text)}</textarea>
      {_cap_hint(text)}
      {_samples_pills()}
      <button class="go" type="submit">Strip &amp; restore</button>
    </form>"""


def _get_vocab_pipe():
    """Lazily build the Whitaker's Words carrier on first /vocab use, then cache it.

    Deferred out of startup so the common demos never carry the ~300 MB lexicon
    data on this 1.9 GB box — only a visitor who actually builds a vocab list
    pays for it, and only once (double-checked lock; the ~5-10s build runs once).
    Artifacts are cached to tmp, so a restart rebuilds the in-RAM structures but
    reuses the on-disk JSON.
    """
    global _ww, _vocab_config
    if _ww is not None:
        return _ww
    with _ww_lock:
        if _ww is not None:
            return _ww
        from latincy_lexicon.build import build as build_lexicon
        lex_dir = pathlib.Path(tempfile.gettempdir()) / "latincy-demos-lexicon"
        lex_dir.mkdir(parents=True, exist_ok=True)
        lex_path, ana_path = lex_dir / "lexicon.json", lex_dir / "analyzer.json"
        if not (lex_path.exists() and ana_path.exists()):
            build_lexicon(output_dir=lex_dir)
        blank = spacy.blank("la")
        blank.add_pipe(
            "whitakers_words",
            config={"lexicon_path": str(lex_path), "analyzer_path": str(ana_path)},
            last=True,
        )
        _vocab_config = PipelineConfig(spacy_model=MODEL)
        _ww = blank
    return _ww


def vocab_list(text):
    """VocabList for *text*, built on the SHARED lg doc (one model, not two).

    whitakers_words is applied to the shared doc post-hoc — the same pattern as
    the lexicon demo — so ``token._.lexicon``/``._.gloss`` populate and
    ``build_vocab_list`` gets citation forms + glosses without a second model.
    Both the inference and the pipe run under the one inference lock.
    """
    ww = _get_vocab_pipe()
    capped = _cap(text)   # tokenize+truncate before taking the inference lock
    with _lock:
        doc = _nlp(capped)
        doc = ww.get_pipe("whitakers_words")(doc)
    return build_vocab_list(doc, _vocab_config)


def vocab_entries(text, sort, dcc):
    """(visible entries, grand total, DCC-core count) after sort + DCC filter.

    Grand total and core count are computed BEFORE the DCC filter so the summary
    can always report the new/core split of the full list.
    """
    vocab = vocab_list(text)
    ordered = {"freq": vocab.by_frequency, "first": vocab.by_first_occurrence}.get(
        sort, vocab.by_alpha
    )()
    entries = [e for e in ordered if e.headword and e.headword[0].isalpha()]
    grand = len(entries)
    core_total = sum(1 for e in entries if is_dcc_core(to_u_form(e.lemma)))
    if dcc == "new":
        entries = [e for e in entries if not is_dcc_core(to_u_form(e.lemma))]
    elif dcc == "core":
        entries = [e for e in entries if is_dcc_core(to_u_form(e.lemma))]
    return entries, grand, core_total


def _vocab_select(name, options, current, label):
    opts = "".join(
        f'<option value="{k}"{" selected" if k == current else ""}>{html.escape(v)}</option>'
        for k, v in options.items()
    )
    return (f'<label class="vselect">{html.escape(label)} '
            f'<select name="{name}">{opts}</select></label>')


def vocab_form(text, sort, dcc):
    return f"""
    <form method="get" action="/vocab">
      <label class="fieldlabel" for="text">Enter Latin text</label>
      <textarea name="text" id="text" rows="6">{html.escape(text)}</textarea>
      {_cap_hint(text)}
      {_samples_pills()}
      <div class="vcontrols">
        {_vocab_select("sort", VOCAB_SORTS, sort, "Sort:")}
        {_vocab_select("dcc", VOCAB_DCC, dcc, "Show:")}
      </div>
      <button class="go" type="submit">Build vocabulary list</button>
    </form>"""


def vocab_result(text, sort, dcc):
    entries, grand, core_total = vocab_entries(text, sort, dcc)
    if not entries:
        return ('<h2 class="summary" id="results" tabindex="-1">No vocabulary found.'
                '<a href="/vocab" class="clear">Clear</a></h2>')
    q = quote(text)
    dl = "".join(
        f'<a href="/vocab.{fmt}?text={q}&sort={sort}&dcc={dcc}" class="clear" download>{lbl}</a>'
        for fmt, (_media, _m, _fn, lbl) in VOCAB_DOWNLOADS.items()
    )
    downloads = f'<div class="downloads"><span class="dllabel">Download:</span>{dl}</div>'
    items = []
    any_core = False
    for e in entries:
        core = is_dcc_core(to_u_form(e.lemma))
        any_core = any_core or core
        marker = f', <span class="vpos">{html.escape(e.pos_marker)}</span>' if e.pos_marker else ""
        gloss = f', {html.escape("; ".join(e.glosses))}' if e.glosses else ""
        freq = f' <span class="vfreq">×{e.frequency}</span>' if e.frequency > 1 else ""
        dot = ('<span class="coredot" title="In the DCC Core Vocabulary"></span>'
               '<span class="visually-hidden"> (DCC core)</span>') if core else ""
        items.append(
            f'<li><b class="vhead">{html.escape(e.headword)}</b>{marker}{gloss}{freq}{dot}</li>'
        )
    breakdown = f" · {grand - core_total} new, {core_total} DCC core" if dcc == "all" else ""
    legend = ('Words in the '
              '<a href="https://dcc.dickinson.edu/vocab/core-vocabulary">DCC Core Vocabulary</a> '
              'are marked with a blue dot <span class="coredot"></span>. ') if any_core else ""
    note = ('<p class="uonly">Citation forms + glosses from Whitaker’s Words via '
            '<a href="https://github.com/latincy/latincy-lexicon"><code>latincy-lexicon</code></a>; '
            'list assembled by '
            '<a href="https://github.com/latincy/latincy-vocab"><code>latincy-vocab</code></a>. '
            f'{legend}'
            'Probabilistic output — verify before classroom use.</p>')
    return (f'<h2 class="summary" id="results" tabindex="-1">{len(entries)} entr'
            f'{"y" if len(entries) == 1 else "ies"} shown{breakdown}.'
            f'<a href="/vocab" class="clear">Clear</a></h2>'
            f'{downloads}{note}<ul class="vocab">{"".join(items)}</ul>')


# --------------------------------------------------------------------------- #
# routes                                                                       #
# --------------------------------------------------------------------------- #
def _clean(text):
    return (text or "").strip()


@app.get("/", response_class=HTMLResponse)
def parser(text: str = Query(None)):
    if text is None:
        return layout("/", f"Universal Dependencies parse from <code>{MODEL}</code>. Enter Latin text or pick a sample.",
                      input_form("/", DEFAULT_TEXT))
    text = _clean(text) or DEFAULT_TEXT
    return layout("/", f"Universal Dependencies parse from <code>{MODEL}</code>.",
                  input_form("/", text) + parser_result(text))


@app.get("/parse.{fmt}")
def parse_download(fmt: str, text: str = Query(None)):
    """The parser table as a downloadable file: tsv (default), csv, json, or conllu."""
    spec = DOWNLOAD_FORMATS.get(fmt)
    if spec is None:
        return Response("Unsupported format", status_code=404, media_type="text/plain")
    media, serialize, _lbl = spec
    text = _clean(text) or DEFAULT_TEXT
    return Response(
        content=serialize(parse_rows(text)),
        media_type=f"{media}; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="latincy-parse.{fmt}"'},
    )


@app.get("/senter", response_class=HTMLResponse)
def senter(text: str = Query(None)):
    if text is None:
        return layout("/senter", "Split a passage into sentences.", input_form("/senter", DEFAULT_TEXT, button="Segment"))
    text = _clean(text) or DEFAULT_TEXT
    return layout("/senter", "Split a passage into sentences.", input_form("/senter", text, button="Segment") + senter_result(text))


@app.get("/ner", response_class=HTMLResponse)
def ner(text: str = Query(None)):
    if text is None:
        return layout("/ner", "Highlight named entities — people, places, and groups (PER, LOC, NORP).", input_form("/ner", DEFAULT_TEXT))
    text = _clean(text) or DEFAULT_TEXT
    return layout("/ner", "Highlight named entities.", input_form("/ner", text) + ner_result(text))


@app.get("/dependency", response_class=HTMLResponse)
def dep(text: str = Query(None)):
    if text is None:
        return layout("/dependency", "Visualize dependency parse trees.", input_form("/dependency", DEFAULT_TEXT))
    text = _clean(text) or DEFAULT_TEXT
    return layout("/dependency", "Visualize dependency parse trees.", input_form("/dependency", text) + dep_result(text))


@app.get("/custom-label", response_class=HTMLResponse)
def cl(text: str = Query(None)):
    if text is None:
        return layout("/custom-label", "Highlight tokens in the DCC Core Latin Vocabulary.", input_form("/custom-label", DEFAULT_TEXT))
    text = _clean(text) or DEFAULT_TEXT
    return layout("/custom-label", "Highlight tokens in the DCC Core Latin Vocabulary.", input_form("/custom-label", text) + customlabel_result(text))


VOCAB_INTRO = ('Build a glossed, textbook-style vocabulary list from a passage — each headword '
               'in its citation form with a short gloss. '
               'Sort by first occurrence, frequency, or alphabetically, and filter against the '
               '<a href="https://dcc.dickinson.edu/vocab/core-vocabulary">DCC Core</a>.')


@app.get("/vocab", response_class=HTMLResponse)
def vocab(text: str = Query(None), sort: str = Query("alpha"), dcc: str = Query("all")):
    sort = sort if sort in VOCAB_SORTS else "alpha"
    dcc = dcc if dcc in VOCAB_DCC else "all"
    if text is None:
        return layout("/vocab", VOCAB_INTRO, vocab_form(VOCAB_DEFAULT, sort, dcc))
    text = _clean(text) or VOCAB_DEFAULT
    return layout("/vocab", VOCAB_INTRO, vocab_form(text, sort, dcc) + vocab_result(text, sort, dcc))


@app.get("/vocab.{fmt}")
def vocab_download(fmt: str, text: str = Query(None), sort: str = Query("alpha"), dcc: str = Query("all")):
    """The vocabulary list as a downloadable file: md (Markdown glossary) or json."""
    spec = VOCAB_DOWNLOADS.get(fmt)
    if spec is None:
        return Response("Unsupported format", status_code=404, media_type="text/plain")
    media, method, filename, _lbl = spec
    sort = sort if sort in VOCAB_SORTS else "alpha"
    dcc = dcc if dcc in VOCAB_DCC else "all"
    text = _clean(text) or VOCAB_DEFAULT
    entries, _grand, _core = vocab_entries(text, sort, dcc)
    content = getattr(VocabList(entries=entries), method)()
    return Response(
        content=content,
        media_type=f"{media}; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


UV_INTRO = ('Rule-based U/V spelling (<code>latincy-preprocess</code>). Whatever you enter is first '
            'stripped to u-only — as in many manuscripts and early printings — then the normalizer '
            'restores consonantal <code>u</code>&nbsp;&rarr;&nbsp;<code>v</code>, scored against your '
            'input (<ins>match</ins> / <del>miss</del>).')


@app.get("/uv", response_class=HTMLResponse)
def uv(text: str = Query(None)):
    if text is None:
        return layout("/uv", UV_INTRO, uv_form(UV_DEFAULT))
    text = _clean(text) or UV_DEFAULT
    return layout("/uv", UV_INTRO, uv_form(text) + uv_result(text))


@app.get("/healthz")
def healthz():
    return {"ok": _nlp is not None}
