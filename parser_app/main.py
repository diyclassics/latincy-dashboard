"""Minimal FastAPI parser demo for the LatinCy `la_core_web_lg` model.

A deliberately plain alternative to the Streamlit dashboard: the model is loaded
ONCE at startup, inference runs in a normal request handler (serialized behind a
lock), and results render as a static HTML table. No Streamlit runtime — which
is where the environment-specific native SIGSEGV lived; plain spaCy inference is
stable on the host.
"""

import html
import json
import threading
from contextlib import asynccontextmanager

import spacy
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

MODEL = "la_core_web_lg"
MAX_TOKENS = 500

SAMPLE_PASSAGES = {
    "Seneca, Ep. 1.1": "Ita fac, mi Lucili; vindica te tibi, et tempus quod adhuc aut auferebatur aut subripiebatur aut excidebat collige et serva.",
    "Vergil, Aen. 1.1": "Arma virumque cano, Troiae qui primus ab oris Italiam fato profugus Laviniaque venit litora, multum ille et terris iactatus et alto vi superum saevae memorem Iunonis ob iram, multa quoque et bello passus, dum conderet urbem inferretque deos Latio, genus unde Latinum Albanique patres atque altae moenia Romae.",
    "Caesar, B.G. 1.1": "Gallia est omnis divisa in partes tres, quarum unam incolunt Belgae, aliam Aquitani, tertiam qui ipsorum lingua Celtae, nostra Galli appellantur.",
    "Ritchie, Fab. 1": "Olim in Graecia puer erat, qui Hercules appellabatur.",
    "Cicero, Cat. 1.1": "Quo usque tandem abutere, Catilina, patientia nostra? quam diu etiam furor iste tuus nos eludet? quem ad finem sese effrenata iactabit audacia?",
}
DEFAULT_TEXT = SAMPLE_PASSAGES["Seneca, Ep. 1.1"]
COLUMNS = ["sent_id", "token_id", "form", "lemma", "upos", "xpos", "feats", "head", "deprel", "ent_type"]

_nlp = None
_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    global _nlp
    _nlp = spacy.load(MODEL)
    if "trf_vectors" in _nlp.pipe_names:
        _nlp.disable_pipe("trf_vectors")
    yield


app = FastAPI(lifespan=lifespan, title="LatinCy Parser")


def _format_morph(morph):
    d = morph.to_dict()
    return ", ".join(f"{k}={v}" for k, v in d.items()) if d else ""


def analyze(text: str):
    with _lock:  # serialize inference (cheap on 1 vCPU; avoids any shared-model races)
        doc = _nlp(text)
    rows, n = [], 0
    for si, sent in enumerate(doc.sents):
        start = sent.start
        for ti, tok in enumerate(sent):
            if n >= MAX_TOKENS:
                break
            head = 0 if tok.head == tok else tok.head.i - start + 1
            rows.append([f"s{si + 1}", ti + 1, tok.text, tok.lemma_, tok.pos_, tok.tag_,
                         _format_morph(tok.morph), head, tok.dep_, tok.ent_type_])
            n += 1
        if n >= MAX_TOKENS:
            break
    return rows


def _results_html(rows):
    if not rows:
        return ""
    thead = "".join(f"<th>{html.escape(c)}</th>" for c in COLUMNS)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>"
        for r in rows
    )
    n_sents = len({r[0] for r in rows})
    tsv = "\t".join(COLUMNS) + "\n" + "\n".join("\t".join(str(v) for v in r) for r in rows)
    return f"""
      <p class="summary">Analyzed {len(rows)} tokens in {n_sents} sentence(s) with <code>{MODEL}</code>.</p>
      <div class="tablewrap"><table class="parse"><thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table></div>
      <details class="tsv"><summary>Copy as TSV (CoNLL-U order)</summary><textarea readonly rows="6">{html.escape(tsv)}</textarea></details>
    """


def render(text: str, rows=None):
    buttons = "".join(
        f'<button type="button" class="pill" onclick="setText({json.dumps(k)})">{html.escape(k)}</button>'
        for k in SAMPLE_PASSAGES
    )
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="LatinCy Text Analyzer — Universal Dependencies parse of Latin text with the la_core_web_lg model.">
<title>LatinCy Parser</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Lexend:wght@400;500;700&display=swap">
<style>
  :root {{ font-family:"Lexend",-apple-system,system-ui,sans-serif; --accent:#1d70c7; }}
  * {{ box-sizing:border-box; }}
  body {{ max-width:64rem; margin:2rem auto; padding:0 1rem; line-height:1.5; color:#1a1a1a; }}
  .skip-link {{ position:absolute; top:-40px; left:.5rem; background:var(--accent); color:#fff; padding:.4rem .75rem; border-radius:0 0 4px 4px; z-index:100; text-decoration:none; }}
  .skip-link:focus {{ top:0; }}
  :focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
  header {{ display:flex; justify-content:space-between; align-items:baseline; border-bottom:1px solid #ccc; padding-bottom:.5rem; }}
  .brand {{ text-decoration:none; color:var(--accent); font-size:1.3rem; font-weight:700; letter-spacing:-0.01em; }}
  .brand .suffix {{ color:#000; }}
  header nav a {{ color:var(--accent); text-decoration:none; font-size:.95rem; margin-left:1rem; }}
  .lede {{ font-size:1.05rem; margin:1.5rem 0 1rem; }}
  textarea[name=text] {{ width:100%; min-height:150px; padding:.75rem 1rem; font:inherit; font-size:1.05rem; border:1px solid #767676; border-radius:4px; resize:vertical; }}
  .pills {{ margin:.7rem 0 .2rem; display:flex; flex-wrap:wrap; gap:.4rem; }}
  .pill {{ font:inherit; border:1px solid #767676; background:#fff; border-radius:999px; padding:.3rem .85rem; font-size:.85rem; color:#333; cursor:pointer; }}
  .pill:hover {{ border-color:var(--accent); color:var(--accent); }}
  .go {{ margin-top:.9rem; background:var(--accent); color:#fff; border:none; border-radius:4px; padding:.7rem 1.4rem; font:inherit; font-size:1.05rem; cursor:pointer; }}
  .go:hover {{ opacity:.9; }}
  .summary {{ margin:1.6rem 0 .5rem; font-size:1.05rem; }}
  .tablewrap {{ overflow:auto; max-height:65vh; border:1px solid #e2e2e2; border-radius:6px; }}
  table.parse {{ border-collapse:collapse; width:auto; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.82rem; }}
  table.parse th, table.parse td {{ text-align:left; padding:.3rem .7rem; white-space:nowrap; border-bottom:1px solid #eee; }}
  table.parse th {{ position:sticky; top:0; background:#fafafa; font-weight:600; }}
  details.tsv {{ margin-top:.8rem; }} details.tsv textarea {{ width:100%; font-family:ui-monospace,monospace; font-size:.75rem; }}
  code {{ background:#f0f1f3; padding:.05rem .35rem; border-radius:4px; }}
  footer {{ margin-top:4rem; color:#666; border-top:1px solid #eee; padding-top:1rem; }}
  footer a {{ color:var(--accent); }}
</style></head>
<body>
  <a class="skip-link" href="#main-content">Skip to main content</a>
  <header>
    <a href="/" class="brand">LatinCy <span class="suffix">Parser</span></a>
    <nav><a href="https://huggingface.co/latincy">Models</a></nav>
  </header>
  <main id="main-content">
    <p class="lede">Universal Dependencies parse from the <code>{MODEL}</code> model. Enter Latin text (max {MAX_TOKENS} tokens) or pick a sample passage.</p>
    <form method="post" action="/analyze">
      <textarea name="text" id="text" aria-label="Latin text to analyze">{html.escape(text)}</textarea>
      <div class="pills">{buttons}</div>
      <button class="go" type="submit">Analyze</button>
    </form>
    {_results_html(rows)}
  </main>
  <footer><small>
    Built on the <a href="https://huggingface.co/latincy">LatinCy</a> <code>{MODEL}</code> model.
    Written by <a href="https://github.com/diyclassics">P.&nbsp;J.&nbsp;Burns</a> + Claude Opus&nbsp;4.8.
  </small></footer>
  <script>
    const SAMPLES = {json.dumps(SAMPLE_PASSAGES)};
    function setText(k) {{ document.getElementById('text').value = SAMPLES[k]; }}
  </script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return render(DEFAULT_TEXT)


@app.post("/analyze", response_class=HTMLResponse)
def do_analyze(text: str = Form("")):
    text = (text or "").strip() or DEFAULT_TEXT
    return render(text, analyze(text))


@app.get("/healthz")
def healthz():
    return {"ok": _nlp is not None}
