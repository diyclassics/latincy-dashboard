"""LatinCy demos — a plain FastAPI site (no Streamlit).

Streamlit's runtime segfaults on inference on this host; plain spaCy is stable.
The lg model is loaded ONCE at startup and shared by every model-based demo
(parser/senter/NER/dependency/custom-label) — inference is serialized behind a
lock. U/V spelling is rule-based (no model). Left nav; parser is the landing
page. LatinCy house style (Lexend, #1d70c7), matching latincy-lexicon-site.
"""

import html
import threading
from contextlib import asynccontextmanager

import spacy
from spacy import displacy
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

from dcc_helpers import DCC_CORE_LEMMAS

MODEL = "la_core_web_lg"
MAX_TOKENS = 500

# Each sample is >= 3 sentences (the demos are more useful multi-sentence).
SAMPLE_PASSAGES = {
    "Seneca, Ep. 1.1": "Ita fac, mi Lucili; vindica te tibi, et tempus quod adhuc aut auferebatur aut subripiebatur aut excidebat collige et serva. Persuade tibi hoc sic esse ut scribo: quaedam tempora eripiuntur nobis, quaedam subducuntur, quaedam effluunt. Turpissima tamen est iactura quae per neglegentiam fit.",
    "Vergil, Aen. 1.1": "Arma virumque cano, Troiae qui primus ab oris Italiam fato profugus Laviniaque venit litora, multum ille et terris iactatus et alto vi superum saevae memorem Iunonis ob iram, multa quoque et bello passus, dum conderet urbem inferretque deos Latio, genus unde Latinum Albanique patres atque altae moenia Romae. Musa, mihi causas memora, quo numine laeso quidve dolens regina deum tot volvere casus insignem pietate virum, tot adire labores impulerit. Tantaene animis caelestibus irae?",
    "Caesar, B.G. 1.1": "Gallia est omnis divisa in partes tres, quarum unam incolunt Belgae, aliam Aquitani, tertiam qui ipsorum lingua Celtae, nostra Galli appellantur. Hi omnes lingua, institutis, legibus inter se differunt. Gallos ab Aquitanis Garumna flumen, a Belgis Matrona et Sequana dividit.",
    "Ritchie, Perseus": "Haec narrantur a poetis de Perseo. Perseus filius erat Iovis, maximi deorum; avus eius Acrisius appellabatur. Acrisius volebat Perseum nepotem suum necare.",
    "Cicero, Cat. 1.1": "Quo usque tandem abutere, Catilina, patientia nostra? quam diu etiam furor iste tuus nos eludet? quem ad finem sese effrenata iactabit audacia?",
}
DEFAULT_TEXT = SAMPLE_PASSAGES["Seneca, Ep. 1.1"]
# A longer u-only passage (Aeneid 1.1-11) so the U/V demo shows many corrections.
UV_DEFAULT = "Arma uirumque cano, Troiae qui primus ab oris Italiam fato profugus Lauiniaque uenit litora, multum ille et terris iactatus et alto ui superum saeuae memorem Iunonis ob iram. Musa, mihi causas memora, quo numine laeso quidue dolens regina deum tot uoluere casus insignem pietate uirum, tot adire labores impulerit."
COLUMNS = ["sent_id", "token_id", "form", "lemma", "upos", "xpos", "feats", "head", "deprel", "ent_type"]

# (path, label) — order = left-nav order; first entry is the landing page.
NAV = [
    ("/", "Parser"),
    ("/senter", "Sentences"),
    ("/ner", "Entities"),
    ("/dependency", "Dependencies"),
    ("/custom-label", "DCC Core"),
    ("/uv", "U/V spelling"),
]

_nlp = None
_lock = threading.Lock()
_uv = None


@asynccontextmanager
async def lifespan(app):
    global _nlp, _uv
    _nlp = spacy.load(MODEL)
    if "trf_vectors" in _nlp.pipe_names:
        _nlp.disable_pipe("trf_vectors")
    from latincy_preprocess.uv import UVNormalizerRules
    _uv = UVNormalizerRules()
    yield


app = FastAPI(lifespan=lifespan, title="LatinCy demos")


def nlp(text):
    """Serialized inference on the shared model (cheap on 1 vCPU; avoids races)."""
    with _lock:
        return _nlp(text)


# --------------------------------------------------------------------------- #
# layout                                                                       #
# --------------------------------------------------------------------------- #
def _cap(text):
    return " ".join(text.split()[: MAX_TOKENS + 50])


def _samples_pills():
    btns = "".join(
        f'<button type="button" class="pill" data-text="{html.escape(v)}">{html.escape(k)}</button>'
        for k, v in SAMPLE_PASSAGES.items()
    )
    return f'<div class="pills">{btns}</div>'


def input_form(action, text, *, samples=True, label="Enter Latin text", button="Analyze", rows=6):
    pills = _samples_pills() if samples else ""
    return f"""
    <form method="post" action="{action}">
      <textarea name="text" id="text" rows="{rows}" aria-label="{html.escape(label)}">{html.escape(text)}</textarea>
      {pills}
      <button class="go" type="submit">{html.escape(button)}</button>
    </form>"""


def layout(active, intro, body):
    nav = "".join(
        f'<a href="{p}"{" class=active" if p == active else ""}>{html.escape(lbl)}</a>'
        for p, lbl in NAV
    )
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="LatinCy demos — Latin NLP with the la_core_web_lg model.">
<title>LatinCy demos</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Lexend:wght@400;500;700&display=swap">
<style>
  :root {{ font-family:"Lexend",-apple-system,system-ui,sans-serif; --accent:#1d70c7; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; line-height:1.5; color:#1a1a1a; display:flex; min-height:100vh; }}
  .skip-link {{ position:absolute; top:-40px; left:.5rem; background:var(--accent); color:#fff; padding:.4rem .75rem; border-radius:0 0 4px 4px; z-index:100; text-decoration:none; }}
  .skip-link:focus {{ top:0; }}
  :focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
  aside {{ width:210px; flex:0 0 210px; border-right:1px solid #e2e2e2; padding:1.4rem 1rem; position:sticky; top:0; height:100vh; overflow:auto; }}
  .brand {{ display:block; text-decoration:none; color:var(--accent); font-size:1.25rem; font-weight:700; letter-spacing:-0.01em; margin-bottom:1.2rem; }}
  .brand .suffix {{ color:#000; }}
  aside nav {{ display:flex; flex-direction:column; gap:.15rem; }}
  aside nav a {{ text-decoration:none; color:#333; padding:.4rem .6rem; border-radius:6px; font-size:.95rem; }}
  aside nav a:hover {{ background:#f2f6fb; color:var(--accent); }}
  aside nav a.active {{ background:var(--accent); color:#fff; }}
  main {{ flex:1; padding:2rem 2.2rem 4rem; max-width:60rem; }}
  h1 {{ font-size:1.6rem; margin:0 0 .3rem; }}
  .lede {{ font-size:1.02rem; color:#444; margin:0 0 1.3rem; }}
  textarea {{ width:100%; padding:.75rem 1rem; font:inherit; font-size:1.03rem; border:1px solid #767676; border-radius:4px; resize:vertical; }}
  .pills {{ margin:.7rem 0 .2rem; display:flex; flex-wrap:wrap; gap:.4rem; }}
  .pill {{ font:inherit; border:1px solid #767676; background:#fff; border-radius:999px; padding:.3rem .85rem; font-size:.83rem; color:#333; cursor:pointer; }}
  .pill:hover {{ border-color:var(--accent); color:var(--accent); }}
  .go {{ margin-top:.9rem; background:var(--accent); color:#fff; border:none; border-radius:4px; padding:.6rem 1.4rem; font:inherit; font-size:1.02rem; cursor:pointer; }}
  .go:hover {{ opacity:.9; }}
  .summary {{ margin:1.6rem 0 .6rem; font-size:1.02rem; }}
  .tablewrap {{ overflow:auto; max-height:62vh; border:1px solid #e2e2e2; border-radius:6px; }}
  table.parse {{ border-collapse:collapse; width:auto; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.82rem; }}
  table.parse th, table.parse td {{ text-align:left; padding:.3rem .7rem; white-space:nowrap; border-bottom:1px solid #eee; }}
  table.parse th {{ position:sticky; top:0; background:#fafafa; font-weight:600; }}
  ol.sents {{ font-size:1.05rem; }} ol.sents li {{ margin:.35rem 0; }}
  .render {{ overflow:auto; border:1px solid #e2e2e2; border-radius:6px; padding:.6rem 1rem; margin-top:.6rem; }}
  mark.core {{ background:#d6ebfb; padding:0 .05em; border-radius:3px; }}
  .textout {{ font-size:1.1rem; line-height:1.7; border:1px solid #e2e2e2; border-radius:6px; padding:.8rem 1rem; margin-top:.6rem; }}
  ins {{ background:#d8f5d8; text-decoration:none; }} del {{ background:#f7d7d7; }}
  .modes {{ border:none; margin:.7rem 0 0; padding:0; display:flex; flex-direction:column; gap:.3rem; }}
  .modes label {{ font-size:.95rem; }}
  .uonly {{ color:#666; font-size:.9rem; margin:.4rem 0; word-break:break-word; }}
  details.tsv {{ margin-top:.8rem; }} details.tsv textarea {{ width:100%; font-family:ui-monospace,monospace; font-size:.75rem; }}
  code {{ background:#f0f1f3; padding:.05rem .35rem; border-radius:4px; }}
  footer {{ margin-top:3rem; color:#666; border-top:1px solid #eee; padding-top:1rem; font-size:.9rem; }}
  footer a {{ color:var(--accent); }}
  @media (max-width:720px) {{ body {{ flex-direction:column; }} aside {{ width:auto; height:auto; position:static; border-right:none; border-bottom:1px solid #e2e2e2; }} aside nav {{ flex-flow:row wrap; }} }}
</style></head>
<body>
  <a class="skip-link" href="#main-content">Skip to main content</a>
  <aside>
    <a href="/" class="brand">LatinCy <span class="suffix">demos</span></a>
    <nav>{nav}</nav>
  </aside>
  <main id="main-content">
    <h1>{html.escape(dict(NAV)[active])}</h1>
    <p class="lede">{intro}</p>
    {body}
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
  </script>
</body></html>"""


# --------------------------------------------------------------------------- #
# demos                                                                        #
# --------------------------------------------------------------------------- #
def _format_morph(morph):
    d = morph.to_dict()
    return ", ".join(f"{k}={v}" for k, v in d.items()) if d else ""


def parser_result(text):
    doc = nlp(_cap(text))
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
    if not rows:
        return ""
    thead = "".join(f"<th>{html.escape(c)}</th>" for c in COLUMNS)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>" for r in rows)
    tsv = "\t".join(COLUMNS) + "\n" + "\n".join("\t".join(str(v) for v in r) for r in rows)
    n_sents = len({r[0] for r in rows})
    return (f'<p class="summary">Analyzed {len(rows)} tokens in {n_sents} sentence(s).</p>'
            f'<div class="tablewrap"><table class="parse"><thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table></div>'
            f'<details class="tsv"><summary>Copy as TSV (CoNLL-U order)</summary>'
            f'<textarea readonly rows="6">{html.escape(tsv)}</textarea></details>')


def senter_result(text):
    doc = nlp(_cap(text))
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    if not sents:
        return ""
    items = "".join(f"<li>{html.escape(s)}</li>" for s in sents)
    return f'<p class="summary">{len(sents)} sentence(s).</p><ol class="sents">{items}</ol>'


def ner_result(text):
    doc = nlp(_cap(text))
    n = len(doc.ents)
    svg = displacy.render(doc, style="ent", page=False, minify=True)
    return (f'<p class="summary">{n} entit{"y" if n == 1 else "ies"} found '
            f'(PER, LOC, NORP).</p><div class="render">{svg}</div>')


def dep_result(text):
    doc = nlp(_cap(text))
    parts = [displacy.render(sent.as_doc(), style="dep", page=False, minify=True,
                             options={"compact": True, "distance": 90})
             for sent in doc.sents]
    if not parts:
        return ""
    return f'<p class="summary">{len(parts)} sentence(s).</p><div class="render">' + "".join(parts) + "</div>"


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
                piece = f'<mark class="core">{piece}</mark>'
        out.append(piece + html.escape(tok.whitespace_))
    if not total:
        return ""
    pct = round(core / total * 100, 1)
    return (f'<p class="summary">{core} of {total} tokens ({pct}%) are in the '
            f'<a href="https://dcc.dickinson.edu/vocab/core-vocabulary">DCC Core</a> vocabulary '
            f'(matched on lemma, u-form).</p><div class="textout">{"".join(out)}</div>')


def _diff_html(a, b):
    """Char-level highlight of b vs a (inserts marked)."""
    import difflib
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        seg = html.escape(b[j1:j2])
        out.append(f"<ins>{seg}</ins>" if tag in ("replace", "insert") else seg if tag == "equal" else "")
    return "".join(out)


def uv_result(text, mode):
    if mode == "test":
        # Remove: strip the gold reference to u-only. Test: restore + score.
        source = text.replace("v", "u").replace("V", "U")
        restored = _uv.normalize(source)
        needed = sum(1 for a, b in zip(text, source) if a != b)
        out, correct = [], 0
        for i, ch in enumerate(restored):
            e = html.escape(ch)
            if i < len(source) and ch != source[i]:      # normalizer restored this char
                if i < len(text) and ch == text[i]:
                    correct += 1
                    out.append(f"<ins>{e}</ins>")        # restored correctly
                else:
                    out.append(f"<del>{e}</del>")         # restored wrongly
            else:
                out.append(e)
        if not needed:
            return ('<p class="summary">No v-spelling in the reference to test — paste text with '
                    'correct <code>v</code> (or pick a sample passage) as the reference.</p>')
        acc = round(correct / needed * 100, 1)
        return (f'<p class="summary">Stripped {needed} char(s) to u-only, then restored '
                f'{acc}% correctly ({correct}/{needed}) vs the reference.</p>'
                f'<p class="uonly">u-only &rarr; <code>{html.escape(source)}</code></p>'
                f'<div class="textout">{"".join(out)}</div>')
    normalized = _uv.normalize(text)
    changed = sum(1 for a, b in zip(text, normalized) if a != b)
    return (f'<p class="summary">{changed} character(s) restored (u &rarr; v).</p>'
            f'<div class="textout">{_diff_html(text, normalized)}</div>')


def uv_form(text, mode):
    def chk(m):
        return " checked" if m == mode else ""
    return f"""
    <form method="post" action="/uv">
      <textarea name="text" id="text" rows="4" aria-label="Latin text">{html.escape(text)}</textarea>
      {_samples_pills()}
      <fieldset class="modes">
        <label><input type="radio" name="mode" value="correct"{chk("correct")}> Correct — restore v in u-only text</label>
        <label><input type="radio" name="mode" value="test"{chk("test")}> Remove &amp; test — strip this reference to u-only, restore, and score</label>
      </fieldset>
      <button class="go" type="submit">Run</button>
    </form>"""


# --------------------------------------------------------------------------- #
# routes                                                                       #
# --------------------------------------------------------------------------- #
def _clean(text):
    return (text or "").strip()


@app.get("/", response_class=HTMLResponse)
def parser_get():
    return layout("/", f"Universal Dependencies parse from <code>{MODEL}</code>. Enter Latin text (max {MAX_TOKENS} tokens) or pick a sample.",
                  input_form("/", DEFAULT_TEXT))


@app.post("/", response_class=HTMLResponse)
def parser_post(text: str = Form("")):
    text = _clean(text) or DEFAULT_TEXT
    return layout("/", f"Universal Dependencies parse from <code>{MODEL}</code>.",
                  input_form("/", text) + parser_result(text))


@app.get("/senter", response_class=HTMLResponse)
def senter_get():
    return layout("/senter", "Split a passage into sentences.", input_form("/senter", DEFAULT_TEXT, button="Segment"))


@app.post("/senter", response_class=HTMLResponse)
def senter_post(text: str = Form("")):
    text = _clean(text) or DEFAULT_TEXT
    return layout("/senter", "Split a passage into sentences.", input_form("/senter", text, button="Segment") + senter_result(text))


@app.get("/ner", response_class=HTMLResponse)
def ner_get():
    return layout("/ner", "Highlight named entities — people, places, and groups (PER, LOC, NORP).", input_form("/ner", DEFAULT_TEXT))


@app.post("/ner", response_class=HTMLResponse)
def ner_post(text: str = Form("")):
    text = _clean(text) or DEFAULT_TEXT
    return layout("/ner", "Highlight named entities.", input_form("/ner", text) + ner_result(text))


@app.get("/dependency", response_class=HTMLResponse)
def dep_get():
    return layout("/dependency", "Visualize dependency parse trees.", input_form("/dependency", DEFAULT_TEXT))


@app.post("/dependency", response_class=HTMLResponse)
def dep_post(text: str = Form("")):
    text = _clean(text) or DEFAULT_TEXT
    return layout("/dependency", "Visualize dependency parse trees.", input_form("/dependency", text) + dep_result(text))


@app.get("/custom-label", response_class=HTMLResponse)
def cl_get():
    return layout("/custom-label", "Highlight tokens in the DCC Core Latin Vocabulary.", input_form("/custom-label", DEFAULT_TEXT))


@app.post("/custom-label", response_class=HTMLResponse)
def cl_post(text: str = Form("")):
    text = _clean(text) or DEFAULT_TEXT
    return layout("/custom-label", "Highlight tokens in the DCC Core Latin Vocabulary.", input_form("/custom-label", text) + customlabel_result(text))


UV_INTRO = ('Rule-based U/V spelling. <b>Correct</b> restores consonantal '
            '<code>u</code>&nbsp;&rarr;&nbsp;<code>v</code>; <b>Remove &amp; test</b> strips a '
            'correct reference to u-only, restores it, and scores the result.')


@app.get("/uv", response_class=HTMLResponse)
def uv_get():
    return layout("/uv", UV_INTRO, uv_form(UV_DEFAULT, "correct"))


@app.post("/uv", response_class=HTMLResponse)
def uv_post(text: str = Form(""), mode: str = Form("correct")):
    text = _clean(text) or UV_DEFAULT
    mode = mode if mode in ("correct", "test") else "correct"
    return layout("/uv", UV_INTRO, uv_form(text, mode) + uv_result(text, mode))


@app.get("/healthz")
def healthz():
    return {"ok": _nlp is not None}
