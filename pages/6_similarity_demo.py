import streamlit as st
import spacy
import numpy as np

st.set_page_config(page_title="Similarity Demo", layout="wide")
st.sidebar.header("Similarity Demo")

st.title("Latin Word Similarity Explorer")

st.markdown(
    """
Explore **word vector similarity** between Latin words. LatinCy uses
[floret](https://github.com/facebookresearch/floret) subword vectors,
so even rare or unseen forms get meaningful representations.

*Only the **md** and **lg** models include word vectors — the sm model does not.*
"""
)

model_selectbox = st.sidebar.selectbox(
    "Choose model:", ("la_core_web_lg", "la_core_web_md")
)


@st.cache_resource
def load_model(model_name):
    return spacy.load(model_name, exclude=["ner", "parser", "senter"])


nlp = load_model(model_selectbox)

# Curated candidate list: common Latin lemmas from DCC Core Vocabulary.
# This is necessary because floret vectors don't support most_similar().
CANDIDATES = [
    "ab", "abeo", "absum", "ac", "accedo", "accido", "accipio", "acer",
    "acies", "ad", "addo", "adduco", "adeo", "adhibeo", "adhuc", "adsum",
    "aduenio", "aduersus", "aedes", "aeger", "aequus", "aer", "aes",
    "aetas", "aeternus", "ager", "agito", "agmen", "ago", "aio", "albus",
    "alienus", "aliquis", "alius", "alo", "alter", "altus", "amicitia",
    "amicus", "amitto", "amnis", "amo", "amor", "amplus", "an", "anima",
    "animal", "animus", "annus", "ante", "antiquus", "aperio", "appello",
    "aptus", "apud", "aqua", "ara", "arbor", "ardeo", "argentum", "arma",
    "ars", "arx", "ascendo", "aspicio", "astrum", "at", "atque", "auctor",
    "auctoritas", "audax", "audeo", "audio", "aufero", "augeo", "aura",
    "aureus", "auris", "aurum", "aut", "autem", "auxilium", "auis",
    "barbarus", "beatus", "bellum", "bene", "bonus", "bos", "breuis",
    "cado", "caecus", "caedes", "caelum", "campus", "canis", "cano",
    "capio", "caput", "careo", "carmen", "carus", "castrum", "casus",
    "causa", "cedo", "celer", "censeo", "centum", "cerno", "certus",
    "cibus", "cingo", "ciuis", "ciuitas", "clamor", "clarus", "classis",
    "claudo", "coepi", "cogito", "cognosco", "cogo", "colo", "color",
    "comes", "committo", "communis", "comparo", "concedo", "condo",
    "confero", "conficio", "coniunx", "conor", "consequor", "consilium",
    "consisto", "constituo", "consul", "consulo", "contemno", "contineo",
    "contra", "conuenio", "copia", "cor", "cornu", "corpus", "credo",
    "creo", "cresco", "crimen", "culpa", "cum", "cunctus", "cupido",
    "cupio", "cur", "cura", "curo", "curro", "currus", "cursus", "custos",
    "damnum", "de", "debeo", "decem", "decerno", "decus", "deduco",
    "defendo", "defero", "deinde", "denique", "descendo", "desero",
    "desidero", "deus", "dexter", "dico", "dies", "difficilis", "dignitas",
    "dignus", "dimitto", "discedo", "disco", "diu", "diues", "diuus", "do",
    "doceo", "doleo", "dolor", "dolus", "dominus", "domus", "donec",
    "donum", "dubius", "duco", "dulcis", "dum", "duo", "durus", "dux",
    "ecce", "edo", "educo", "efficio", "ego", "egregius", "enim", "eo",
    "eques", "equus", "ergo", "eripio", "erro", "error", "et", "etiam",
    "ex", "excipio", "exemplum", "exeo", "exerceo", "exercitus", "exigo",
    "existimo", "exsilium", "exspecto", "extremus",
    "fabula", "facies", "facilis", "facinus", "facio", "fallo", "falsus",
    "fama", "fames", "familia", "fatum", "felix", "femina", "fere", "fero",
    "ferrum", "ferus", "fidelis", "fides", "filia", "filius", "fingo",
    "finis", "fio", "flamma", "fleo", "flos", "flumen", "fluo", "foedus",
    "fons", "forma", "fortis", "fortuna", "forum", "frango", "frater",
    "frons", "fructus", "fruor", "frustra", "fuga", "fugio", "fundo",
    "funus", "furor",
    "gaudeo", "gaudium", "gens", "genus", "gero", "gigno", "gladius",
    "gloria", "gradus", "gratia", "gratus", "grauis",
    "habeo", "haud", "hic", "hiems", "homo", "honestus", "honor", "hora",
    "hortor", "hospes", "hostis", "humanus", "humus",
    "iaceo", "iacio", "iam", "ibi", "idem", "igitur", "ignis", "ille",
    "imago", "imperator", "imperium", "impero", "impetus", "impleo", "in",
    "incido", "incipio", "inde", "infero", "ingenium", "ingens",
    "inimicus", "initium", "iniuria", "inquam", "instituo", "insula",
    "integer", "intellego", "inter", "interficio", "intra", "inuenio",
    "inuidia", "ipse", "ira", "is", "iste", "ita", "itaque", "iter",
    "iterum", "iubeo", "iudex", "iudicium", "iudico", "iugum", "iungo",
    "ius", "iustus", "iuuenis", "iuuo",
    "labor", "laboro", "lacrima", "laetus", "lapis", "latus", "laudo",
    "laus", "legatus", "legio", "lego", "leuis", "lex", "liber",
    "libertas", "limen", "lingua", "littera", "litus", "locus", "longus",
    "loquor", "lumen", "luna", "lux",
    "magis", "magister", "magnus", "maior", "malo", "malus", "maneo",
    "manus", "mare", "maritus", "mater", "materia", "maximus", "medius",
    "melior", "membrum", "memini", "memoria", "mens", "mensa", "mereo",
    "metuo", "metus", "meus", "miles", "mille", "minus", "miror", "misceo",
    "miser", "mitto", "modus", "moenia", "mollis", "moneo", "mons", "mora",
    "morbus", "morior", "mors", "mortalis", "mos", "moueo", "mox",
    "mulier", "multus", "mundus", "munus", "murus", "muto",
    "nam", "narro", "nascor", "natura", "natus", "nauis", "ne", "nec",
    "necesse", "nego", "negotium", "nemo", "nemus", "neque", "nescio",
    "nihil", "nimius", "nisi", "nobilis", "noceo", "nolo", "nomen", "non",
    "nos", "nosco", "noster", "notus", "nouus", "nox", "nullus", "numen",
    "numerus", "numquam", "nunc", "nuntius",
    "ob", "occido", "occupo", "occurro", "oculus", "odi", "odium",
    "offero", "olim", "omnis", "onus", "opera", "oportet", "oppidum",
    "ops", "optimus", "opto", "opus", "oratio", "orbis", "ordo", "orior",
    "oro", "os", "ostendo", "otium",
    "par", "parco", "parens", "pareo", "pario", "paro", "pars", "paruus",
    "pateo", "pater", "patior", "patria", "pauci", "pauper", "pax",
    "pectus", "pecunia", "pecus", "pello", "per", "perdo", "pereo",
    "periculum", "permitto", "perpetuus", "pes", "peto", "pietas", "pius",
    "placeo", "plebs", "plenus", "plurimus", "plus", "poena", "poeta",
    "pondus", "pono", "populus", "porta", "porto", "posco", "possum",
    "post", "posterus", "postquam", "potens", "potestas", "praebeo",
    "praeda", "praemium", "praesens", "praesto", "praeter", "praetor",
    "precor", "premo", "pretium", "primus", "princeps", "prior",
    "priuatus", "pro", "probo", "procedo", "procul", "proelium",
    "proficiscor", "prohibeo", "promitto", "prope", "propero", "proprius",
    "propter", "prosum", "prouincia", "publicus", "pudor", "puella",
    "puer", "pugna", "pugno", "pulcher", "puto",
    "quaero", "qualis", "quam", "quamquam", "quando", "quantus", "quare",
    "quattuor", "queror", "qui", "quia", "quidam", "quidem", "quin",
    "quis", "quisquam", "quisque", "quo", "quondam", "quoniam", "quoque",
    "rapio", "rarus", "ratio", "recipio", "rectus", "reddo", "redeo",
    "refero", "regio", "regnum", "rego", "relinquo", "reliquus", "reperio",
    "res", "respicio", "respondeo", "retineo", "reus", "rex", "rogo",
    "rumpo", "rus",
    "sacer", "sacerdos", "saeculum", "saepe", "saeuus", "salus", "sanctus",
    "sanguis", "sanus", "sapiens", "sapientia", "satis", "saxum", "scelus",
    "scientia", "scio", "scribo", "secundus", "sed", "sedeo", "sedes",
    "semper", "senatus", "senex", "sensus", "sententia", "sentio",
    "sequor", "sermo", "seruio", "seruo", "seruus", "si", "sic", "sidus",
    "signum", "silua", "similis", "simul", "sine", "sino", "sinus",
    "socius", "sol", "soleo", "solus", "soluo", "somnus", "soror", "sors",
    "spatium", "species", "specto", "spero", "spes", "spiritus", "statuo",
    "stella", "sto", "studeo", "studium", "sub", "subeo", "subito", "sui",
    "sum", "summus", "sumo", "super", "superbus", "supero", "supplicium",
    "supra", "surgo", "suscipio", "sustineo", "suus",
    "taceo", "talis", "tam", "tamen", "tandem", "tango", "tantus",
    "tectum", "tellus", "telum", "tempestas", "templum", "tempus", "tendo",
    "teneo", "tener", "terra", "terreo", "tertius", "testis", "timeo",
    "timor", "tollo", "tot", "totus", "trado", "traho", "transeo", "tres",
    "tribunus", "tristis", "tu", "tum", "turba", "turpis", "tutus", "tuus",
    "ubi", "ullus", "ultimus", "ultra", "umbra", "umquam", "unda", "unde",
    "unus", "urbs", "usque", "usus", "ut", "uterque", "utilis", "utor",
    "uxor",
    "uaco", "ualeo", "uanus", "uarius", "uates", "ueho", "uel", "uenio",
    "uentus", "uerbum", "uereor", "uero", "uerto", "uerus", "uestigium",
    "uestis", "uetus", "uia", "uicinus", "uictor", "uictoria", "uideo",
    "uinco", "uinculum", "uinum", "uir", "uirgo", "uirtus", "uis",
    "uita", "uitium", "uiuo", "uoco", "uolo", "uoluntas", "uoluptas",
    "uos", "uotum", "uox", "uulgus", "uulnus", "uultus",
]


@st.cache_data
def get_candidate_vectors(_nlp, candidates):
    """Pre-compute vectors for all candidate words."""
    vectors = {}
    for word in candidates:
        token = _nlp.vocab[word]
        if token.has_vector:
            vectors[word] = token.vector
    return vectors


candidate_vectors = get_candidate_vectors(nlp, CANDIDATES)

tab1, tab2 = st.tabs(["Find Similar Words", "Compare Two Words"])

with tab1:
    word = st.text_input("Enter a Latin word:", value="rex", key="sim_word")
    n_results = st.slider("Number of results:", min_value=1, max_value=10, value=3)

    if st.button("Find Similar", key="btn_similar"):
        token = nlp.vocab[word]
        if not token.has_vector:
            st.error(f"No vector found for '{word}'.")
        else:
            word_vec = token.vector
            word_norm = np.linalg.norm(word_vec)
            results = []
            for candidate, cand_vec in candidate_vectors.items():
                if candidate == word:
                    continue
                sim = np.dot(word_vec, cand_vec) / (
                    word_norm * np.linalg.norm(cand_vec)
                )
                results.append((candidate, float(sim)))
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:n_results]

            st.markdown(f"**Top {len(results)} words most similar to *{word}*:**")
            for i, (w, score) in enumerate(results, 1):
                st.text(f"{i:3d}. {w:<20s} {score:.4f}")

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        word_a = st.text_input("First word:", value="rex", key="word_a")
    with col2:
        word_b = st.text_input("Second word:", value="regina", key="word_b")

    if st.button("Compare", key="btn_compare"):
        doc_a = nlp(word_a)
        doc_b = nlp(word_b)

        has_a = doc_a[0].has_vector if len(doc_a) > 0 else False
        has_b = doc_b[0].has_vector if len(doc_b) > 0 else False

        if not has_a:
            st.error(f"No vector found for '{word_a}'.")
        elif not has_b:
            st.error(f"No vector found for '{word_b}'.")
        else:
            similarity = doc_a.similarity(doc_b)
            st.metric(
                label=f"Similarity: {word_a} ↔ {word_b}",
                value=f"{similarity:.4f}",
            )
            if similarity > 0.7:
                st.success("These words are **highly similar**.")
            elif similarity > 0.4:
                st.info("These words are **moderately similar**.")
            else:
                st.warning("These words are **not very similar**.")
