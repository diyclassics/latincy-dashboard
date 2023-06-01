import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Token, Span
import streamlit as st
from spacy_streamlit import visualize_spans

st.set_page_config(page_title='Custom Label Demo', layout="wide")
st.sidebar.header("Custom Label Demo")

default_text = """Ita fac, mi Lucili; vindica te tibi, et tempus, quod adhuc aut auferebatur aut subripiebatur aut excidebat, collige et serva."""

@Language.factory("dcc_core")
def create_dcc_core_merger(nlp, name):
    return DCCCoreMerger(nlp.vocab)

class DCCCoreMerger:
    def __init__(self, vocab):
        patterns = [
            [
            {"LEMMA": {"IN": ["ab", "abeo", "absum", "ac", "accedo", "accido", "accipio", "acer", "acies", "ad", "addo", "adduco", "adeo", "adeo", "adhibeo", "adhuc", "adsum", "aduenio", "aduersus", "aduerto", "aedes", "aeger", "aequor", "aequus", "aer", "aes", "aetas", "aeternus", "aether", "aeuum", "affero", "afficio", "ager", "agito", "agmen", "ago", "aio", "albus", "alienus", "aliquando", "aliquis", "aliter", "alius", "alo", "alter", "altus", "amicitia", "amicus", "amitto", "amnis", "amo", "amor", "amplus", "an", "anima", "animal", "animus", "annus", "ante", "antequam", "antiquus", "aperio", "appareo", "appello", "aptus", "apud", "aqua", "ara", "arbitror", "arbor", "ardeo", "argentum", "arma", "ars", "aruum", "arx", "ascendo", "aspicio", "astrum", "at", "atque", "auctor", "auctoritas", "audax", "audeo", "audio", "aufero", "augeo", "aura", "aureus", "auris", "aurum", "aut", "autem", "auxilium", "auis", "barbarus", "beatus", "bellum", "bene", "beneficium", "bonus", "bos", "breuis", "cado", "caecus", "caedes", "caedo", "caelestis", "caelum", "campus", "candidus", "canis", "cano", "capio", "caput", "careo", "carmen", "carus", "castrum", "castus", "casus", "causa", "caueo", "cedo", "celebro", "celer", "censeo", "centum", "cerno", "certo", "certus", "ceterus", "cibus", "cingo", "cinis", "circa", "citus", "ciuis", "ciuitas", "clamor", "clarus", "classis", "claudo", "coepi", "cogito", "cognosco", "cogo", "cohors", "colligo", "colo", "color", "coma", "comes", "committo", "communis", "comparo", "compono", "concedo", "condicio", "condo", "confero", "conficio", "confiteor", "coniunx", "conor", "consequor", "consilium", "consisto", "constituo", "consto", "consuetudo", "consul", "consulo", "consumo", "contemno", "contineo", "contingo", "contra", "conuenio", "conuerto", "conuiuium", "copia", "cor", "cornu", "corpus", "corrumpo", "credo", "creo", "cresco", "crimen", "culpa", "cum", "cunctus", "cupido", "cupio", "cur", "cura", "curo", "curro", "currus", "cursus", "custos", "damno", "damnum", "de", "debeo", "decem", "decerno", "decet", "decus", "deduco", "defendo", "defero", "deficio", "deinde", "dein", "denique", "descendo", "desero", "desidero", "desino", "desum", "deus", "dexter", "dico", "dies", "differo", "difficilis", "dignitas", "dignus", "diligo", "dimitto", "discedo", "disciplina", "disco", "diu", "diuersus", "diues", "diuido", "diuitiae", "diuus", "do", "doceo", "doleo", "dolor", "dolus", "dominus", "domus", "donec", "dono", "donum", "dormio", "dubito", "dubius", "duco", "dulcis", "dum", "duo", "durus", "dux",  "ecce", "edico", "edo", "educo", "efficio", "effundo", "ego", "egredior", "egregius", "eligo", "enim", "eo", "eo", "epistula", "eques", "equus", "ergo", "eripio", "erro", "error", "et", "etiam", "ex", "excipio", "exemplum", "exeo", "exerceo", "exercitus", "exigo", "existimo", "experior", "exsilium", "exspecto", "extremus", "fabula", "facies", "facilis", "facinus", "facio", "factum", "fallo", "falsus", "fama", "fames", "familia", "fateor", "fatum", "fax", "felix", "femina", "fere", "fero", "ferrum", "ferus", "fessus", "fidelis", "fides", "filia", "filius", "fingo", "finis", "fio", "flamma", "fleo", "flos", "fluctus", "ﬂumen", "fluo", "foedus", "fons", "for", "fore", "forma", "fors", "forsitan", "fortis", "fortuna", "forum", "frango", "frater", "frequens", "frons", "fructus", "frumentum", "fruor", "frustra", "fuga", "fugio", "fugo", "fundo", "funus", "furor",  "gaudeo", "gaudium", "gens", "genus", "gero", "gigno", "gladius", "gloria", "gradus", "gratia", "gratus", "grauis",  "habeo", "haud", "hic", "hic", "hiems", "hodie", "homo", "honestus", "honor", "hora", "hortor", "hospes", "hostis", "huc", "humanus", "humus", "iaceo", "iacio", "iam", "ibi", "ictus", "idem", "ideo", "igitur", "ignis", "ille", "illic", "illuc", "imago", "imperator", "imperium", "impero", "impetus", "impleo", "impono", "in", "incido", "incipio", "inde", "indico", "infero", "inferus", "ingenium", "ingens", "ingratus", "ingredior", "inimicus", "initium", "iniuria", "inquam", "instituo", "insula", "integer", "intellego", "intendo", "inter", "interficio", "interim", "interrogo", "intersum", "intra", "intro", "inuenio", "inuidia", "ipse", "ira", "irascor", "is", "iste", "ita", "itaque", "item", "iter", "iterum", "iubeo", "iudex", "iudicium", "iudico", "iugum", "iungo", "iuro", "ius", "iustus", "iuuenis", "iuuo",  "labor", "laboro", "lacrima", "laedo", "laetus", "lapis", "lateo", "latus", "latus", "laudo", "laus", "legatus", "legio", "lego", "leuis", "lex", "liber", "liber", "libertas", "libet", "libido", "licet", "limen", "lingua", "littera", "litus", "locus", "longus", "loquor", "lumen", "luna", "lux",  "maestus", "magis", "magister", "magnitudo", "magnus", "maior", "malo", "malus", "maneo", "manus", "mare", "maritus", "mater", "materia", "maximus", "medius", "melior", "membrum", "memini", "memoria", "mens", "mensa", "mereo", "metuo", "metus", "meus", "miles", "mille", "minus", "miror", "misceo", "miser", "mitto", "modo", "modus", "moenia", "mollis", "moneo", "mons", "mora", "morbus", "morior", "moror", "mors", "mortalis", "mos", "moueo", "mox", "mulier", "multitudo", "multus", "mundus", "mundus", "munus", "murus", "muto",  "nam", "narro", "nascor", "natura", "natus", "nauis", "ne", "ne", "nec", "necesse", "necessitas", "nefas", "nego", "negotium", "nemo", "nemus", "neque", "nescio", "niger", "nihil", "nimius", "nisi", "ni", "nobilis", "noceo", "nolo", "nomen", "non", "nondum", "nos", "nosco", "noster", "notus", "nouus", "nox", "nudus", "nullus", "num", "numen", "numerus", "numquam", "nunc", "nuntius",  "ob", "occido", "occupo", "occurro", "oculus", "odi", "odium", "offero", "ofﬁcium", "olim", "omnis", "onus", "opera", "oportet", "oppidum", "ops", "optimus", "opto", "opus", "oratio", "orbis", "ordo", "orior", "oro", "os", "os", "ostendo", "otium",  "paene", "par", "parco", "parens", "pareo", "pario", "paro", "pars", "parum", "paruus", "pateo", "pater", "patior", "patria", "pauci", "paulo", "pauper", "pax", "pecco", "pectus", "pecunia", "pecus", "pello", "pendo", "per", "perdo", "pereo", "pergo", "periculum", "permitto", "perpetuus", "pertineo", "peruenio", "pes", "peto", "pietas", "pius", "placeo", "plebs", "plenus", "plerusque", "plurimus", "plus", "poena", "poeta", "pondus", "pono", "pontus", "populus", "porta", "porto", "posco", "possum", "post", "postea", "posterus", "postquam", "potens", "potestas", "potis", "praebeo", "praeceptum", "praecipio", "praeda", "praemium", "praesens", "praesidium", "praesto", "praeter", "praeterea", "praetor", "precor", "premo", "pretium", "prex", "primus", "princeps", "principium", "prior", "priuatus", "pro", "probo", "procedo", "procul", "prodo", "proelium", "proficiscor", "prohibeo", "promitto", "prope", "propior", "propero", "propono", "proprius", "propter", "prosum", "protinus", "prouincia", "publicus", "pudor", "puella", "puer", "pugna", "pugno", "pulcher", "puto",  "qua", "quaero", "qualis", "quam", "quamquam", "quamuis", "quando", "quantum", "quantus", "quare", "quasi", "quattuor", "que", "quemadmodum", "queror", "qui", "quia", "quicumque", "quid", "quidam", "quidem", "quiesco", "quin", "quippe", "quis", "quisquam", "quisque", "quisquis", "quo", "quomodo", "quondam", "quoniam", "quoque", "quotiens",  "rapio", "rarus", "ratio", "recedo", "recens", "recipio", "rectus", "reddo", "redeo", "refero", "regio", "regius", "regnum", "rego", "relinquo", "reliquus", "reor", "reperio", "repeto", "res", "respicio", "respondeo", "retineo", "reus", "reuerto", "reuoco", "rex", "rideo", "ripa", "rogo", "rumpo", "rursus", "rus",  "sacer", "sacerdos", "saeculum", "saepe", "saeuus", "salus", "sanctus", "sanguis", "sanus", "sapiens", "sapientia", "satis", "sat", "saxum", "scelus", "scientia", "scilicet", "scio", "scribo", "secundus", "securus", "sed", "sedeo", "sedes", "semel", "semper", "senatus", "senex", "sensus", "sententia", "sentio", "sepulcrum", "sequor", "sermo", "seruio", "seruo", "seruus", "seu", "si", "sic", "sicut", "sidus", "signum", "silua", "similis", "simul", "sine", "singuli", "sino", "sinus", "siue", "socius", "sol", "soleo", "solus", "soluo", "somnus", "sono", "soror", "sors", "spargo", "spatium", "species", "specto", "spero", "spes", "spiritus", "statim", "statuo", "stella", "sto", "studeo", "studium", "sub", "subeo", "subito", "sui", "sum", "summus", "sumo", "super", "superbus", "supero", "supersum", "superus", "supplicium", "supra", "surgo", "suscipio", "sustineo", "suus",  "taceo", "talis", "tam", "tamen", "tamquam", "tandem", "tango", "tantus", "tardus", "tectum", "tego", "tellus", "telum", "tempestas", "templum", "tempus", "tendo", "tenebrae", "teneo", "tener", "tento", "tergum", "terra", "terreo", "tertius", "testis", "timeo", "timor", "tollo", "tot", "totus", "trado", "traho", "transeo", "tres", "tribunus", "tristis", "tu", "tum", "turba", "turbo", "turpis", "tutus", "tuus",  "ubi", "ullus", "ultimus", "ultra", "umbra", "umquam", "unda", "unde", "undique", "unus", "urbs", "usque", "usus", "ut", "uterque", "utilis", "utor", "utrum", "uxor",  "uaco", "uacuus", "uagus", "ualeo", "ualidus", "uanus", "uarius", "uates", "ue", "ueho", "uel", "uelut", "uenio", "uentus", "uerbum", "uereor", "uero", "uerto", "uerus", "uester", "uestigium", "uestis", "ueto", "uetus", "uia", "uicinus", "uictor", "uictoria", "uideo", "uinco", "uinculum", "uinum", "uir", "uirgo", "uirtus", "uis", "uita", "uitium", "uito", "uiuo", "uix", "uoco", "uolo", "uolucer", "uoluntas", "uoluptas", "uos", "uotum", "uox", "uulgus", "uulnus", "uultus"]}}
            ]]
        # Register a new token extension to flag core vocabulary
        Token.set_extension("is_dcc_core", force=True, default=False)
        self.matcher = Matcher(vocab)
        self.matcher.add("DCC_CORE", patterns)

    def __call__(self, doc):
        # This method is invoked when the component is called on a Doc
        doc.spans["dcc_core"] = []
        matches = self.matcher(doc)
        spans = []  # Collect the matched spans here
        for match_id, start, end in matches:
            spans.append(doc[start:end])
            doc.spans["dcc_core"].append(Span(doc, start, end, "CORE"))
            for span in spans:
                for token in span:
                    token._.is_dcc_core = True  # Mark token as bad HTML
        return doc

st.title("LatinCy DCC Core Visualizer")

# Using object notation
model_selectbox = st.sidebar.selectbox(
    "Choose model:",
    ("la_core_web_lg", "la_core_web_md", "la_core_web_sm")
)

nlp = spacy.load(model_selectbox)

# Add component to pipeline
nlp.add_pipe("dcc_core", last=True)

text = st.text_area(
    "Enter some text to analyze (max 100 tokens)", value=default_text, height=200
)
if st.button("Analyze"):
    doc = nlp(text.replace("v", "u").replace("V", "U"))
    len_doc = len([token for token in doc if not token.is_punct])
    len_dcc = len(doc.spans["dcc_core"])
    st.text(f"Analyzed {len_doc} tokens with {len_dcc} core vocabulary items ({round((len_dcc/len_doc)*100, 2)}%) ")
    visualize_spans(doc, spans_key="dcc_core", show_table=False, displacy_options={"colors": {"CORE": "#09a3d5"}})
