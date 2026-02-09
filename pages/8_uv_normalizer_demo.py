import streamlit as st
from typing import Dict, Any

from latincy_uv import UVNormalizerRules, NormalizationResult

st.set_page_config(page_title="U/V Normalizer Demo", layout="wide")
st.sidebar.header("U/V Normalizer Demo")

# Sample text (Seneca, Epistulae Morales 1)
SAMPLE_TEXT = """Ita fac, mi Lucili: vindica te tibi, et tempus quod adhuc aut auferebatur aut subripiebatur aut excidebat collige et serva. Persuade tibi hoc sic esse ut scribo: quaedam tempora eripiuntur nobis, quaedam subducuntur, quaedam effluunt."""

SAMPLE_TEXT_UONLY = SAMPLE_TEXT.replace("v", "u").replace("V", "U")

# HTML styling
HTML_GREEN = '<span style="color: #28a745; font-weight: bold">'
HTML_RED = '<span style="color: #dc3545; font-weight: bold">'
HTML_GREY = '<span style="color: #6c757d">'
HTML_END = "</span>"


@st.cache_resource
def get_normalizer() -> UVNormalizerRules:
    """Get cached normalizer instance."""
    return UVNormalizerRules()


def to_uonly(text: str) -> str:
    """Convert text to u-only spelling (all v -> u)."""
    return text.replace("v", "u").replace("V", "U")


def colorize_changes(original: str, normalized: str, reference: str = None) -> str:
    """Create HTML with color-coded changes."""
    result = [HTML_GREY]

    for i, (orig, norm) in enumerate(zip(original, normalized)):
        if orig != norm:
            if reference and i < len(reference):
                if norm == reference[i]:
                    result.append(f"{HTML_END}{HTML_GREEN}{norm}{HTML_END}{HTML_GREY}")
                else:
                    result.append(f"{HTML_END}{HTML_RED}{norm}{HTML_END}{HTML_GREY}")
            else:
                result.append(f"{HTML_END}{HTML_GREEN}{norm}{HTML_END}{HTML_GREY}")
        else:
            result.append(orig)

    result.append(HTML_END)
    return "".join(result)


def calculate_metrics(source: str, normalized: str, reference: str) -> Dict[str, Any]:
    """Calculate accuracy metrics."""
    min_len = min(len(source), len(normalized), len(reference))

    total_uv = 0
    correct = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    changes_needed = 0
    changes_made = 0

    for i in range(min_len):
        src, norm, ref = source[i], normalized[i], reference[i]

        if src.lower() in ("u", "v"):
            total_uv += 1

            needed = src != ref
            made = src != norm

            if needed:
                changes_needed += 1
            if made:
                changes_made += 1

            if norm == ref:
                correct += 1
                if needed and made:
                    true_positives += 1
            else:
                if not needed and made:
                    false_positives += 1
                elif needed and not made:
                    false_negatives += 1

    accuracy = correct / total_uv if total_uv > 0 else 1.0
    precision = true_positives / changes_made if changes_made > 0 else 1.0
    recall = true_positives / changes_needed if changes_needed > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "total_uv": total_uv,
        "correct": correct,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "changes_needed": changes_needed,
        "changes_made": changes_made,
    }


def show_rule_details(result: NormalizationResult):
    """Show detailed rule application information."""
    if not result.changes:
        st.info("No changes made - text already normalized")
        return

    st.subheader(f"Rule Applications ({len(result.changes)} changes)")

    by_rule: Dict[str, list] = {}
    for change in result.changes:
        if change.rule not in by_rule:
            by_rule[change.rule] = []
        by_rule[change.rule].append(change)

    for rule, changes in sorted(by_rule.items()):
        with st.expander(f"**{rule}** ({len(changes)} changes)"):
            for change in changes[:10]:
                st.markdown(
                    f"- `{change.context}` -> {change.original} -> **{change.normalized}**"
                )
            if len(changes) > 10:
                st.markdown(f"*...and {len(changes) - 10} more*")


normalizer = get_normalizer()

st.title("Latin U/V Normalizer")
st.markdown(
    "Rule-based U/V normalization for Latin using "
    "[latincy-uv](https://github.com/diyclassics/latincy-uv). "
    "Converts consonantal 'u' to 'v' and vocalic 'v' to 'u'."
)

tab1, tab2, tab3 = st.tabs(["Normalize", "Evaluate", "About"])

# === NORMALIZE TAB ===
with tab1:
    st.markdown(
        "Enter Latin text with u-only spelling and normalize it "
        "to proper u/v distinction."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        text = st.text_area(
            "Input text (u-only spelling):",
            value=SAMPLE_TEXT_UONLY,
            height=200,
            help="Enter Latin text with 'u' for both vowels and consonants",
        )

    show_details = st.checkbox("Show rule details", value=False)

    if st.button("Normalize", type="primary"):
        if not text.strip():
            st.warning("Please enter some text")
        else:
            result = normalizer.normalize_detailed(text)

            with col2:
                st.markdown("**Normalized text:**")
                colored = colorize_changes(text, result.normalized)
                st.markdown(colored, unsafe_allow_html=True)

                n_changes = len(result.changes)
                st.markdown(f"*{n_changes} change{'s' if n_changes != 1 else ''} made*")

            if show_details:
                show_rule_details(result)

# === EVALUATE TAB ===
with tab2:
    st.markdown(
        "Enter correctly normalized text to evaluate accuracy. "
        "The system will convert it to u-only form, normalize it, "
        "and compare against your reference."
    )

    reference = st.text_area(
        "Reference text (correctly normalized):",
        value=SAMPLE_TEXT,
        height=200,
        help="Enter Latin text with correct U/V distinction",
    )

    if st.button("Evaluate", type="primary"):
        if not reference.strip():
            st.warning("Please enter reference text")
        else:
            source = to_uonly(reference)
            normalized = normalizer.normalize(source)

            metrics = calculate_metrics(source, normalized, reference)

            st.subheader("Evaluation Results")
            colored = colorize_changes(source, normalized, reference)
            st.markdown(colored, unsafe_allow_html=True)

            st.markdown(
                f"**Legend:** {HTML_GREEN}Correct{HTML_END} &middot; "
                f"{HTML_RED}Incorrect{HTML_END} &middot; "
                f"{HTML_GREY}Unchanged{HTML_END}",
                unsafe_allow_html=True,
            )

            st.subheader("Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
            col2.metric("Precision", f"{metrics['precision']:.1%}")
            col3.metric("Recall", f"{metrics['recall']:.1%}")
            col4.metric("F1 Score", f"{metrics['f1']:.1%}")

            with st.expander("Detailed Statistics"):
                st.markdown(f"""
                - **Total U/V characters:** {metrics['total_uv']}
                - **Correct:** {metrics['correct']}
                - **Changes needed:** {metrics['changes_needed']}
                - **Changes made:** {metrics['changes_made']}
                - **True positives:** {metrics['true_positives']}
                - **False positives:** {metrics['false_positives']}
                - **False negatives:** {metrics['false_negatives']}
                """)

# === ABOUT TAB ===
with tab3:
    st.markdown("""
    ## About

    This tool normalizes Latin U/V spelling:

    - **Consonantal 'u' -> 'v'**: e.g., *ueni* -> *veni*, *uia* -> *via*
    - **Vocalic 'v' -> 'u'**: e.g., *lvna* -> *luna*

    ### Rule Categories

    The normalizer applies rules in priority order:

    1. **QU digraph**: `qu` always stays as `qu` (*quod*, *aqua*)
    2. **NGU/GU digraph**: `ngu`/`gu` before vowel stays `u` (*lingua*, *sanguis*)
    3. **Word exceptions**: Morphological exceptions (*cui*, *sua*, *fuit*)
    4. **Perfect tense**: U-perfect verb forms (*fuit*, *potuit*, *fuisse*)
    5. **Double-u patterns**: Context-dependent (*servus*, *fluvius*, *iuvat*)
    6. **Initial position**: Before vowel -> `v`, before consonant -> `u`
    7. **Intervocalic**: Between vowels -> `v` (*novus*, *brevis*)
    8. **Before consonant**: -> `u` (*scriptum*, *causa*)
    9. **Word-final**: -> `u` (*tu*, *cum*)
    10. **Post-consonant**: Before vowel -> `v` (*silva*, *servo*)

    ### Accuracy

    - **Curated test set (100 sentences):** 100%
    - **UD Latin treebanks (~1800 sentences):** ~98%

    ### Source

    [GitHub: diyclassics/latincy-uv](https://github.com/diyclassics/latincy-uv)
    """)
