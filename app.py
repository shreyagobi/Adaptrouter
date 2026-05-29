# dashboard/app.py  — COMPLETE CORRECTED + IMPROVED VERSION
# Fixes applied:
#   1.  str | None  → Optional[str]  (Python 3.9 compat)
#   2.  Session state init moved BEFORE helper functions
#   3.  query_input_val synced after submit so textarea doesn't re-populate
#   4.  HTML-escape answer before injecting into answer-box div (XSS fix)
#   5.  Substring match replaced with Jaccard similarity (adaptive query matching)
#   6.  SHAP classifier cached with @st.cache_resource (no per-query disk load)
#   7.  Silent except: pass tightened to specific AttributeError in banner
#   8.  Double-submission guard — pending cleared atomically before processing
#   9.  traceback moved to top-level imports
#  10.  Jaccard fuzzy matching with configurable threshold
#  11.  feedback_chain on history items for iterative improvement tracking
#  12.  History list capped at 50 entries in memory

import sys
import os
import time
import html
import traceback
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from adaptrouter.config import LLM_ROUTER_PATH
if LLM_ROUTER_PATH not in sys.path:
    sys.path.insert(0, LLM_ROUTER_PATH)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AdaptRouter — Smart Assistant",
    page_icon="→",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

  :root {
    --bg-base:       #0a0e17;
    --bg-card:       #111827;
    --bg-deep:       #0d1117;
    --border-dim:    #1f2937;
    --border-mid:    #30363d;
    --text-muted:    #8b949e;
    --text-body:     #c9d1d9;
    --accent-blue:   #3b82f6;
    --accent-green:  #22c55e;
    --accent-purple: #a855f7;
    --accent-amber:  #f59e0b;
    --accent-red:    #ef4444;
    --pill-fast-bg:  #1a3d2b;
    --pill-fast-bd:  #2d7a4f;
    --pill-fast-tx:  #5dbb85;
    --pill-smart-bg: #2d1a3d;
    --pill-smart-bd: #6b21a8;
    --pill-smart-tx: #c084fc;
  }

  .answer-box {
    background: var(--bg-card);
    border: 1px solid var(--border-dim);
    border-left: 3px solid var(--accent-blue);
    border-radius: 10px;
    padding: 18px 22px;
    margin: 10px 0 16px;
    font-family: 'Inter', sans-serif;
    font-size: 14.5px;
    line-height: 1.8;
    color: var(--text-body);
    white-space: pre-wrap;
    word-break: break-word;
  }
  .answer-box.adapted {
    border-left-color: var(--accent-green);
  }

  .routing-row {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin: 8px 0 14px;
  }
  .routing-pill {
    background: #161d2b;
    border: 1px solid var(--border-dim);
    border-radius: 20px;
    padding: 3px 11px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-muted);
  }
  .routing-pill.fast  { background: var(--pill-fast-bg);  border-color: var(--pill-fast-bd);  color: var(--pill-fast-tx);  }
  .routing-pill.smart { background: var(--pill-smart-bg); border-color: var(--pill-smart-bd); color: var(--pill-smart-tx); }
  .routing-pill.adapted { background: #1a3320; border-color: var(--accent-green); color: var(--accent-green); }

  .shap-box {
    background: var(--bg-deep);
    border: 1px solid var(--border-mid);
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--text-muted);
    margin: 6px 0;
    line-height: 1.65;
  }
  .word-chip {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    margin: 3px 3px;
  }
  .chip-complex { background:#2d1b4e; color:#c084fc; border:1px solid #6b21a8; }
  .chip-simple  { background:#1a2e1a; color:#4ade80; border:1px solid #166534; }
  .chip-neutral { background:#1e2128; color:#6b7280; border:1px solid #374151; }

  .learn-banner {
    background: #0f2d1e;
    border: 1px solid #2d7a4f;
    border-radius: 8px;
    padding: 9px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #a7f3d0;
    margin: 6px 0 10px;
  }

  .adapted-banner {
    background: #0f2d1e;
    border: 1px solid var(--accent-green);
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 12px;
    color: var(--accent-green);
    margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace;
  }

  .feedback-prompt-box {
    background: #16101f;
    border: 1px solid #4b1d6b;
    border-radius: 10px;
    padding: 14px 18px 6px;
    margin: 10px 0 4px;
  }
  .feedback-prompt-label {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    color: #d1b3ff;
    margin-bottom: 6px;
  }

  .section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 16px 0 6px;
  }

  .history-answer-box {
    background: var(--bg-deep);
    border: 1px solid var(--border-mid);
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    line-height: 1.7;
    color: var(--text-muted);
    white-space: pre-wrap;
    word-break: break-word;
    margin-top: 6px;
  }

  .feedback-note {
    background: #16101f;
    border-left: 2px solid var(--accent-purple);
    border-radius: 4px;
    padding: 6px 12px;
    font-size: 12px;
    color: #c4b5fd;
    margin-top: 6px;
    font-style: italic;
  }

  /* FIX 11 — feedback chain entry */
  .feedback-chain-entry {
    background: #0f1a2b;
    border-left: 2px solid var(--accent-blue);
    border-radius: 4px;
    padding: 5px 11px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    color: #7eb3ff;
    margin-top: 4px;
  }

  /* FIX 13 — keyboard hint */
  .kbd-hint {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-muted);
    opacity: 0.65;
    margin-top: 2px;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2 — SESSION STATE init BEFORE any helper that reads st.session_state
# ─────────────────────────────────────────────────────────────────────────────
_SS_DEFAULTS: dict = {
    "history"        : [],
    "pending"        : None,
    "feedback_counts": {"helpful": 0, "unhelpful": 0, "total": 0},
    "retrain_events" : [],
    "show_fb_box"    : False,
    "fb_note_draft"  : "",
    "query_input_val": "",
    "processing"     : False,   # FIX 8 — double-submit guard
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AdaptRouter…")
def load_router():
    try:
        from adaptrouter import AdaptRouter
        return AdaptRouter(domain="assistant"), None
    except Exception as exc:
        return None, str(exc)


@st.cache_resource(show_spinner=False)   # FIX 6 — classifier cached once
def _load_shap_clf():
    try:
        import joblib
        clf_path = os.path.join(LLM_ROUTER_PATH, "models", "router_classifier.pkl")
        if os.path.exists(clf_path):
            return joblib.load(clf_path)
    except Exception:
        pass
    return None


router, err = load_router()
if router is None:
    st.error(f"Could not load AdaptRouter: {err}")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS  (defined after session state is guaranteed to exist)
# ─────────────────────────────────────────────────────────────────────────────

# FIX 5 / Improvement 10 — configurable Jaccard threshold
_JACCARD_THRESHOLD = 0.55


def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings (0.0–1.0)."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _get_prior_feedback_context(query: str) -> Optional[str]:   # FIX 1
    """
    Return the most recent negative-feedback note for a sufficiently similar
    historical query, or None if no match exceeds the Jaccard threshold.
    """
    q           = query.strip()
    best_score  = 0.0
    best_note   = None
    for item in st.session_state.history:
        if item.get("feedback") is not False:
            continue
        note = item.get("feedback_note", "").strip()
        if not note:
            continue
        prev_q = item["result"].get("query", "").strip()
        score  = _jaccard(q, prev_q)
        if score >= _JACCARD_THRESHOLD and score > best_score:
            best_score = score
            best_note  = note
    return best_note


def _build_adaptive_prompt(query: str, feedback_note: str) -> str:
    return (
        f"A user previously asked: \"{query}\"\n"
        f"They found the answer unhelpful and said: \"{feedback_note}\"\n\n"
        f"Please give a significantly improved answer that directly addresses "
        f"their concern. Be clearer, more concrete, and correct any issues "
        f"they mentioned.\n\n"
        f"Question: {query}"
    )


def _run_shap(query: str) -> Optional[dict]:   # FIX 6
    clf = _load_shap_clf()
    if clf is None:
        return None
    try:
        from adaptrouter.explainer import RoutingExplainer
        return RoutingExplainer(classifier=clf).explain(query)
    except Exception:
        return None


def _handle_feedback(
    router_obj,
    pending_item: dict,
    was_helpful: bool,
    rating: Optional[int] = None,          # FIX 1
    feedback_note: Optional[str] = None,
) -> dict:
    """Record feedback, update counters, archive result, cap history."""

    query_id = pending_item["query_id"]

    # FIX 8 — clear pending atomically BEFORE any side-effectful call
    st.session_state.pending       = None
    st.session_state.show_fb_box   = False
    st.session_state.fb_note_draft = ""

    try:
        fb_result = router_obj.feedback(
            query_id    = query_id,
            was_helpful = was_helpful,
            rating      = rating,
            **({"feedback_note": feedback_note} if feedback_note else {}),
        )
    except TypeError:
        try:
            fb_result = router_obj.feedback(
                query_id    = query_id,
                was_helpful = was_helpful,
                rating      = rating,
            )
        except Exception:
            fb_result = {"retrain_triggered": False, "examples_until_retrain": "?"}
    except Exception:
        fb_result = {"retrain_triggered": False, "examples_until_retrain": "?"}

    # Update session counters
    c = st.session_state.feedback_counts
    c["total"] += 1
    if was_helpful:
        c["helpful"] += 1
    else:
        c["unhelpful"] += 1

    # FIX 11 — maintain feedback_chain for iterative improvement
    chain = list(pending_item.get("feedback_chain", []))
    if feedback_note:
        chain.append({
            "note"     : feedback_note,
            "helpful"  : was_helpful,
            "rating"   : rating,
            "timestamp": time.strftime("%H:%M:%S"),
        })

    # FIX 12 — archive and cap at 50
    st.session_state.history.insert(0, {
        **pending_item,
        "feedback"      : was_helpful,
        "feedback_label": "helpful" if was_helpful else "unhelpful",
        "rating"        : rating,
        "feedback_note" : feedback_note or "",
        "feedback_chain": chain,
    })
    st.session_state.history = st.session_state.history[:50]

    if fb_result.get("retrain_triggered"):
        st.session_state.retrain_events.append(time.strftime("%H:%M:%S"))

    return fb_result


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## AdaptRouter — Self-Improving Assistant")
st.caption(
    "Routes your question to the right AI model · Answers it · "
    "Explains why · Learns from your feedback"
)

# Learning banner — FIX 7: catch only AttributeError, not bare except
counts   = st.session_state.feedback_counts
fb_total = counts["total"]
if fb_total > 0:
    try:
        new_count = (
            router._feedback_store.count_new_labelled()
            if router._feedback_store else 0
        )
        remaining = max(0, 20 - new_count)
        msg = (
            "✓ Retraining triggered — router is improving from your feedback!"
            if remaining == 0
            else (
                f'{fb_total} feedback signal{"s" if fb_total != 1 else ""} collected · '
                f'{remaining} more needed to trigger automatic retraining'
            )
        )
        st.markdown(f'<div class="learn-banner">{msg}</div>',
                    unsafe_allow_html=True)
    except AttributeError:   # FIX 7
        pass

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# QUERY INPUT
# ─────────────────────────────────────────────────────────────────────────────
with st.form("query_form", clear_on_submit=False):
    query = st.text_area(
        "Your question",
        value       = st.session_state.query_input_val,
        placeholder = (
            "Simple → fast model:  'What is the capital of France?'\n"
            "Complex → smart model: 'Explain how attention works in transformers'"
        ),
        height = 90,
        key    = "query_input",
    )
    # FIX 13 — keyboard shortcut hint
    st.markdown(
        '<div class="kbd-hint">Tip: Ctrl+Enter to submit</div>',
        unsafe_allow_html=True,
    )
    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        submitted = st.form_submit_button(
            "Ask →", type="primary", use_container_width=True
        )
    with col_btn2:
        clear = st.form_submit_button("Clear", use_container_width=True)

# FIX 3 — Clear ONLY resets input; never touches pending or history
if clear:
    st.session_state.query_input_val = ""
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS QUERY
# ─────────────────────────────────────────────────────────────────────────────
if submitted and query.strip() and not st.session_state.processing:

    st.session_state.processing      = True   # FIX 8
    st.session_state.query_input_val = query.strip()   # FIX 3

    # Auto-archive previous unrated result
    if st.session_state.pending is not None:
        st.session_state.history.insert(0, {
            **st.session_state.pending,
            "feedback"      : None,
            "feedback_label": "skipped",
            "rating"        : None,
            "feedback_note" : "",
            "feedback_chain": [],
        })
        st.session_state.history       = st.session_state.history[:50]
        st.session_state.pending       = None
        st.session_state.show_fb_box   = False
        st.session_state.fb_note_draft = ""

    # Adaptive context lookup — FIX 5 / improvement 10
    prior_note = _get_prior_feedback_context(query.strip())
    adapted    = prior_note is not None
    eff_query  = (
        _build_adaptive_prompt(query.strip(), prior_note)
        if adapted else query.strip()
    )

    with st.spinner("Routing and generating answer…"):
        try:
            result          = router.route(eff_query)
            result["query"] = query.strip()   # always store original user query
            result["adapted"] = adapted

            shap_explanation = _run_shap(query.strip())   # FIX 6

            st.session_state.pending = {
                "result"        : result,
                "shap"          : shap_explanation,
                "timestamp"     : time.strftime("%H:%M:%S"),
                "query_id"      : result["query_id"],
                "feedback_chain": [],
            }
            st.session_state.show_fb_box   = False
            st.session_state.fb_note_draft = ""

        except Exception as exc:
            st.error(f"Routing error: {exc}")
            st.code(traceback.format_exc())   # FIX 9

    st.session_state.processing = False   # FIX 8 — release guard


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY CURRENT RESULT
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.pending is not None:
    p       = st.session_state.pending
    result  = p["result"]
    shap    = p["shap"]
    is_fast = "8b" in result.get("model_used", "").lower()
    adapted = result.get("adapted", False)

    model_label = (
        "⚡ Fast Model (Llama 3.1 8B)" if is_fast
        else "🧠 Smart Model (Llama 3.3 70B)"
    )
    st.markdown(f"### {model_label}")

    if adapted:
        st.markdown(
            '<div class="adapted-banner">'
            '♻️  Adapted answer — improved based on your previous feedback'
            '</div>',
            unsafe_allow_html=True,
        )

    # FIX 4 — HTML-escape before injecting into div (XSS prevention)
    answer   = result.get("answer", "No answer received.")
    safe_ans = html.escape(answer)
    box_cls  = "answer-box adapted" if adapted else "answer-box"

    if result.get("error") == "realtime_query":
        st.warning(answer)
    else:
        st.markdown(
            f'<div class="{box_cls}">{safe_ans}</div>',
            unsafe_allow_html=True,
        )

    # Routing pills
    label_cls  = "fast" if is_fast else "smart"
    conf_pct   = round(result["confidence"] * 100, 1)
    p_simp     = round(result["p_simple"]   * 100, 1)
    p_comp     = round(result["p_complex"]  * 100, 1)
    trust_txt  = "trusted" if result["trusted"] else "below threshold → smart"
    tokens     = result.get("total_tokens", 0)
    lat        = result.get("latency_s", 0)
    adapt_pill = (
        '<span class="routing-pill adapted">♻ adapted</span>' if adapted else ""
    )
    st.markdown(
        f'<div class="routing-row">'
        f'<span class="routing-pill {label_cls}">{result["label"].upper()}</span>'
        f'<span class="routing-pill">conf {conf_pct}%</span>'
        f'<span class="routing-pill">P(simple) {p_simp}%</span>'
        f'<span class="routing-pill">P(complex) {p_comp}%</span>'
        f'<span class="routing-pill">{trust_txt}</span>'
        f'<span class="routing-pill">{lat}s</span>'
        f'<span class="routing-pill">{tokens} tok</span>'
        f'<span class="routing-pill">{p["timestamp"]}</span>'
        f'{adapt_pill}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # SHAP explanation
    if shap and "error" not in shap:
        with st.expander("Why this model? (SHAP routing explanation)"):
            shap_summary = html.escape(shap.get("shap_summary", ""))   # FIX 4
            st.markdown(
                f'<div class="shap-box">{shap_summary}</div>',
                unsafe_allow_html=True,
            )
            word_scores = shap.get("word_scores", [])
            if word_scores:
                st.markdown("**Word contributions to routing decision:**")
                chips = ""
                for w in word_scores[:10]:
                    d   = w["direction"]
                    cls = (
                        "chip-complex" if d == "→ complex"
                        else "chip-simple" if d == "→ simple"
                        else "chip-neutral"
                    )
                    safe_word = html.escape(w["word"])   # FIX 4
                    chips += (
                        f'<span class="word-chip {cls}">'
                        f'{safe_word} ({w["score"]:+.3f})'
                        f'</span>'
                    )
                st.markdown(chips, unsafe_allow_html=True)
                st.caption(
                    "Purple = pushed toward smart model  |  "
                    "Green = pushed toward fast model  |  "
                    "Gray = neutral"
                )
            for f in shap.get("decision_factors", []):
                words = ", ".join(f.get("words", []))
                st.caption(
                    f"{f.get('factor','').title()}: [{words}] "
                    f"→ {f.get('effect','')} "
                    f"(strength={f.get('strength','')})"
                )

    # ── FEEDBACK ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">Feedback</div>',
                unsafe_allow_html=True)
    st.caption("Your feedback trains the router to make better decisions.")

    fc1, fc2, fc3, fc4 = st.columns([1, 1, 1, 1])

    with fc1:
        if st.button("👍  Helpful", type="primary",
                     use_container_width=True, key="fb_yes"):
            fb_res    = _handle_feedback(router, p, was_helpful=True)
            remaining = fb_res.get("examples_until_retrain", "?")
            if fb_res.get("retrain_triggered"):
                st.success("Retraining triggered! Router is improving.")
            else:
                st.success(f"Thanks! {remaining} more needed to retrain.")
            st.rerun()

    with fc2:
        if st.button("👎  Not helpful",
                     use_container_width=True, key="fb_no"):
            st.session_state.show_fb_box = True
            st.rerun()

    with fc3:
        rating_val = st.select_slider(
            "Rate",
            options          = [1, 2, 3, 4, 5],
            value            = 3,
            label_visibility = "collapsed",
            key              = "rating_slider",
        )

    with fc4:
        if st.button("Submit rating",
                     use_container_width=True, key="fb_rate"):
            fb_res    = _handle_feedback(
                router, p,
                was_helpful = rating_val >= 3,
                rating      = rating_val,
            )
            remaining = fb_res.get("examples_until_retrain", "?")
            if fb_res.get("retrain_triggered"):
                st.success("Retraining triggered! Router is improving.")
            else:
                st.info(f"Rating {rating_val}/5 saved. {remaining} more needed.")
            st.rerun()

    # ── NEGATIVE FEEDBACK TEXT BOX ───────────────────────────────────────────
    if st.session_state.show_fb_box:
        st.markdown(
            '<div class="feedback-prompt-box">'
            '<div class="feedback-prompt-label">'
            '💬 What was wrong with the answer? How can it be improved?'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        note = st.text_area(
            label            = "Feedback note",
            value            = st.session_state.fb_note_draft,
            placeholder      = (
                "e.g. 'The explanation was too abstract. "
                "Please give a step-by-step example.'"
            ),
            height           = 90,
            key              = "fb_note_input",
            label_visibility = "collapsed",
        )
        st.session_state.fb_note_draft = note

        col_send, col_skip = st.columns([2, 1])
        with col_send:
            if st.button("Submit feedback →", type="primary",
                         use_container_width=True, key="fb_note_submit"):
                fb_res    = _handle_feedback(
                    router, p,
                    was_helpful   = False,
                    feedback_note = note.strip() or None,
                )
                remaining = fb_res.get("examples_until_retrain", "?")
                if fb_res.get("retrain_triggered"):
                    st.success("Retraining triggered! Router is improving.")
                else:
                    st.info(
                        f"Feedback saved. {remaining} more signals needed. "
                        "Ask the same question again for an improved answer."
                    )
                st.rerun()
        with col_skip:
            if st.button("Skip note", use_container_width=True,
                         key="fb_note_skip"):
                fb_res    = _handle_feedback(
                    router, p,
                    was_helpful   = False,
                    feedback_note = None,
                )
                remaining = fb_res.get("examples_until_retrain", "?")
                st.info(f"Noted. {remaining} more signals needed.")
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# CONVERSATION HISTORY  (FIX 12 — list capped at 50 on write; show latest 8)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown('<div class="section-label">Previous questions</div>',
                unsafe_allow_html=True)

    for i, item in enumerate(st.session_state.history[:8]):
        r       = item["result"]
        fb      = item.get("feedback_label", "skipped")
        note    = item.get("feedback_note", "")
        chain   = item.get("feedback_chain", [])
        is_fast = "8b" in r.get("model_used", "").lower()
        icon    = "⚡" if is_fast else "🧠"
        fb_icon = "👍" if fb == "helpful" else "👎" if fb == "unhelpful" else "—"
        conf    = round(r["confidence"] * 100, 1)
        adapted = r.get("adapted", False)
        adapt_tag = " ♻" if adapted else ""

        with st.expander(
            f"{fb_icon} {icon} {r['query'][:65]}{adapt_tag} "
            f"[{r['label']} · {conf}%]"
        ):
            # FIX 4 — escape history answers
            safe_hist = html.escape(r["answer"][:600])
            ellipsis  = "…" if len(r["answer"]) > 600 else ""
            st.markdown(
                f'<div class="history-answer-box">{safe_hist}{ellipsis}</div>',
                unsafe_allow_html=True,
            )

            if note:
                safe_note = html.escape(note)
                st.markdown(
                    f'<div class="feedback-note">📝 Your note: {safe_note}</div>',
                    unsafe_allow_html=True,
                )

            # FIX 11 — show feedback chain when there are multiple iterations
            if len(chain) > 1:
                st.markdown("**Feedback chain for this question:**")
                for entry in chain:
                    h_icon    = "👍" if entry.get("helpful") else "👎"
                    r_str     = f" · {entry['rating']}/5" if entry.get("rating") else ""
                    safe_cn   = html.escape(entry.get("note", ""))
                    note_part = f" — {safe_cn}" if safe_cn else ""
                    st.markdown(
                        f'<div class="feedback-chain-entry">'
                        f'{h_icon} {entry.get("timestamp","")}{r_str}{note_part}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            st.caption(
                f"Model: {r['model_used']} · "
                f"Latency: {r['latency_s']}s · "
                f"Tokens: {r.get('total_tokens', 0)} · "
                f"Feedback: {fb} · "
                f"Time: {item.get('timestamp', '')}"
                + (" · adapted" if adapted else "")
            )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Learning Progress")

    c = st.session_state.feedback_counts
    st.metric("Total feedback", c["total"])
    st.metric("Helpful",        c["helpful"])
    st.metric("Not helpful",    c["unhelpful"])

    if router and router._feedback_store:
        try:
            stats     = router._feedback_store.get_stats()
            new_count = stats.get("new_for_retrain", 0)
            remaining = max(0, 20 - new_count)
            events    = stats.get("retraining_events", 0)
            rate      = stats.get("feedback_rate_pct", 0)
            health    = stats.get("feedback_health", "unknown")

            st.metric("Until retrain",     remaining)
            st.metric("Retraining events", events)

            color = (
                "green"  if health == "healthy"
                else "orange" if health == "low"
                else "red"
            )
            st.markdown(f"Feedback rate: :{color}[{rate}% — {health}]")
        except AttributeError:   # FIX 7
            pass

    if router and router._retrainer:
        try:
            acc = router._retrainer.get_current_accuracy()
            st.metric("Classifier accuracy", f"{acc:.1%}")
        except Exception:
            pass

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    1. You ask a question
    2. Router classifies in **~5ms** locally
    3. **Simple** → Fast model (8B) ~0.46s
    4. **Complex** → Smart model (70B) ~0.81s
    5. You rate the answer
    6. After **20 ratings** → auto-retrain
    7. Router improves for your domain

    **Adaptive answers**
    - Click 👎 and describe the issue
    - Ask the same question again
    - Answer is re-generated using your note
    - Similarity threshold: **55% word overlap**
    """)

    st.markdown("---")

    if st.session_state.retrain_events:
        st.markdown("### Retraining history")
        for t in st.session_state.retrain_events[-5:]:
            st.caption(f"Retrained at {t}")

    st.markdown("---")
    st.caption("AdaptRouter v0.1.0")
    st.caption("Self-improving LLM routing middleware")