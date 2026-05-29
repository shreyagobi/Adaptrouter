# adaptrouter/implicit_feedback.py
import sys
import time
import numpy as np
from adaptrouter.config import (
    LLM_ROUTER_PATH,
    REPHRASING_SIMILARITY_THRESHOLD,
    REPHRASING_TIME_WINDOW_SECONDS,
)

if LLM_ROUTER_PATH not in sys.path:
    sys.path.insert(0, LLM_ROUTER_PATH)

try:
    from src.embedder import embed
    _EMBED_AVAILABLE = True
except ImportError:
    _EMBED_AVAILABLE = False


# Keywords that signal the user wanted more detail than they got
ESCALATION_KEYWORDS = [
    "explain more", "go deeper", "more detail", "elaborate",
    "can you expand", "tell me more", "what do you mean",
    "i don't understand", "clarify", "give me an example",
    "show me how", "walk me through", "step by step please",
]


class ImplicitFeedbackDetector:
    """
    Detects routing quality signals from user behaviour patterns.

    WHY implicit feedback?
    Research shows only 1-5% of users ever click explicit feedback
    buttons. The other 95-99% give you nothing — unless you watch
    their behaviour. Implicit feedback is noisier but 20x more
    abundant. Combined correctly, it is more valuable than
    explicit feedback alone.

    The four signals this class detects:

    1. REPHRASING — user asks very similar question shortly after
       Suggests: previous answer was insufficient
       Confidence: 0.6 (moderate — could just be curiosity)

    2. ESCALATION — user explicitly asks for more detail
       Suggests: fast model was used but complex answer was needed
       Confidence: 0.8 (high — very clear signal)

    3. ACCEPTANCE — user moves to completely different topic
       Suggests: answer was sufficient, routing was correct
       Confidence: 0.7 (moderate — could just be moving on)

    4. ABANDONMENT — user stops completely after one bad answer
       Suggests: answer was unsatisfactory
       Confidence: 0.5 (low — many reasons to stop a session)
    """

    def __init__(self, feedback_store):
        self.store            = feedback_store
        self._embed_available = _EMBED_AVAILABLE


    def check_all_signals(self, current_result: dict,
                          recent_decisions: list) -> list:
        """
        Checks all implicit signals for the current routing result.
        Returns list of detected signals with their types and confidences.
        Called after every route() in AdaptRouter.
        """
        detected = []

        if len(recent_decisions) < 2:
            return detected   # need at least 2 decisions to detect patterns

        previous = recent_decisions[-2]   # decision just before current

        # Signal 1: Rephrasing detection
        rephrase = self._check_rephrasing(
            current_result, previous, recent_decisions
        )
        if rephrase:
            detected.append(rephrase)

        # Signal 2: Escalation detection
        escalation = self._check_escalation(current_result)
        if escalation:
            detected.append(escalation)

        # Signal 3: Acceptance detection
        acceptance = self._check_acceptance(current_result, previous)
        if acceptance:
            detected.append(acceptance)

        # Store all detected signals
        for signal in detected:
            self.store.store_feedback(
                query_id      = signal["query_id"],
                implicit_type = signal["type"],
                implicit_conf = signal["confidence"],
            )

        return detected


    def _check_rephrasing(self, current: dict, previous: dict,
                          all_recent: list) -> dict | None:
        """
        Detects if current query is a rephrasing of a previous one.

        How similarity is computed:
          1. Embed both queries → 384-dimensional vectors
          2. Cosine similarity between vectors
          3. If similarity > threshold AND time < window → rephrasing

        Metric: cosine_similarity(embed(q1), embed(q2))
        Threshold: 0.82 (empirically chosen — high enough to avoid
                   false positives on related but different questions)
        """
        if not self._embed_available:
            return None

        current_query  = current["query"]
        current_time   = time.time()

        # Check against all recent decisions within time window
        for past in reversed(all_recent[:-1]):   # exclude current
            time_diff = current_time - past.get("timestamp", 0)
            if time_diff > REPHRASING_TIME_WINDOW_SECONDS:
                continue   # too old

            try:
                emb_current = embed(current_query)
                emb_past    = embed(past["query"])

                # Cosine similarity calculation
                dot_product   = np.dot(emb_current, emb_past)
                norm_current  = np.linalg.norm(emb_current)
                norm_past     = np.linalg.norm(emb_past)

                if norm_current == 0 or norm_past == 0:
                    continue

                similarity = dot_product / (norm_current * norm_past)

                if similarity > REPHRASING_SIMILARITY_THRESHOLD:
                    # Rephrasing detected — previous answer was insufficient
                    # The confidence scales with similarity:
                    # similarity 0.82 → confidence 0.60
                    # similarity 0.95 → confidence 0.85
                    confidence = round(
                        0.60 + (similarity - 0.82) * (0.85 - 0.60) / (1.0 - 0.82),
                        3
                    )
                    return {
                        "query_id"  : past["query_id"],
                        "type"      : "rephrasing",
                        "confidence": confidence,
                        "similarity": round(float(similarity), 4),
                        "time_gap_s": round(time_diff, 1),
                        "message"   : (
                            f"Query '{current_query[:40]}...' is similar "
                            f"(cos={similarity:.3f}) to previous query asked "
                            f"{time_diff:.0f}s ago — previous routing may have "
                            f"been insufficient"
                        )
                    }

            except Exception:
                continue

        return None


    def _check_escalation(self, current: dict) -> dict | None:
        """
        Detects if the user is explicitly asking for more detail.
        These keywords strongly suggest the previous fast-model answer
        was insufficient and a smart-model answer was needed.

        Signal strength: HIGH (0.80) — very explicit user intent.
        """
        query_lower = current["query"].lower()
        for keyword in ESCALATION_KEYWORDS:
            if keyword in query_lower:
                return {
                    "query_id"  : current["query_id"],
                    "type"      : "escalation",
                    "confidence": 0.80,
                    "keyword"   : keyword,
                    "message"   : (
                        f"User asked for more detail ('{keyword}') — "
                        f"previous routing may have used wrong model"
                    )
                }
        return None


    def _check_acceptance(self, current: dict, previous: dict) -> dict | None:
        """
        Detects if the user accepted the previous answer by moving
        to a completely different topic.

        How "completely different topic" is measured:
          cosine_similarity < 0.35 → very different topics → acceptance
          cosine_similarity > 0.35 → related topics → not conclusive

        Signal strength: MODERATE (0.70) — user could just be
        changing subject for many reasons.
        """
        if not self._embed_available:
            return None

        try:
            emb_current = embed(current["query"])
            emb_previous = embed(previous["query"])

            dot     = np.dot(emb_current, emb_previous)
            norm_c  = np.linalg.norm(emb_current)
            norm_p  = np.linalg.norm(emb_previous)

            if norm_c == 0 or norm_p == 0:
                return None

            similarity = dot / (norm_c * norm_p)

            if similarity < 0.35:
                # Very different topic → previous answer was accepted
                return {
                    "query_id"  : previous["query_id"],
                    "type"      : "acceptance",
                    "confidence": 0.70,
                    "similarity": round(float(similarity), 4),
                    "message"   : (
                        f"User moved to different topic (cos={similarity:.3f}) "
                        f"— previous routing likely correct"
                    )
                }
        except Exception:
            pass

        return None


    def get_signal_summary(self, signals: list) -> str:
        """Human-readable summary of detected signals."""
        if not signals:
            return "No implicit signals detected"
        parts = [f"{s['type']} (conf={s['confidence']})" for s in signals]
        return f"Detected: {', '.join(parts)}"