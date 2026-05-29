# adaptrouter/core.py  — COMPLETE FIXED VERSION
import sys
import os
import uuid
import time
from adaptrouter.config import (
    LLM_ROUTER_PATH,
    GROQ_API_KEY,
    LONG_QUERY_WORD_LIMIT,
)

if LLM_ROUTER_PATH not in sys.path:
    sys.path.insert(0, LLM_ROUTER_PATH)

try:
    from src.router import RouterAgent as _BaseRouterAgent
    from src.embedder import embed as _embed
    _BASE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import llm-router ({e})")
    print(f"Check LLM_ROUTER_PATH in .env: {LLM_ROUTER_PATH}")
    _BASE_AVAILABLE = False


class AdaptRouter:
    """
    Self-improving LLM router.

    Every query goes through:
    1. Input validation (length check)
    2. Embedding + classification (5ms local)
    3. Confidence threshold check
    4. Route to fast or smart model
    5. Log decision to SQLite
    6. Check for implicit feedback signals
    7. Trigger retraining if threshold reached

    The user only needs to call:
      result = router.route(query)
      router.feedback(result["query_id"], was_helpful=True)
    Everything else is automatic.
    """

    def __init__(self, mode: str = "lr", domain: str = "general"):
        print("Initialising AdaptRouter...")

        self.domain     = domain
        self.mode       = mode
        self.session_id = str(uuid.uuid4())[:8]

        # ── LOAD BASE ROUTER ──────────────────────────────────────────────────
        if _BASE_AVAILABLE:
            self._base_router = _BaseRouterAgent(mode=mode)
            print(f"  Base router     : llm-router RouterAgent ({mode} mode)")
        else:
            self._base_router = None
            print("  Base router     : NOT AVAILABLE")

        # ── FEEDBACK STORE ────────────────────────────────────────────────────
        try:
            from adaptrouter.feedback_store import FeedbackStore
            self._feedback_store = FeedbackStore()
            print(f"  Feedback store  : active → {self._feedback_store.db_path}")
        except Exception as e:
            self._feedback_store = None
            print(f"  Feedback store  : failed ({e})")

        # ── FEEDBACK COLLECTOR + IMPLICIT DETECTOR ────────────────────────────
        # FIX 5: This was previously disconnected — now fully wired
        try:
            from adaptrouter.feedback_collector import FeedbackCollector
            self._feedback_collector = FeedbackCollector(
                feedback_store=self._feedback_store
            )
            print(f"  Feedback collector : active")
        except Exception as e:
            self._feedback_collector = None
            print(f"  Feedback collector : failed ({e})")

        # ── ADAPTIVE RETRAINER ────────────────────────────────────────────────
        # FIX 5: Retrainer now wired into FeedbackCollector
        try:
            from adaptrouter.retrainer import AdaptiveRetrainer
            self._retrainer = AdaptiveRetrainer(
                feedback_store=self._feedback_store
            )
            if self._feedback_collector is not None:
                self._feedback_collector.set_retrainer(self._retrainer)
            print(f"  Retrainer       : active (threshold={self._retrainer.get_current_accuracy():.1%})")
        except Exception as e:
            self._retrainer = None
            print(f"  Retrainer       : failed ({e})")

        # ── IN-MEMORY DECISION HISTORY (for implicit feedback) ─────────────────
        self._recent_decisions = []

        print(f"  Domain          : {domain}")
        print(f"  Session ID      : {self.session_id}")
        print("AdaptRouter ready.\n")


    def classify(self, query: str) -> dict:
        """
        Classifies query — includes long query guard (Fix 7).
        Returns classification dict without making any API call.
        """
        # FIX 7: Long query guard — very long queries skip classifier
        word_count = len(query.split())
        if word_count > LONG_QUERY_WORD_LIMIT:
            return {
                "label"     : "complex",
                "confidence": 0.99,
                "p_simple"  : 0.01,
                "p_complex" : 0.99,
                "trusted"   : True,
                "reason"    : f"auto-complex: {word_count} words > {LONG_QUERY_WORD_LIMIT} limit",
            }

        if self._base_router is None:
            return {
                "label": "complex", "confidence": 0.5,
                "p_simple": 0.5, "p_complex": 0.5, "trusted": False,
                "reason": "base_router_unavailable"
            }

        return self._base_router.classify(query)


    def route(self, query: str, user_id: str = None) -> dict:
        """
        Full routing pipeline:
        classify → route to model → log → check implicit feedback → return
        """
        if self._base_router is None:
            return self._fallback_result(query)

        query_id   = str(uuid.uuid4())[:12]
        start_time = time.time()

        # Step 1: classify
        classification = self.classify(query)

        # Step 2: route to model
        label      = classification["label"]
        confidence = classification["confidence"]
        trusted    = classification["trusted"]

        if label == "simple" and trusted:
            from src.models import call_fast_model
            model_result   = call_fast_model(query)
            routing_reason = "simple + confident"
        elif label == "complex":
            from src.models import call_smart_model
            model_result   = call_smart_model(query)
            routing_reason = "complex query"
        else:
            from src.models import call_smart_model
            model_result   = call_smart_model(query)
            routing_reason = f"low confidence ({confidence:.2f}) — defaulted to smart"

        latency = round(time.time() - start_time, 3)

        # Step 3: build result
        result = {
            "query"            : query,
            "query_id"         : query_id,
            "answer"           : model_result["answer"],
            "label"            : label,
            "confidence"       : confidence,
            "p_simple"         : classification["p_simple"],
            "p_complex"        : classification["p_complex"],
            "trusted"          : trusted,
            "routing_reason"   : routing_reason,
            "model_used"       : model_result["model_used"],
            "latency_s"        : latency,
            "total_tokens"     : model_result["total_tokens"],
            "prompt_tokens"    : model_result["prompt_tokens"],
            "completion_tokens": model_result["completion_tokens"],
            "domain"           : self.domain,
            "session_id"       : self.session_id,
            "user_id"          : user_id,
            "word_count"       : len(query.split()),
        }

        # Step 4: store decision
        if self._feedback_store is not None:
            try:
                self._feedback_store.store_decision(result)
            except Exception as e:
                print(f"Warning: could not store decision ({e})")

        # Step 5: update recent decisions for implicit detection
        self._recent_decisions.append({
            "query_id"  : query_id,
            "query"     : query,
            "label"     : label,
            "routed_to" : model_result["model_used"],
            "timestamp" : time.time(),
            "confidence": confidence,
        })
        if len(self._recent_decisions) > 20:
            self._recent_decisions.pop(0)

        # Step 6: check implicit feedback (FIX 5 — now actually fires)
        if self._feedback_collector is not None:
            try:
                signals = self._feedback_collector.check_implicit(
                    result, self._recent_decisions
                )
                if signals:
                    result["implicit_signals"] = signals
            except Exception as e:
                pass

        return result


    def feedback(self, query_id: str, was_helpful: bool,
                 rating: int = None) -> dict:
        """
        Records explicit user feedback.
        After 20+ feedback signals the router retrains automatically.

        Usage:
            result = router.route("your query")
            # ... user reads answer ...
            router.feedback(result["query_id"], was_helpful=True)
        """
        if self._feedback_collector is None:
            return {
                "status"  : "no_collector",
                "query_id": query_id,
                "message" : "FeedbackCollector not initialised"
            }

        return self._feedback_collector.record_explicit(
            query_id    = query_id,
            was_helpful = was_helpful,
            rating      = rating,
        )


    def get_stats(self) -> dict:
        """Returns current performance and feedback statistics."""
        stats = {
            "session_id"             : self.session_id,
            "domain"                 : self.domain,
            "mode"                   : self.mode,
            "decisions_this_session" : len(self._recent_decisions),
            "feedback_store_active"  : self._feedback_store is not None,
            "retrainer_active"       : self._retrainer is not None,
            "implicit_detection"     : self._feedback_collector is not None,
        }

        if self._feedback_store is not None:
            try:
                stats.update(self._feedback_store.get_stats())
            except Exception:
                pass

        if self._retrainer is not None:
            stats["current_val_accuracy"] = self._retrainer.get_current_accuracy()

        return stats


    def _fallback_result(self, query: str) -> dict:
        return {
            "query"          : query,
            "query_id"       : str(uuid.uuid4())[:12],
            "answer"         : "Router unavailable. Check LLM_ROUTER_PATH in .env.",
            "label"          : "unknown",
            "confidence"     : 0.0,
            "p_simple"       : 0.5,
            "p_complex"      : 0.5,
            "trusted"        : False,
            "model_used"     : "none",
            "latency_s"      : 0.0,
            "total_tokens"   : 0,
            "prompt_tokens"  : 0,
            "completion_tokens": 0,
            "routing_reason" : "base_router_unavailable",
            "domain"         : self.domain,
            "session_id"     : self.session_id,
            "error"          : "base_router_import_failed",
        }