# adaptrouter/feedback_collector.py
import threading
from adaptrouter.feedback_store import FeedbackStore
from adaptrouter.implicit_feedback import ImplicitFeedbackDetector
from adaptrouter.config import RETRAIN_THRESHOLD, MIN_RETRAIN_INTERVAL_HOURS


class FeedbackCollector:
    """
    Unified API for all feedback operations.

    WHY a separate collector class?
    FeedbackStore handles persistence (SQLite operations).
    ImplicitFeedbackDetector handles signal detection.
    FeedbackCollector orchestrates them and adds business logic:
      - When to trigger retraining
      - How to combine explicit + implicit signals
      - Thread-safe retrain triggering

    The developer only needs to interact with FeedbackCollector.
    They never call FeedbackStore or ImplicitFeedbackDetector directly.
    This is the Facade pattern — one simple interface over multiple
    complex subsystems.
    """

    def __init__(self, feedback_store: FeedbackStore):
        self.store    = feedback_store
        self.detector = ImplicitFeedbackDetector(feedback_store=feedback_store)
        self._retrain_lock = threading.Lock()
        self._retrainer    = None   # set by AdaptRouter on Day 5


    def set_retrainer(self, retrainer):
        """Links the retraining engine — called by AdaptRouter on Day 5."""
        self._retrainer = retrainer


    def record_explicit(self, query_id: str, was_helpful: bool,
                        rating: int = None) -> dict:
        """
        Records explicit user feedback.
        Called when user clicks thumbs up/down or rates the answer.

        Returns a dict indicating:
        - feedback was stored
        - whether retraining was triggered
        - how many more examples needed for next retrain
        """
        self.store.store_feedback(
            query_id    = query_id,
            was_helpful = was_helpful,
            user_rating = rating,
        )

        new_count         = self.store.count_new_labelled()
        retrain_triggered = False
        retrain_result    = None

        if self.should_retrain():
            retrain_triggered = True
            retrain_result    = self._trigger_retrain_async()

        remaining = max(0, RETRAIN_THRESHOLD - new_count)

        return {
            "status"            : "recorded",
            "query_id"          : query_id,
            "was_helpful"       : was_helpful,
            "new_labelled_count": new_count,
            "retrain_triggered" : retrain_triggered,
            "retrain_result"    : retrain_result,
            "examples_until_retrain": remaining,
            "message"           : (
                "Retraining triggered!" if retrain_triggered
                else f"{remaining} more feedback examples until next retrain"
            )
        }


    def record_implicit(self, query_id: str, implicit_type: str,
                        confidence: float) -> dict:
        """
        Records implicit feedback detected by ImplicitFeedbackDetector.
        Lower-weighted than explicit — reflected in confidence score.
        """
        self.store.store_feedback(
            query_id      = query_id,
            implicit_type = implicit_type,
            implicit_conf = confidence,
            # was_helpful is None for implicit — we infer it during retraining
        )

        return {
            "status"       : "recorded",
            "query_id"     : query_id,
            "implicit_type": implicit_type,
            "confidence"   : confidence,
        }


    def check_implicit(self, current_result: dict,
                       recent_decisions: list) -> list:
        """
        Runs implicit feedback detection on the latest routing result.
        Called automatically by AdaptRouter.route() after every decision.
        Returns list of detected signals.
        """
        signals = self.detector.check_all_signals(current_result, recent_decisions)
        return signals


    def should_retrain(self) -> bool:
        """
        Decides whether retraining should be triggered now.

        Three conditions must ALL be true:
        1. Enough new labelled examples (RETRAIN_THRESHOLD)
        2. Enough time since last retrain (MIN_RETRAIN_INTERVAL_HOURS)
        3. A retrainer is available (set on Day 5)

        WHY minimum time interval?
        Retraining on tiny batches (5 new examples) is worse than
        waiting and retraining on larger batches (20+ examples).
        Statistical noise in small batches can push accuracy DOWN.
        The minimum interval ensures meaningful batch sizes.
        """
        new_count = self.store.count_new_labelled()

        # Condition 1: enough data
        if new_count < RETRAIN_THRESHOLD:
            return False

        # Condition 2: enough time
        hours_since = self.store.hours_since_last_retrain()
        if hours_since < MIN_RETRAIN_INTERVAL_HOURS:
            return False

        # Condition 3: retrainer available
        if self._retrainer is None:
            return False

        return True


    def _trigger_retrain_async(self) -> dict:
        """
        Triggers retraining in a background thread so it does not block
        the user-facing route() call.

        WHY background thread?
        Retraining takes 1-5 seconds. If it ran in the foreground,
        every user request during retraining would hang.
        The background thread lets routing continue while retraining runs.
        The lock prevents two retraining jobs from running simultaneously.
        """
        if not self._retrain_lock.acquire(blocking=False):
            # Another retrain is already running
            return {"status": "already_running"}

        def _retrain_job():
            try:
                result = self._retrainer.retrain()
                print(f"\n[AdaptRouter] Retraining complete: {result}")
            except Exception as e:
                print(f"\n[AdaptRouter] Retraining failed: {e}")
            finally:
                self._retrain_lock.release()

        thread = threading.Thread(target=_retrain_job, daemon=True)
        thread.start()

        return {"status": "triggered", "thread_id": thread.ident}


    def get_feedback_summary(self) -> dict:
        """Returns current feedback statistics."""
        store_stats = self.store.get_stats()
        store_stats["retrain_threshold"]  = RETRAIN_THRESHOLD
        store_stats["should_retrain_now"] = self.should_retrain()
        return store_stats