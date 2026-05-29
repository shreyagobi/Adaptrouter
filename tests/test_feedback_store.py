# tests/test_feedback_store.py
import pytest
import os
import tempfile
from adaptrouter.feedback_store import FeedbackStore


@pytest.fixture
def store():
    """Creates a temporary store for each test — clean slate every time."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    s = FeedbackStore(db_path=db_path)
    yield s
    s.close()
    os.unlink(db_path)


SAMPLE_RESULT = {
    "query_id"   : "test123",
    "query"      : "What is the capital of France?",
    "label"      : "simple",
    "model_used" : "llama-3.1-8b-instant",
    "confidence" : 0.75,
    "latency_s"  : 0.46,
    "domain"     : "general",
    "session_id" : "abc12345",
}


class TestStoreDecision:

    def test_store_decision_succeeds(self, store):
        store.store_decision(SAMPLE_RESULT)
        stats = store.get_stats()
        assert stats["total_decisions"] == 1

    def test_duplicate_query_id_ignored(self, store):
        store.store_decision(SAMPLE_RESULT)
        store.store_decision(SAMPLE_RESULT)   # same query_id
        stats = store.get_stats()
        assert stats["total_decisions"] == 1   # not duplicated


class TestStoreFeedback:

    def test_explicit_positive_feedback(self, store):
        store.store_decision(SAMPLE_RESULT)
        store.store_feedback("test123", was_helpful=True)
        stats = store.get_stats()
        assert stats["total_feedback"] == 1
        assert stats["positive_rate"] == 1.0

    def test_explicit_negative_feedback(self, store):
        store.store_decision(SAMPLE_RESULT)
        store.store_feedback("test123", was_helpful=False)
        stats = store.get_stats()
        assert stats["positive_rate"] == 0.0

    def test_implicit_feedback_stored(self, store):
        store.store_decision(SAMPLE_RESULT)
        store.store_feedback("test123", implicit_type="rephrasing", implicit_conf=0.7)
        stats = store.get_stats()
        assert stats["implicit_feedback"] == 1


class TestLabelledFeedback:

    def test_correct_label_derived_helpful(self, store):
        """When was_helpful=True, true_label = predicted_label."""
        store.store_decision(SAMPLE_RESULT)
        store.store_feedback("test123", was_helpful=True)
        labelled = store.get_labelled_feedback()
        assert len(labelled) == 1
        assert labelled[0]["true_label"] == "simple"

    def test_wrong_label_derived_unhelpful(self, store):
        """When was_helpful=False and routed to fast, true_label = complex."""
        store.store_decision(SAMPLE_RESULT)
        store.store_feedback("test123", was_helpful=False)
        labelled = store.get_labelled_feedback()
        assert labelled[0]["true_label"] == "complex"

    def test_count_new_labelled(self, store):
        store.store_decision(SAMPLE_RESULT)
        store.store_feedback("test123", was_helpful=True)
        assert store.count_new_labelled() == 1

    def test_mark_as_used_removes_from_count(self, store):
        store.store_decision(SAMPLE_RESULT)
        store.store_feedback("test123", was_helpful=True)
        labelled = store.get_labelled_feedback()
        ids = [r["feedback_id"] for r in labelled]
        store.mark_feedback_as_used(ids)
        assert store.count_new_labelled() == 0


class TestRetrainingHistory:

    def test_hours_since_retrain_default(self, store):
        """Returns 999 when never retrained."""
        assert store.hours_since_last_retrain() == 999.0

    def test_log_retrain_event(self, store):
        store.log_retrain_event(0.80, 0.90, 20, "improved")
        stats = store.get_stats()
        assert stats["retraining_events"] == 1