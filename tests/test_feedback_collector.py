# tests/test_feedback_collector.py
import pytest
import os
import tempfile
from adaptrouter.feedback_store import FeedbackStore
from adaptrouter.feedback_collector import FeedbackCollector


@pytest.fixture
def store():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    s = FeedbackStore(db_path=db_path)
    yield s
    os.unlink(db_path)


@pytest.fixture
def collector(store):
    return FeedbackCollector(feedback_store=store)


SAMPLE_RESULT = {
    "query_id": "qtest01", "query": "What is AI?",
    "label": "simple", "model_used": "llama-3.1-8b-instant",
    "confidence": 0.8, "latency_s": 0.5,
    "domain": "general", "session_id": "sess01"
}


class TestRecordExplicit:

    def test_positive_feedback_recorded(self, collector, store):
        store.store_decision(SAMPLE_RESULT)
        result = collector.record_explicit("qtest01", was_helpful=True)
        assert result["status"] == "recorded"
        assert result["was_helpful"] is True

    def test_negative_feedback_recorded(self, collector, store):
        store.store_decision(SAMPLE_RESULT)
        result = collector.record_explicit("qtest01", was_helpful=False)
        assert result["status"] == "recorded"
        assert result["new_labelled_count"] == 1

    def test_retrain_not_triggered_below_threshold(self, collector, store):
        store.store_decision(SAMPLE_RESULT)
        result = collector.record_explicit("qtest01", was_helpful=True)
        assert result["retrain_triggered"] is False

    def test_examples_until_retrain_counted(self, collector, store):
        store.store_decision(SAMPLE_RESULT)
        result = collector.record_explicit("qtest01", was_helpful=True)
        assert result["examples_until_retrain"] == 19   # 20 - 1


class TestShouldRetrain:

    def test_should_not_retrain_without_retrainer(self, collector, store):
        """No retrainer set → should_retrain always False."""
        assert collector.should_retrain() is False

    def test_should_not_retrain_below_threshold(self, collector, store):
        store.store_decision(SAMPLE_RESULT)
        store.store_feedback("qtest01", was_helpful=True)
        assert collector.should_retrain() is False  # only 1, need 20


class TestRecordImplicit:

    def test_implicit_recorded(self, collector, store):
        store.store_decision(SAMPLE_RESULT)
        result = collector.record_implicit("qtest01", "rephrasing", 0.7)
        assert result["status"] == "recorded"
        assert result["implicit_type"] == "rephrasing"
        assert result["confidence"] == 0.7


class TestFeedbackSummary:

    def test_summary_has_threshold(self, collector):
        summary = collector.get_feedback_summary()
        assert "retrain_threshold" in summary
        assert summary["retrain_threshold"] == 20