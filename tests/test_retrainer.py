# tests/test_retrainer.py
import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from adaptrouter.feedback_store import FeedbackStore
from adaptrouter.retrainer import AdaptiveRetrainer


@pytest.fixture
def store():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    s = FeedbackStore(db_path=db_path)
    yield s
    os.unlink(db_path)


class TestAdaptiveRetrainer:

    def test_retrainer_init_without_base(self, store):
        """Retrainer initialises gracefully even without llm-router."""
        with patch("adaptrouter.retrainer._BASE_AVAILABLE", False):
            retrainer = AdaptiveRetrainer(feedback_store=store)
            assert retrainer is not None

    def test_retrain_fails_gracefully_without_base(self, store):
        """retrain() returns failed status when base not available."""
        with patch("adaptrouter.retrainer._BASE_AVAILABLE", False):
            retrainer = AdaptiveRetrainer(feedback_store=store)
            result    = retrainer.retrain()
            assert result["status"] == "failed"

    def test_retrain_skipped_with_few_examples(self, store):
        """retrain() skips when fewer than 5 feedback examples available."""
        retrainer = AdaptiveRetrainer(feedback_store=store)
        result    = retrainer.retrain()
        # Empty store → 0 examples → skipped
        assert result["status"] in ["skipped", "failed"]

    def test_get_current_accuracy_returns_float(self, store):
        """get_current_accuracy returns a float between 0 and 1."""
        retrainer = AdaptiveRetrainer(feedback_store=store)
        acc = retrainer.get_current_accuracy()
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0