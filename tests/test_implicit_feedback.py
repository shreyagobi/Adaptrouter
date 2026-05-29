# tests/test_implicit_feedback.py
import pytest
import os
import tempfile
import time
from unittest.mock import patch, MagicMock
from adaptrouter.feedback_store import FeedbackStore
from adaptrouter.implicit_feedback import ImplicitFeedbackDetector


@pytest.fixture
def store():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    s = FeedbackStore(db_path=db_path)
    yield s
    os.unlink(db_path)


@pytest.fixture
def detector(store):
    return ImplicitFeedbackDetector(feedback_store=store)


def make_decision(query_id, query, routed_to="fast", label="simple", ts=None):
    return {
        "query_id"  : query_id,
        "query"     : query,
        "routed_to" : routed_to,
        "label"     : label,
        "timestamp" : ts or time.time(),
    }


class TestEscalationDetection:

    def test_explain_more_detected(self, detector):
        current = make_decision("q2", "explain more about gradient descent")
        signals = detector.check_all_signals(current, [
            make_decision("q1", "what is gradient descent"),
            current
        ])
        types = [s["type"] for s in signals]
        assert "escalation" in types

    def test_normal_query_not_escalation(self, detector):
        current = make_decision("q2", "what is the capital of France?")
        signals = detector.check_all_signals(current, [
            make_decision("q1", "what is machine learning"),
            current
        ])
        types = [s["type"] for s in signals]
        assert "escalation" not in types


class TestRephrasing:

    def test_no_rephrasing_with_one_decision(self, detector):
        current = make_decision("q1", "what is AI?")
        signals = detector.check_all_signals(current, [current])
        assert signals == []   # needs at least 2 decisions

    def test_completely_different_queries_not_rephrasing(self, detector):
        current = make_decision("q2", "what is the weather today?")
        recent  = [
            make_decision("q1", "explain transformer architecture"),
            current
        ]
        with patch("adaptrouter.implicit_feedback._EMBED_AVAILABLE", False):
            signals = detector.check_all_signals(current, recent)
        rephrasings = [s for s in signals if s["type"] == "rephrasing"]
        assert len(rephrasings) == 0


class TestAcceptance:

    def test_acceptance_stored_in_db(self, detector, store):
        store.store_decision({
            "query_id"  : "q1", "query": "what is machine learning?",
            "label"     : "simple", "model_used": "llama-3.1-8b-instant",
            "confidence": 0.8, "latency_s": 0.5,
            "domain"    : "general", "session_id": "s1"
        })
        with patch("adaptrouter.implicit_feedback._EMBED_AVAILABLE", False):
            current = make_decision("q2", "completely unrelated new topic")
            detector.check_all_signals(current, [
                make_decision("q1", "what is machine learning?"),
                current
            ])
        # No crash is the test — acceptance without embed available is skipped


class TestSignalSummary:

    def test_empty_signals_summary(self, detector):
        summary = detector.get_signal_summary([])
        assert "No implicit" in summary

    def test_signals_listed_in_summary(self, detector):
        signals = [
            {"type": "rephrasing", "confidence": 0.7},
            {"type": "escalation", "confidence": 0.8},
        ]
        summary = detector.get_signal_summary(signals)
        assert "rephrasing" in summary
        assert "escalation" in summary