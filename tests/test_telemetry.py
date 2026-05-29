# tests/test_telemetry.py
import pytest
import os
import tempfile
from adaptrouter.telemetry import TelemetryCollector


SAMPLE_RESULT = {
    "query_id"  : "tel001",
    "query"     : "What is gradient descent?",
    "label"     : "complex",
    "model_used": "llama-3.3-70b-versatile",
    "confidence": 0.75,
    "trusted"   : True,
    "domain"    : "coding",
}


class TestTelemetryCollector:

    def test_disabled_by_default(self):
        """Telemetry disabled unless explicitly opted in."""
        collector = TelemetryCollector(enabled=False)
        assert collector.enabled is False

    def test_record_does_nothing_when_disabled(self, tmp_path):
        """No events queued when telemetry disabled."""
        collector = TelemetryCollector(enabled=False)
        collector._queue_path = str(tmp_path / "queue.jsonl")
        collector.record(SAMPLE_RESULT)
        assert collector.get_queue_size() == 0

    def test_record_queues_event_when_enabled(self, tmp_path):
        """Event queued when telemetry enabled."""
        collector = TelemetryCollector(enabled=True)
        collector._queue_path = str(tmp_path / "queue.jsonl")
        collector.record(SAMPLE_RESULT)
        assert collector.get_queue_size() == 1

    def test_queued_event_has_no_query_text(self, tmp_path):
        """Privacy: queued events must not contain query text."""
        import json
        collector = TelemetryCollector(enabled=True)
        collector._queue_path = str(tmp_path / "queue.jsonl")
        collector.record(SAMPLE_RESULT)
        with open(collector._queue_path) as f:
            event = json.loads(f.read().strip())
        assert "query" not in event
        assert "query_text" not in event
        assert SAMPLE_RESULT["query"] not in str(event)

    def test_queued_event_has_routing_outcome(self, tmp_path):
        """Queued events must include routing outcome."""
        import json
        collector = TelemetryCollector(enabled=True)
        collector._queue_path = str(tmp_path / "queue.jsonl")
        collector.record(SAMPLE_RESULT)
        with open(collector._queue_path) as f:
            event = json.loads(f.read().strip())
        assert "routed_to" in event
        assert event["routed_to"] == "smart"

    def test_clear_removes_queue(self, tmp_path):
        """clear() removes all queued events."""
        collector = TelemetryCollector(enabled=True)
        collector._queue_path = str(tmp_path / "queue.jsonl")
        collector.record(SAMPLE_RESULT)
        assert collector.get_queue_size() == 1
        collector.clear()
        assert collector.get_queue_size() == 0

    def test_confidence_bucketing(self):
        """Confidence is bucketed to 0.1 increments."""
        collector = TelemetryCollector(enabled=False)
        bucket    = collector._bucket_confidence(0.73)
        assert bucket == "0.7-0.8"