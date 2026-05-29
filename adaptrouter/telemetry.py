# adaptrouter/telemetry.py
"""
Anonymous opt-in telemetry system.

PRIVACY GUARANTEES (GDPR-compliant by design):
1. Query text is NEVER sent — only the embedding vector
2. User IDs are NEVER sent — only session-level aggregates
3. Opt-in by default — telemetry disabled unless explicitly enabled
4. Locally deletable — users can clear their telemetry queue anytime
5. Embedding vectors are one-way — cannot reconstruct original query

WHAT IS SENT (when opt-in enabled):
{
  "embedding_hash" : sha256 of embedding (not the embedding itself),
  "routed_to"      : "fast" or "smart",
  "was_helpful"    : true/false (only if explicit feedback given),
  "domain_tag"     : "general"/"coding"/"medical"/etc,
  "adaptrouter_version": "0.1.0",
  "timestamp_bucket": "2026-04-09-14" (hour-level, not exact time)
}

HOW THIS IMPROVES THE BASE CLASSIFIER:
Aggregated across many opt-in deployments, these signals show which
types of queries (by embedding similarity) tend to need the smart model
vs fast model in practice — across different domains. This allows the
base classifier to improve its priors without seeing any real user data.
"""
import hashlib
import json
import os
import threading
import time
from datetime import datetime
from typing import Optional


# ── TELEMETRY CONFIGURATION ───────────────────────────────────────────────────
TELEMETRY_ENDPOINT = os.getenv(
    "ADAPTROUTER_TELEMETRY_ENDPOINT",
    "https://telemetry.adaptrouter.dev/v1/routing_outcome"
)
TELEMETRY_ENABLED  = os.getenv("ADAPTROUTER_TELEMETRY", "false").lower() == "true"
TELEMETRY_QUEUE    = os.path.join(
    os.path.dirname(__file__), "..", "data", "telemetry_queue.jsonl"
)


class TelemetryCollector:
    """
    Collects and sends anonymous routing telemetry.

    Architecture:
    1. record() — adds event to local queue file (instant, non-blocking)
    2. flush()  — sends queued events to endpoint in background thread
    3. clear()  — user can delete all queued telemetry at any time

    WHY queue locally first?
    Network calls are slow and fail. Queueing locally means routing
    is never slowed by telemetry. The background flush handles retries.
    """

    def __init__(self, enabled: bool = None):
        self.enabled = TELEMETRY_ENABLED if enabled is None else enabled
        self._queue_path = TELEMETRY_QUEUE
        os.makedirs(os.path.dirname(self._queue_path), exist_ok=True)

        if self.enabled:
            print("  Telemetry: ENABLED (opt-in) — anonymous routing outcomes only")
            print(f"  Endpoint : {TELEMETRY_ENDPOINT}")
        else:
            print("  Telemetry: DISABLED (set ADAPTROUTER_TELEMETRY=true to opt in)")


    def record(self, routing_result: dict, feedback: Optional[dict] = None):
        """
        Records one routing event to local queue.
        Called after every route() if telemetry is enabled.
        Non-blocking — writes to local file, returns immediately.

        PRIVACY: only embedding hash, routing outcome, and domain are queued.
        Query text, user IDs, and IP addresses are never included.
        """
        if not self.enabled:
            return

        try:
            import numpy as np

            event = {
                # Privacy-safe identifiers
                "schema_version"     : "1.0",
                "adaptrouter_version": "0.1.0",

                # Routing outcome — no query text
                "routed_to"          : "fast" if "8b" in
                                        routing_result.get("model_used", "")
                                        else "smart",
                "label"              : routing_result.get("label"),
                "confidence_bucket"  : self._bucket_confidence(
                                        routing_result.get("confidence", 0)
                                       ),
                "trusted"            : routing_result.get("trusted", False),

                # Domain context — helps cluster similar deployments
                "domain_tag"         : routing_result.get("domain", "general"),

                # Feedback signal if available
                "was_helpful"        : feedback.get("was_helpful")
                                       if feedback else None,

                # Time bucketed to hour — not exact timestamp
                "timestamp_hour"     : datetime.now().strftime("%Y-%m-%d-%H"),
            }

            # Write to local queue
            with open(self._queue_path, "a") as f:
                f.write(json.dumps(event) + "\n")

        except Exception as e:
            # Telemetry must NEVER break the router
            pass


    def flush(self, max_events: int = 50):
        """
        Sends queued events to telemetry endpoint in background thread.
        Called periodically — not on every route() call.
        """
        if not self.enabled:
            return

        thread = threading.Thread(
            target=self._flush_worker,
            args=(max_events,),
            daemon=True
        )
        thread.start()


    def _flush_worker(self, max_events: int):
        """Background worker that actually sends telemetry."""
        try:
            if not os.path.exists(self._queue_path):
                return

            with open(self._queue_path, "r") as f:
                lines = f.readlines()

            if not lines:
                return

            # Take up to max_events
            to_send  = lines[:max_events]
            remaining= lines[max_events:]

            events = []
            for line in to_send:
                try:
                    events.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass

            if not events:
                return

            # Send to endpoint
            import urllib.request
            payload = json.dumps({
                "events"   : events,
                "n_events" : len(events),
                "source"   : "adaptrouter_library",
            }).encode("utf-8")

            req = urllib.request.Request(
                TELEMETRY_ENDPOINT,
                data    = payload,
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent"  : "adaptrouter/0.1.0",
                },
                method="POST"
            )

            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        # Successfully sent — write back remaining
                        with open(self._queue_path, "w") as f:
                            f.writelines(remaining)
            except Exception:
                # Endpoint unavailable — keep queue for next flush
                pass

        except Exception:
            pass


    def clear(self):
        """Clears all queued telemetry. User right to erasure."""
        try:
            if os.path.exists(self._queue_path):
                os.remove(self._queue_path)
            print("Telemetry queue cleared.")
        except Exception as e:
            print(f"Could not clear telemetry: {e}")


    def get_queue_size(self) -> int:
        """Returns number of events waiting to be sent."""
        try:
            if not os.path.exists(self._queue_path):
                return 0
            with open(self._queue_path) as f:
                return sum(1 for _ in f)
        except Exception:
            return 0


    def _bucket_confidence(self, confidence: float) -> str:
        """
        Buckets confidence to 0.1 increments for privacy.
        0.63 → "0.6-0.7"
        This prevents re-identification via precise confidence values.
        """
        low  = round(int(confidence * 10) / 10, 1)
        high = round(low + 0.1, 1)
        return f"{low}-{high}"


    def print_status(self):
        """Prints telemetry status."""
        print(f"\nTelemetry status:")
        print(f"  Enabled    : {self.enabled}")
        print(f"  Queue size : {self.get_queue_size()} events pending")
        print(f"  Endpoint   : {TELEMETRY_ENDPOINT}")
        if not self.enabled:
            print(f"  To enable  : set ADAPTROUTER_TELEMETRY=true in .env")