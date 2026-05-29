# adaptrouter/drift_detector.py
import sys
import numpy as np
from datetime import datetime
from adaptrouter.config import LLM_ROUTER_PATH, DRIFT_THRESHOLD

if LLM_ROUTER_PATH not in sys.path:
    sys.path.insert(0, LLM_ROUTER_PATH)

try:
    from src.embedder import embed_batch
    from data.training_queries import TRAINING_DATA
    _BASE_AVAILABLE = True
except ImportError:
    _BASE_AVAILABLE = False


class DriftDetector:
    """
    Detects when live query distribution shifts away from training data.
    """

    def __init__(self, feedback_store=None, threshold: float = None):
        self.store          = feedback_store
        self.threshold      = threshold or DRIFT_THRESHOLD
        self.training_mean  = None
        self.drift_history  = []

        self._compute_training_centroid()
        self.auto_calibrate_threshold()

    # ✅ FIXED: Proper indentation + correct placement
    def auto_calibrate_threshold(self, n_bootstrap: int = 100):
        """
        Calibrates drift threshold using bootstrap sampling.
        Prevents false alerts on normal in-distribution query variation.
        """
        if not _BASE_AVAILABLE or self.training_mean is None:
            print("  DriftDetector: cannot auto-calibrate — training data unavailable")
            return

        try:
            texts = [d[0] for d in TRAINING_DATA]
            X     = embed_batch(texts)
            scores = []

            for _ in range(n_bootstrap):
                idx    = np.random.choice(len(texts), size=len(texts)//2, replace=False)
                subset = X[idx]

                score  = float(np.linalg.norm(
                    self.training_mean - subset.mean(axis=0)
                ))
                scores.append(score)

            old_threshold  = self.threshold
            self.threshold = float(np.percentile(scores, 95))

            print(f"  DriftDetector: threshold calibrated "
                  f"{old_threshold:.4f} → {self.threshold:.4f} "
                  f"(95th percentile of {n_bootstrap} bootstrap samples)")

        except Exception as e:
            print(f"  DriftDetector: calibration failed ({e})")

    def _compute_training_centroid(self):
        """
        Computes centroid of training data.
        """
        if not _BASE_AVAILABLE:
            return

        try:
            texts              = [d[0] for d in TRAINING_DATA]
            X_train            = embed_batch(texts)
            self.training_mean = X_train.mean(axis=0)

            print(f"  DriftDetector: training centroid computed "
                  f"({len(texts)} queries, {X_train.shape[1]} dims)")

        except Exception as e:
            print(f"  DriftDetector: could not compute training centroid ({e})")

    def compute_drift_score(self, live_queries: list) -> dict:
        """
        Computes drift score between training and live query distributions.
        """
        if self.training_mean is None:
            return {"status": "unavailable", "reason": "training centroid not computed"}

        if not live_queries:
            return {"status": "unavailable", "reason": "no live queries provided"}

        if not _BASE_AVAILABLE:
            return {"status": "unavailable", "reason": "embedding model not available"}

        try:
            X_live    = embed_batch(live_queries)
            live_mean = X_live.mean(axis=0)

            diff        = self.training_mean - live_mean
            drift_score = float(np.linalg.norm(diff))
            drift_score = round(drift_score, 6)

            alert    = drift_score > self.threshold
            severity = self._get_severity(drift_score)

            self.drift_history.append({
                "timestamp": datetime.now().isoformat(),
                "score": drift_score,
                "n_queries": len(live_queries),
                "alert": alert,
            })

            result = {
                "status": "computed",
                "drift_score": drift_score,
                "threshold": self.threshold,
                "alert": alert,
                "severity": severity,
                "n_live_queries": len(live_queries),
                "recommendation": self._get_recommendation(drift_score, severity),
            }

            if alert:
                print(f"\n[DriftDetector] ALERT: drift_score={drift_score:.4f} "
                      f"> threshold={self.threshold} ({severity})")
                print(f"  Recommendation: {result['recommendation']}")

            return result

        except Exception as e:
            return {"status": "failed", "reason": str(e)}

    def _get_severity(self, score: float) -> str:
        if score < 0.05:
            return "none"
        elif score < self.threshold:
            return "mild"
        elif score < self.threshold * 2:
            return "moderate"
        else:
            return "severe"

    def _get_recommendation(self, score: float, severity: str) -> str:
        recommendations = {
            "none": "No action needed. Query distribution matches training data.",
            "mild": "Monitor closely. Consider collecting domain-specific examples.",
            "moderate": "Retrain recommended. Add 20+ examples from your current query distribution.",
            "severe": "Urgent retrain needed. Current queries are very different from training data. "
                      "Router confidence scores are unreliable.",
        }
        return recommendations.get(severity, "Unknown severity.")

    def check_from_store(self, n_recent: int = 50) -> dict:
        if self.store is None:
            return {"status": "unavailable", "reason": "no feedback store"}

        try:
            import sqlite3
            conn = sqlite3.connect(self.store.db_path)

            rows = conn.execute(
                "SELECT query_text FROM routing_decisions "
                "ORDER BY timestamp DESC LIMIT ?", (n_recent,)
            ).fetchall()

            conn.close()

            live_queries = [r[0] for r in rows]

            if not live_queries:
                return {"status": "unavailable", "reason": "no queries in store yet"}

            return self.compute_drift_score(live_queries)

        except Exception as e:
            return {"status": "failed", "reason": str(e)}

    def get_drift_trend(self) -> list:
        return self.drift_history[-20:]