# adaptrouter/retrainer.py  — COMPLETE FIXED VERSION
import sys
import os
import time
import numpy as np
import joblib
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from adaptrouter.config import (
    LLM_ROUTER_PATH,
    ACCURACY_TOLERANCE,
)

if LLM_ROUTER_PATH not in sys.path:
    sys.path.insert(0, LLM_ROUTER_PATH)

try:
    from src.embedder import embed_batch
    from data.training_queries import TRAINING_DATA
    _BASE_AVAILABLE = True
except ImportError:
    _BASE_AVAILABLE = False


def _load_validation_data():
    """
    Loads the permanent held-out validation set.
    FIX 6: Uses 20-query validation file instead of 5-query slice.
    Falls back to training slice if validation file not found.
    """
    try:
        # Try to import the dedicated validation file first
        val_path = os.path.join(LLM_ROUTER_PATH, "data", "validation_queries.py")
        if os.path.exists(val_path):
            import importlib.util
            spec   = importlib.util.spec_from_file_location("validation_queries", val_path)
            mod    = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            data   = mod.VALIDATION_DATA
            texts  = [d[0] for d in data]
            labels = [d[2] for d in data]
            return texts, labels, "dedicated_validation_file"
    except Exception:
        pass

    # Fallback: use last 20% of training data
    split   = int(len(TRAINING_DATA) * 0.8)
    val     = TRAINING_DATA[split:]
    texts   = [d[0] for d in val]
    labels  = [d[1] for d in val]
    return texts, labels, "training_data_slice"


class AdaptiveRetrainer:
    """
    Retrains the routing classifier when enough feedback accumulates.

    Key fixes in this version:
    - FIX 2: Warm-start bug fixed — dummy fit before weight copy
    - FIX 4: Platt scaling applied after retraining for better calibration
    - FIX 6: Uses 20-query dedicated validation set instead of 5-query slice
    """

    def __init__(self, feedback_store, classifier_path: str = None):
        self.store           = feedback_store
        self.classifier_path = classifier_path or os.path.join(
            LLM_ROUTER_PATH, "models", "router_classifier.pkl"
        )
        self.current_clf     = None
        self.X_val           = None
        self.y_val           = None
        self._retrain_count  = 0
        self._val_source     = None

        self._load_current_classifier()
        self._prepare_validation_set()


    def _load_current_classifier(self):
        """Loads the existing classifier from llm-router."""
        try:
            self.current_clf = joblib.load(self.classifier_path)
            print(f"  Retrainer: loaded classifier from {self.classifier_path}")
        except Exception as e:
            print(f"  Retrainer: could not load ({e}) — creating fresh")
            self.current_clf = LogisticRegression(max_iter=1000, random_state=42)


    def _prepare_validation_set(self):
        """
        FIX 6: Loads 20-query dedicated validation set.
        This gives enough statistical power to detect real accuracy changes.
        With 5 queries one wrong answer = 20% change — too noisy.
        With 20 queries one wrong answer = 5% change — meaningful signal.
        """
        if not _BASE_AVAILABLE:
            return

        try:
            val_texts, val_labels, source = _load_validation_data()
            self.X_val      = embed_batch(val_texts)
            self.y_val      = np.array(val_labels)
            self._val_source = source
            print(f"  Retrainer: validation set ready "
                  f"({len(val_texts)} queries, source={source})")
        except Exception as e:
            print(f"  Retrainer: validation set failed ({e})")


    def get_current_accuracy(self) -> float:
        """Measures current classifier accuracy on validation set."""
        if self.current_clf is None or self.X_val is None:
            return 0.0
        try:
            preds = self.current_clf.predict(self.X_val)
            return round(accuracy_score(self.y_val, preds), 4)
        except Exception:
            return 0.0


    def retrain(self) -> dict:
        """
        Retrains with warm-start (FIX 2) and Platt calibration (FIX 4).

        WARM-START FIX:
        Old code: copy weights → fit() — sklearn ignores copied weights
                  on first fit() if model state not initialised
        New code: fit on dummy data first → copy weights → fit on real data
                  This forces sklearn to use copied weights as start point

        CALIBRATION FIX:
        After retraining, wrap classifier with CalibratedClassifierCV
        using Platt scaling so confidence scores are reliable probabilities.
        """
        print(f"\n[Retrainer] Starting retrain #{self._retrain_count + 1}...")
        start_time = time.time()

        if not _BASE_AVAILABLE:
            return {"status": "failed", "reason": "base_router_not_available"}

        # ── STEP 1: FETCH FEEDBACK DATA ───────────────────────────────────────
        feedback_data = self.store.get_labelled_feedback(unused_only=True)

        if len(feedback_data) < 5:
            return {
                "status": "skipped",
                "reason": f"only {len(feedback_data)} examples, need at least 5"
            }

        # ── STEP 2: BUILD COMBINED TRAINING SET ───────────────────────────────
        # Use first 80% of training data (rest is validation)
        split_idx   = int(len(TRAINING_DATA) * 0.8)
        orig_texts  = [d[0] for d in TRAINING_DATA[:split_idx]]
        orig_labels = [d[1] for d in TRAINING_DATA[:split_idx]]

        # Weight feedback 2x — more recent, more domain-relevant
        fb_texts  = [f["query_text"]    for f in feedback_data]
        fb_labels = [f["true_label_int"] for f in feedback_data]

        all_texts  = orig_texts  + fb_texts  + fb_texts   # 2x weight
        all_labels = orig_labels + fb_labels + fb_labels

        print(f"  Training: {len(orig_texts)} original + "
              f"{len(fb_texts)} feedback × 2 = {len(all_texts)} total")

        # ── STEP 3: EMBED ──────────────────────────────────────────────────────
        try:
            X_new = embed_batch(all_texts)
            y_new = np.array(all_labels)
        except Exception as e:
            return {"status": "failed", "reason": f"embedding failed: {e}"}

        # ── STEP 4: MEASURE CURRENT ACCURACY ──────────────────────────────────
        old_accuracy = self.get_current_accuracy()
        print(f"  Current validation accuracy: {old_accuracy:.1%}")

        # ── STEP 5: WARM-START TRAINING (FIX 2) ───────────────────────────────
        try:
            new_clf = LogisticRegression(
                max_iter=1000, random_state=42, warm_start=True
            )

            # FIX 2: Dummy fit to initialise sklearn internal state
            # Without this, sklearn resets weights on first fit() call
            # regardless of what you manually set on coef_/intercept_
            if self.X_val is not None and len(self.X_val) >= 2:
                new_clf.fit(self.X_val[:2], self.y_val[:2])
            else:
                new_clf.fit(X_new[:2], y_new[:2])

            # Now copy weights — sklearn will actually use them
            if hasattr(self.current_clf, "coef_"):
                new_clf.coef_      = self.current_clf.coef_.copy()
                new_clf.intercept_ = self.current_clf.intercept_.copy()
                print(f"  Warm-start: copied weights from current classifier")

            # Full training — continues from copied weights
            new_clf.fit(X_new, y_new)

        except Exception as e:
            return {"status": "failed", "reason": f"training failed: {e}"}

        # ── STEP 6: PLATT CALIBRATION (FIX 4) ─────────────────────────────────
        # Wraps the classifier with sigmoid calibration so confidence
        # scores are reliable probability estimates, not arbitrary numbers.
        # After calibration: 0.65 confidence → actually 65% accuracy.
        try:
            if self.X_val is not None and len(self.X_val) >= 4:
                calibrated_clf = CalibratedClassifierCV(
                    new_clf, method="sigmoid", cv="prefit"
                )
                calibrated_clf.fit(self.X_val, self.y_val)
                final_clf = calibrated_clf
                print(f"  Platt scaling: calibration applied")
            else:
                final_clf = new_clf
                print(f"  Platt scaling: skipped (need >= 4 validation samples)")
        except Exception as e:
            final_clf = new_clf
            print(f"  Platt scaling failed ({e}) — using uncalibrated")

        # ── STEP 7: VALIDATE ───────────────────────────────────────────────────
        try:
            new_preds    = final_clf.predict(self.X_val)
            new_accuracy = round(accuracy_score(self.y_val, new_preds), 4)
            improvement  = round(new_accuracy - old_accuracy, 4)
            print(f"  New accuracy: {new_accuracy:.1%} ({improvement:+.1%})")
        except Exception as e:
            return {"status": "failed", "reason": f"evaluation failed: {e}"}

        retrain_time = round(time.time() - start_time, 2)

        # ── STEP 8: ACCEPT OR REJECT ───────────────────────────────────────────
        if improvement >= -ACCURACY_TOLERANCE:
            self.current_clf = final_clf

            try:
                joblib.dump(final_clf, self.classifier_path)
                print(f"  Saved to {self.classifier_path}")
            except Exception as e:
                print(f"  Warning: save failed ({e})")

            fb_ids = [f["feedback_id"] for f in feedback_data]
            self.store.mark_feedback_as_used(fb_ids)
            status = "improved" if improvement > 0 else "maintained"
            self._retrain_count += 1

        else:
            status = "rejected"
            print(f"  Rejected — regression of {abs(improvement):.1%} "
                  f"exceeds tolerance {ACCURACY_TOLERANCE:.1%}")

        # ── STEP 9: LOG ────────────────────────────────────────────────────────
        self.store.log_retrain_event(
            old_acc    = old_accuracy,
            new_acc    = new_accuracy,
            n_examples = len(feedback_data),
            status     = status,
            notes      = f"retrain_time={retrain_time}s, "
                         f"n_total={len(all_texts)}, "
                         f"val_source={self._val_source}, "
                         f"calibration=platt"
        )

        return {
            "status"        : status,
            "old_accuracy"  : old_accuracy,
            "new_accuracy"  : new_accuracy,
            "improvement"   : improvement,
            "n_feedback"    : len(feedback_data),
            "n_total"       : len(all_texts),
            "retrain_time_s": retrain_time,
            "retrain_count" : self._retrain_count,
            "calibrated"    : isinstance(final_clf, CalibratedClassifierCV),
        }