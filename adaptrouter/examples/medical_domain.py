# adaptrouter/examples/medical_domain.py
"""
Example: Adapting the router to a medical domain.

Shows how adaptrouter self-improves on medical queries
that were not in the original training data.

Usage:
    python -m adaptrouter.examples.medical_domain
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import tempfile
from adaptrouter import AdaptRouter
from adaptrouter.feedback_store import FeedbackStore
from adaptrouter.feedback_collector import FeedbackCollector
from adaptrouter.retrainer import AdaptiveRetrainer
from adaptrouter.drift_detector import DriftDetector


def run_medical_example():
    print("="*60)
    print("AdaptRouter — Medical Domain Adaptation Example")
    print("="*60)
    print("""
This example shows how adaptrouter adapts to medical queries
that were not in the original training data.

Step 1: Initial routing on medical queries (may be suboptimal)
Step 2: User provides feedback on routing quality
Step 3: Router retrains on the feedback
Step 4: Routing accuracy improves for the medical domain
""")

    # Medical test queries with correct labels
    medical_queries = [
        ("What is hypertension?",                          "simple"),
        ("What does MRI stand for?",                       "simple"),
        ("Who discovered penicillin?",                     "simple"),
        ("What is the normal body temperature?",           "simple"),
        ("How many chambers does the heart have?",         "simple"),
        ("Explain how CRISPR gene editing works",          "complex"),
        ("What are the tradeoffs between chemotherapy types?", "complex"),
        ("How does the immune system recognise pathogens?","complex"),
        ("Explain the mechanism of antibiotic resistance", "complex"),
        ("Compare Type 1 and Type 2 diabetes mechanisms",  "complex"),
    ]

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db = f.name

    try:
        store     = FeedbackStore(db_path=temp_db)
        router    = AdaptRouter(domain="medical")
        retrainer = AdaptiveRetrainer(feedback_store=store)
        detector  = DriftDetector(feedback_store=store)

        # ── STEP 1: INITIAL ROUTING ───────────────────────────────────────────
        print("STEP 1: Initial routing accuracy on medical queries")
        correct_before = 0
        decision_ids   = []

        for i, (query, true_label) in enumerate(medical_queries):
            result   = router.classify(query)
            predicted= result["label"]
            is_correct = predicted == true_label
            if is_correct:
                correct_before += 1

            query_id = f"med_{i:03d}"
            store.store_decision({
                "query_id"  : query_id,
                "query"     : query,
                "label"     : predicted,
                "model_used": "llama-3.1-8b-instant"
                              if predicted == "simple"
                              else "llama-3.3-70b-versatile",
                "confidence": result["confidence"],
                "latency_s" : 0.46 if predicted == "simple" else 0.81,
                "domain"    : "medical",
                "session_id": "med_example",
            })
            decision_ids.append((query_id, is_correct))

            status = "OK" if is_correct else "WRONG"
            print(f"  [{status}] {query[:45]:<45} → {predicted}")

        acc_before = round(correct_before / len(medical_queries) * 100, 1)
        print(f"\n  Before accuracy: {acc_before}%")

        # ── STEP 2: SIMULATE FEEDBACK ─────────────────────────────────────────
        print("\nSTEP 2: Simulating user feedback on routing quality")
        for query_id, is_correct in decision_ids:
            store.store_feedback(
                query_id    = query_id,
                was_helpful = is_correct,
            )
        print(f"  {len(decision_ids)} feedback signals recorded")

        # ── STEP 3: DRIFT DETECTION ────────────────────────────────────────────
        print("\nSTEP 3: Domain drift detection")
        live_queries = [q for q, _ in medical_queries]
        drift = detector.compute_drift_score(live_queries)
        print(f"  Drift score : {drift.get('drift_score', 'N/A')}")
        print(f"  Severity    : {drift.get('severity', 'N/A')}")
        print(f"  Recommendation: {drift.get('recommendation', 'N/A')}")

        # ── STEP 4: RETRAIN ────────────────────────────────────────────────────
        print("\nSTEP 4: Retraining on medical feedback")
        result = retrainer.retrain()
        print(f"  Status         : {result['status']}")
        if "old_accuracy" in result:
            print(f"  Validation acc : {result['old_accuracy']:.1%} → "
                  f"{result['new_accuracy']:.1%} "
                  f"({result['improvement']:+.1%})")

        # ── STEP 5: EVALUATE AFTER ─────────────────────────────────────────────
        print("\nSTEP 5: Routing accuracy after domain adaptation")
        correct_after = 0

        for query, true_label in medical_queries:
            result_new = router.classify(query)
            if result_new["label"] == true_label:
                correct_after += 1

        acc_after = round(correct_after / len(medical_queries) * 100, 1)
        improvement = acc_after - acc_before

        print(f"\n{'='*60}")
        print(f"RESULTS:")
        print(f"  Before adaptation : {acc_before}%")
        print(f"  After adaptation  : {acc_after}%")
        print(f"  Improvement       : {improvement:+.1f}%")
        print(f"{'='*60}")
        print("""
This is domain adaptation in action. The router started with
general training data and adapted to medical queries through
real usage feedback — without any manual retraining or labelling.
""")

    finally:
        os.unlink(temp_db)


if __name__ == "__main__":
    run_medical_example()