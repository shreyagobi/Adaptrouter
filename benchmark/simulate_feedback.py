# benchmark/simulate_feedback.py
"""
Simulates a realistic feedback scenario to demonstrate self-improvement.
Useful for testing and demonstrations without real users.

Runs 3 rounds:
  Round 1: Fresh router — baseline accuracy
  Round 2: 20 feedback signals — first retrain
  Round 3: 20 more signals — second retrain

Shows accuracy improving across rounds.
"""
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptrouter.feedback_store import FeedbackStore
from adaptrouter.feedback_collector import FeedbackCollector
from adaptrouter.retrainer import AdaptiveRetrainer
from adaptrouter.drift_detector import DriftDetector


SIMULATION_QUERIES = [
    # Domain-shifted queries — medical context
    # These are NOT in the original training data
    # Router will initially misroute some — feedback corrects it
    ("What is hypertension?",                          "simple"),
    ("What does an MRI scan show?",                    "simple"),
    ("Who discovered penicillin?",                     "simple"),
    ("What is the normal human body temperature?",     "simple"),
    ("How many chambers does the heart have?",         "simple"),
    ("Explain how CRISPR gene editing works",          "complex"),
    ("What are the tradeoffs between chemotherapy types?", "complex"),
    ("How does the immune system recognise pathogens?","complex"),
    ("Explain the mechanism of antibiotic resistance", "complex"),
    ("What is the difference between Type 1 and Type 2 diabetes?", "complex"),
]


def run_simulation():
    print("="*60)
    print("FEEDBACK SIMULATION — 3 rounds of improvement")
    print("="*60)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db = f.name

    store     = FeedbackStore(db_path=temp_db)
    retrainer = AdaptiveRetrainer(feedback_store=store)
    detector  = DriftDetector(feedback_store=store)

    try:
        from adaptrouter import AdaptRouter
        router = AdaptRouter(domain="medical_simulation")
    except Exception as e:
        print(f"Could not load AdaptRouter: {e}")
        os.unlink(temp_db)
        return

    accuracy_history = []

    for round_num in range(1, 4):
        print(f"\n--- Round {round_num} ---")

        # Evaluate current routing accuracy
        correct = 0
        decisions = []
        for i, (query, true_label) in enumerate(SIMULATION_QUERIES):
            clf_result = router.classify(query)
            predicted  = clf_result["label"]
            is_correct = predicted == true_label
            if is_correct:
                correct += 1

            query_id = f"sim_r{round_num}_{i:02d}"

            # Store decision
            store.store_decision({
                "query_id"  : query_id,
                "query"     : query,
                "label"     : predicted,
                "model_used": "llama-3.1-8b-instant" if predicted == "simple"
                              else "llama-3.3-70b-versatile",
                "confidence": clf_result["confidence"],
                "latency_s" : 0.46 if predicted == "simple" else 0.81,
                "domain"    : "medical_simulation",
                "session_id": f"sim_round_{round_num}",
            })

            # Store feedback (honest — based on whether routing was correct)
            store.store_feedback(
                query_id    = query_id,
                was_helpful = is_correct,
            )

            decisions.append((query_id, is_correct))

        accuracy = round(correct / len(SIMULATION_QUERIES) * 100, 1)
        accuracy_history.append(accuracy)
        new_labelled = store.count_new_labelled()

        print(f"  Accuracy       : {accuracy}% ({correct}/{len(SIMULATION_QUERIES)})")
        print(f"  New labelled   : {new_labelled}")

        # Retrain if threshold reached (override threshold for simulation)
        if new_labelled >= 5:
            print(f"  Triggering retrain...")
            result = retrainer.retrain()
            print(f"  Retrain result : {result['status']}")
            if "improvement" in result:
                print(f"  Improvement    : {result['improvement']:+.1%}")

    # Drift detection on simulation queries
    print(f"\n--- Drift Detection ---")
    live_queries = [q for q, _ in SIMULATION_QUERIES]
    drift = detector.compute_drift_score(live_queries)
    print(f"  Drift score : {drift.get('drift_score', 'N/A')}")
    print(f"  Severity    : {drift.get('severity', 'N/A')}")
    print(f"  Alert       : {drift.get('alert', False)}")
    print(f"  Note        : Medical queries are outside CS/ML training distribution")
    print(f"  Action      : {drift.get('recommendation', 'N/A')}")

    print(f"\n--- Accuracy Trend ---")
    for i, acc in enumerate(accuracy_history, 1):
        bar = "█" * int(acc / 5)
        print(f"  Round {i}: {acc:5.1f}% {bar}")

    print(f"\n--- Summary ---")
    if len(accuracy_history) >= 2:
        total_improvement = accuracy_history[-1] - accuracy_history[0]
        print(f"  Start accuracy   : {accuracy_history[0]}%")
        print(f"  Final accuracy   : {accuracy_history[-1]}%")
        print(f"  Total improvement: {total_improvement:+.1f}%")

    os.unlink(temp_db)
    print(f"\nSimulation complete!")


if __name__ == "__main__":
    run_simulation()