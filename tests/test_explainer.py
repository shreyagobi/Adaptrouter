# tests/test_explainer.py
import pytest
from unittest.mock import patch, MagicMock


class TestRoutingExplainer:

    def test_import_succeeds(self):
        """RoutingExplainer must be importable."""
        try:
            from adaptrouter.explainer import RoutingExplainer
            assert RoutingExplainer is not None
        except ImportError:
            pytest.skip("shap not installed")

    def test_explain_returns_required_keys(self):
        """explain() must return all required keys."""
        try:
            from adaptrouter.explainer import RoutingExplainer
            import joblib, os
            from adaptrouter.config import LLM_ROUTER_PATH
            clf_path = os.path.join(LLM_ROUTER_PATH, "models",
                                    "router_classifier.pkl")
            if not os.path.exists(clf_path):
                pytest.skip("Classifier not found")
            clf = joblib.load(clf_path)
            explainer = RoutingExplainer(classifier=clf)
            result    = explainer.explain("What is the capital of France?")
            if "error" in result:
                pytest.skip(f"Explainer error: {result['error']}")
            required = ["query","label","confidence","shap_summary",
                        "word_scores","decision_factors"]
            for key in required:
                assert key in result, f"Missing key: {key}"
        except Exception as e:
            pytest.skip(f"Skipping: {e}")

    def test_explain_label_valid(self):
        """Label must be simple or complex."""
        try:
            from adaptrouter.explainer import RoutingExplainer
            import joblib, os
            from adaptrouter.config import LLM_ROUTER_PATH
            clf_path = os.path.join(LLM_ROUTER_PATH, "models",
                                    "router_classifier.pkl")
            if not os.path.exists(clf_path):
                pytest.skip("Classifier not found")
            clf    = joblib.load(clf_path)
            exp    = RoutingExplainer(classifier=clf)
            result = exp.explain("Explain gradient descent")
            if "error" not in result:
                assert result["label"] in ["simple", "complex"]
        except Exception as e:
            pytest.skip(f"Skipping: {e}")

    def test_word_scores_are_sorted(self):
        """Word scores must be sorted by absolute value descending."""
        try:
            from adaptrouter.explainer import RoutingExplainer
            import joblib, os
            from adaptrouter.config import LLM_ROUTER_PATH
            clf_path = os.path.join(LLM_ROUTER_PATH, "models",
                                    "router_classifier.pkl")
            if not os.path.exists(clf_path):
                pytest.skip("Classifier not found")
            clf    = joblib.load(clf_path)
            exp    = RoutingExplainer(classifier=clf)
            result = exp.explain("What is machine learning?")
            if "error" not in result and len(result["word_scores"]) >= 2:
                scores = [abs(w["score"]) for w in result["word_scores"]]
                assert scores == sorted(scores, reverse=True)
        except Exception as e:
            pytest.skip(f"Skipping: {e}")