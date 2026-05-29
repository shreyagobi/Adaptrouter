# day7_explainer_demo.py
import sys
import os
import joblib

# Add llm-router to path
from dotenv import load_dotenv
load_dotenv()
from adaptrouter.config import LLM_ROUTER_PATH
if LLM_ROUTER_PATH not in sys.path:
    sys.path.insert(0, LLM_ROUTER_PATH)

from adaptrouter.explainer import RoutingExplainer

print("="*60)
print("DAY 7 — SHAP Routing Explainer Demo")
print("="*60)

# Load classifier
clf_path = os.path.join(LLM_ROUTER_PATH, "models", "router_classifier.pkl")
clf      = joblib.load(clf_path)

print("\nBuilding SHAP explainer (takes ~10s first time)...")
explainer = RoutingExplainer(classifier=clf, background_size=20)

# Test queries — mix of obvious and tricky
test_queries = [
    "What is the capital of Japan?",
    "Explain gradient descent step by step",
    "What is recursion?",              # tricky
    "Explain what 2+2 is",            # tricky
    "What are the tradeoffs between SQL and NoSQL?",
    "Who wrote Harry Potter?",
]

print("\n" + "="*60)
print("ROUTING EXPLANATIONS")
print("="*60)

for query in test_queries:
    explanation = explainer.explain(query)
    explainer.print_explanation(explanation)

print("\n" + "="*60)
print("DAY 7 COMPLETE — SHAP explainability working!")
print("="*60)
print("""

  "Unlike RouteLLM which is a black box, my router explains
   every routing decision — showing which words pushed it
   toward the fast or smart model, using SHAP values from
   the logistic regression classifier."
""")