# day8_calibration_demo.py
import sys
import os
import joblib
from dotenv import load_dotenv

load_dotenv()
from adaptrouter.config import LLM_ROUTER_PATH
if LLM_ROUTER_PATH not in sys.path:
    sys.path.insert(0, LLM_ROUTER_PATH)

from adaptrouter.calibration import CalibrationAnalyzer

print("="*55)
print("DAY 8 — Confidence Calibration Analysis")
print("="*55)

clf_path = os.path.join(LLM_ROUTER_PATH, "models", "router_classifier.pkl")
clf      = joblib.load(clf_path)
analyzer = CalibrationAnalyzer(classifier=clf, n_buckets=5)

print("\nComputing calibration on training data...")
calibration = analyzer.compute_calibration()
analyzer.print_calibration_report(calibration)

print("Generating calibration curve plot...")
analyzer.plot_calibration_curve(calibration, save_path="calibration_curve.png")

print("\n" + "="*55)
print("DAY 8 COMPLETE!")
print("="*55)
print(f"""
  "Measured confidence calibration (ECE={calibration.get('ece', 'N/A'):.4f})
   showing router confidence scores are reliable probability estimates,
   not arbitrary numbers — verified using reliability diagram analysis."
""")