# adaptrouter/calibration.py
import sys
import numpy as np
import json
import os
from adaptrouter.config import LLM_ROUTER_PATH

if LLM_ROUTER_PATH not in sys.path:
    sys.path.insert(0, LLM_ROUTER_PATH)

try:
    from src.embedder import embed_batch
    from data.training_queries import TRAINING_DATA
    _BASE_AVAILABLE = True
except ImportError:
    _BASE_AVAILABLE = False


class CalibrationAnalyzer:
    """
    Measures and visualises confidence calibration of the routing classifier.

    WHAT IS CALIBRATION?
    A classifier is perfectly calibrated if, across all predictions where
    it says "I am X% confident," it is actually correct X% of the time.

    Example:
      Well calibrated:   says 80% confident → correct 80% of the time
      Overconfident:     says 90% confident → correct 70% of the time
      Underconfident:    says 60% confident → correct 80% of the time

    HOW ECE (Expected Calibration Error) IS CALCULATED:
    1. Group predictions into confidence buckets: [0.5-0.6], [0.6-0.7], etc.
    2. For each bucket: measure actual accuracy within that bucket
    3. ECE = weighted average of |confidence - accuracy| across buckets
       ECE = Σ (|bucket| / N) × |avg_confidence_in_bucket - accuracy_in_bucket|

    Perfect calibration = ECE of 0.0
    Typical ML models  = ECE of 0.05-0.15
    Your target        = ECE < 0.10

    WHY CALIBRATION MATTERS FOR A ROUTER:
    If confidence scores are unreliable, your threshold (0.65) is meaningless.
    You need: "when I say 65% confident it's simple, it really is 65% likely
    to be simple." Otherwise the threshold is just a number you picked randomly.
    """

    def __init__(self, classifier, n_buckets: int = 10):
        self.classifier = classifier
        self.n_buckets  = n_buckets


    def compute_calibration(self, queries: list = None,
                            labels: list = None) -> dict:
        """
        Computes calibration metrics on a dataset.

        If no queries provided, uses full training data.
        Returns ECE, calibration curve data, and per-bucket stats.

        HOW EACH METRIC IS CALCULATED:

        confidence: max(P(simple), P(complex)) from classifier
        predicted : argmax of probabilities
        correct   : 1 if predicted == true_label else 0

        bucket_confidence: mean confidence of all predictions in bucket
        bucket_accuracy  : mean correctness of all predictions in bucket
        bucket_weight    : fraction of total predictions in this bucket
        bucket_ece       : weight × |bucket_confidence - bucket_accuracy|

        ECE = sum of all bucket_ece values
        """
        if not _BASE_AVAILABLE:
            return {"error": "base router not available"}

        # Use training data if none provided
        if queries is None:
            queries = [d[0] for d in TRAINING_DATA]
            labels  = [d[1] for d in TRAINING_DATA]

        try:
            X          = embed_batch(queries)
            y_true     = np.array(labels)
            proba      = self.classifier.predict_proba(X)
            y_pred     = self.classifier.predict(X)
            confidences= np.max(proba, axis=1)

            # Compute per-sample correctness
            correct = (y_pred == y_true).astype(float)

            # Build calibration curve
            bucket_edges  = np.linspace(0.5, 1.0, self.n_buckets + 1)
            bucket_data   = []
            ece           = 0.0
            n_total       = len(queries)

            for i in range(self.n_buckets):
                low, high = bucket_edges[i], bucket_edges[i + 1]
                mask      = (confidences >= low) & (confidences < high)

                if mask.sum() == 0:
                    continue

                bucket_conf = confidences[mask].mean()
                bucket_acc  = correct[mask].mean()
                bucket_n    = mask.sum()
                weight      = bucket_n / n_total
                bucket_ece  = weight * abs(bucket_conf - bucket_acc)
                ece        += bucket_ece

                bucket_data.append({
                    "confidence_range": f"{low:.2f}-{high:.2f}",
                    "avg_confidence"  : round(float(bucket_conf), 4),
                    "actual_accuracy" : round(float(bucket_acc), 4),
                    "n_samples"       : int(bucket_n),
                    "weight"          : round(float(weight), 4),
                    "bucket_ece"      : round(float(bucket_ece), 6),
                    "calibration_gap" : round(float(bucket_conf - bucket_acc), 4),
                })

            overall_accuracy = float(correct.mean())
            avg_confidence   = float(confidences.mean())

            return {
                "ece"               : round(float(ece), 6),
                "overall_accuracy"  : round(overall_accuracy, 4),
                "avg_confidence"    : round(avg_confidence, 4),
                "overconfidence"    : round(avg_confidence - overall_accuracy, 4),
                "n_samples"         : n_total,
                "bucket_data"       : bucket_data,
                "interpretation"    : self._interpret_ece(ece),
            }

        except Exception as e:
            return {"error": str(e)}


    def _interpret_ece(self, ece: float) -> str:
        """Interprets ECE value in plain English."""
        if ece < 0.05:
            return "Excellent calibration — confidence scores are very reliable"
        elif ece < 0.10:
            return "Good calibration — confidence scores are mostly reliable"
        elif ece < 0.15:
            return "Fair calibration — consider Platt scaling to improve"
        else:
            return "Poor calibration — confidence scores may be misleading"


    def plot_calibration_curve(self, calibration_data: dict,
                               save_path: str = None):
        """
        Plots reliability diagram (calibration curve).

        The diagonal line represents perfect calibration.
        Points above diagonal = underconfident (actual accuracy > predicted)
        Points below diagonal = overconfident (actual accuracy < predicted)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            if "error" in calibration_data or not calibration_data.get("bucket_data"):
                print("No calibration data to plot")
                return

            buckets    = calibration_data["bucket_data"]
            conf_vals  = [b["avg_confidence"] for b in buckets]
            acc_vals   = [b["actual_accuracy"] for b in buckets]
            weights    = [b["n_samples"] for b in buckets]

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # ── PLOT 1: Reliability diagram ───────────────────────────────────
            ax1 = axes[0]
            ax1.plot([0.5, 1.0], [0.5, 1.0], "k--", label="Perfect calibration",
                     alpha=0.7, linewidth=1.5)

            scatter = ax1.scatter(conf_vals, acc_vals, s=[w*50 for w in weights],
                                  c=weights, cmap="Blues", alpha=0.8,
                                  edgecolors="navy", linewidth=0.5)

            ax1.plot(conf_vals, acc_vals, "b-", alpha=0.4, linewidth=1)
            plt.colorbar(scatter, ax=ax1, label="Number of samples")

            ax1.set_xlabel("Mean Predicted Confidence", fontsize=12)
            ax1.set_ylabel("Actual Accuracy", fontsize=12)
            ax1.set_title(
                f"Reliability Diagram\nECE = {calibration_data['ece']:.4f} — "
                f"{calibration_data['interpretation'].split(' — ')[0]}",
                fontsize=11
            )
            ax1.set_xlim([0.45, 1.05])
            ax1.set_ylim([0.45, 1.05])
            ax1.legend(fontsize=10)
            ax1.grid(alpha=0.3)

            # Shade calibration gap
            for conf, acc in zip(conf_vals, acc_vals):
                color = "red" if conf > acc else "green"
                ax1.annotate("", xy=(conf, acc), xytext=(conf, conf),
                             arrowprops=dict(arrowstyle="-", color=color,
                                             alpha=0.5, linewidth=2))

            # ── PLOT 2: Confidence distribution ───────────────────────────────
            ax2 = axes[1]
            ranges  = [b["confidence_range"] for b in buckets]
            heights = [b["n_samples"] for b in buckets]
            colors  = ["green" if abs(b["calibration_gap"]) < 0.05
                       else "orange" if abs(b["calibration_gap"]) < 0.10
                       else "red" for b in buckets]

            bars = ax2.bar(range(len(ranges)), heights, color=colors, alpha=0.7,
                           edgecolor="white", linewidth=0.5)
            ax2.set_xticks(range(len(ranges)))
            ax2.set_xticklabels(ranges, rotation=45, ha="right", fontsize=9)
            ax2.set_xlabel("Confidence Range", fontsize=12)
            ax2.set_ylabel("Number of Samples", fontsize=12)
            ax2.set_title("Confidence Distribution\n(green=well calibrated, "
                          "orange=mild gap, red=large gap)", fontsize=11)

            green_patch  = mpatches.Patch(color="green",  label="Gap < 5%")
            orange_patch = mpatches.Patch(color="orange", label="Gap 5-10%")
            red_patch    = mpatches.Patch(color="red",    label="Gap > 10%")
            ax2.legend(handles=[green_patch, orange_patch, red_patch], fontsize=9)
            ax2.grid(alpha=0.3, axis="y")

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"  Calibration curve saved to: {save_path}")
            else:
                plt.savefig("calibration_curve.png", dpi=150, bbox_inches="tight")
                print("  Calibration curve saved to: calibration_curve.png")

            plt.close()

        except ImportError:
            print("matplotlib not installed — install with: pip install matplotlib")
        except Exception as e:
            print(f"Plot failed: {e}")


    def print_calibration_report(self, calibration_data: dict):
        """Prints calibration results to terminal."""
        if "error" in calibration_data:
            print(f"Calibration error: {calibration_data['error']}")
            return

        print(f"\n{'='*55}")
        print("CALIBRATION REPORT")
        print(f"{'='*55}")
        print(f"  ECE (Expected Calibration Error) : "
              f"{calibration_data['ece']:.6f}")
        print(f"  Overall accuracy                 : "
              f"{calibration_data['overall_accuracy']:.1%}")
        print(f"  Average confidence               : "
              f"{calibration_data['avg_confidence']:.1%}")
        print(f"  Overconfidence bias              : "
              f"{calibration_data['overconfidence']:+.1%}")
        print(f"  Interpretation                   : "
              f"{calibration_data['interpretation']}")

        print(f"\n  Per-bucket calibration:")
        print(f"  {'Range':<12} {'Conf':>8} {'Acc':>8} {'Gap':>8} {'N':>5}")
        print(f"  {'─'*45}")

        for b in calibration_data["bucket_data"]:
            gap_str = f"{b['calibration_gap']:+.3f}"
            color   = "" if abs(b["calibration_gap"]) < 0.05 else "  ← gap"
            print(f"  {b['confidence_range']:<12} "
                  f"{b['avg_confidence']:>8.3f} "
                  f"{b['actual_accuracy']:>8.3f} "
                  f"{gap_str:>8} "
                  f"{b['n_samples']:>5}{color}")

        print(f"{'='*55}\n")