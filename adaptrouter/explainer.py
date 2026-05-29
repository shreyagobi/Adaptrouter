# adaptrouter/explainer.py
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from adaptrouter.config import LLM_ROUTER_PATH
from adaptrouter.config import SHAP_DIRECTION_THRESHOLD

if LLM_ROUTER_PATH not in sys.path:
    sys.path.insert(0, LLM_ROUTER_PATH)

try:
    import shap
    from src.embedder import embed, embed_batch
    from data.training_queries import TRAINING_DATA
    _AVAILABLE = True
except ImportError as e:
    _AVAILABLE = False
    print(f"Explainer dependencies missing: {e}")


class RoutingExplainer:
    """
    Explains WHY the router made each routing decision using SHAP values.

    HOW SHAP VALUES ARE CALCULATED:
    SHAP uses the concept of Shapley values from cooperative game theory.
    Each embedding dimension is a "player" in a game.
    The "payout" is the classification score.
    SHAP asks: how much did each dimension contribute to the final score?

    For logistic regression, SHAP has an exact linear formula:
      SHAP_i = coefficient_i × (feature_i - baseline_i)

    Where:
      coefficient_i = the weight the classifier learned for dimension i
      feature_i     = the actual value of dimension i for this query
      baseline_i    = the average value of dimension i across training data

    Dimensions with large |SHAP| values are the most influential.
    Positive SHAP → pushed toward complex
    Negative SHAP → pushed toward simple

    WHY we also show word-level explanations:
    Embedding dimensions are not human-readable ("dimension 247 = +0.034"
    means nothing). We bridge this gap by:
    1. Computing SHAP values for the full query embedding
    2. Also computing SHAP for each word's individual embedding
    3. Showing which words have the most complex-leaning embeddings
    This is an approximation but gives actionable human-readable insight.
    """

    def __init__(self, classifier, background_size: int = 20):
        """
        classifier     : the trained LogisticRegression from retrainer
        background_size: number of background samples for SHAP baseline
                         More = more accurate but slower. 20 is good balance.
        """
        if not _AVAILABLE:
            raise ImportError("shap and sentence-transformers required")

        self.classifier     = classifier
        self._explainer     = None
        self._background    = None
        self._background_size = background_size
        self._word_cache    = {}   # cache word embeddings for speed

        self._build_explainer()


    def _build_explainer(self):
        """
        Builds the SHAP explainer using training data as background.

        The background dataset is the reference point — SHAP measures
        how much each feature deviates from the background average and
        how much that deviation affects the prediction.

        WHY use training data as background?
        It represents the "average query" in the system.
        SHAP values then show how THIS query differs from average
        in ways that affected the routing decision.
        """
        try:
            texts = [d[0] for d in TRAINING_DATA[:self._background_size]]
            X_bg  = embed_batch(texts)

            # LinearExplainer is exact for logistic regression —
            # no approximation needed unlike TreeExplainer or DeepExplainer
            self._explainer  = shap.LinearExplainer(
                self.classifier, X_bg,
                feature_perturbation="interventional"
            )
            self._background = X_bg
            print(f"  Explainer: SHAP LinearExplainer built "
                  f"({self._background_size} background samples)")
        except Exception as e:
            print(f"  Explainer: build failed ({e})")


    def explain(self, query: str, top_k: int = 10) -> dict:
        """
        Generates a full SHAP explanation for one routing decision.

        Returns:
          query           : the input query
          label           : simple or complex
          confidence      : classifier confidence
          shap_summary    : overall explanation in plain English
          top_dimensions  : top_k most influential embedding dimensions
          word_scores     : word-level complexity scores
          decision_factors: human-readable factors that drove the decision
        """
        if self._explainer is None:
            return {"error": "Explainer not initialised"}

        try:
            # Step 1: embed the query
            query_emb = embed(query).reshape(1, -1)

            # Step 2: get prediction
            proba     = self.classifier.predict_proba(query_emb)[0]
            p_simple  = round(float(proba[0]), 4)
            p_complex = round(float(proba[1]), 4)
            label     = "simple" if p_simple >= p_complex else "complex"
            confidence= max(p_simple, p_complex)

            # Step 3: compute SHAP values
            # shap_values shape: (1, 384, 2) → one set per class
            # We care about class 1 (complex) — positive = more complex
            shap_values  = self._explainer.shap_values(query_emb)

            # For binary classification, shap_values is a list of 2 arrays
            # shap_values[1] = SHAP for "complex" class
            if isinstance(shap_values, list):
                shap_complex = shap_values[1][0]  # shape: (384,)
            else:
                shap_complex = shap_values[0]

            # Step 4: find most influential dimensions
            abs_shap     = np.abs(shap_complex)
            top_indices  = np.argsort(abs_shap)[::-1][:top_k]

            top_dimensions = []
            for idx in top_indices:
                top_dimensions.append({
                    "dimension"   : int(idx),
                    "shap_value"  : round(float(shap_complex[idx]), 6),
                    "direction"   : "→ complex" if shap_complex[idx] > 0
                                    else "→ simple",
                    "magnitude"   : round(float(abs_shap[idx]), 6),
                })

            # Step 5: word-level scores
            word_scores = self._compute_word_scores(query, shap_complex)

            # Step 6: generate human-readable summary
            summary = self._generate_summary(
                label, confidence, top_dimensions, word_scores
            )

            return {
                "query"          : query,
                "label"          : label,
                "confidence"     : round(confidence, 4),
                "p_simple"       : p_simple,
                "p_complex"      : p_complex,
                "shap_summary"   : summary,
                "top_dimensions" : top_dimensions,
                "word_scores"    : word_scores,
                "total_shap_mass": round(float(np.sum(np.abs(shap_complex))), 4),
                "decision_factors": self._extract_decision_factors(
                    word_scores, label
                ),
            }

        except Exception as e:
            return {"error": str(e), "query": query}


    def _compute_word_scores(self, query: str, shap_complex: np.ndarray) -> list:
        """
        Approximates word-level complexity contribution.

        METHOD:
        1. Split query into words
        2. Embed each word individually
        3. Project each word embedding onto the SHAP direction
           (dot product with shap_complex)
        4. Words with high positive projection = push toward complex
           Words with high negative projection = push toward simple

        WHY this is an approximation:
        The true SHAP value is for the FULL query embedding, not individual
        words. Word embeddings combine non-linearly in sentence embeddings.
        But this approximation gives genuinely useful signal — words like
        "tradeoffs", "explain", "architecture" consistently score high
        on the complex direction across many queries.

        HOW THE SCORE IS CALCULATED:
          word_score = dot(embed(word), shap_complex) / ||shap_complex||

        This measures how much the word's embedding aligns with the
        "direction of complexity" in embedding space.
        """
        words  = query.split()
        scores = []

        # Normalise SHAP vector for stable dot products
        shap_norm  = np.linalg.norm(shap_complex)
        if shap_norm == 0:
            return scores
        shap_unit  = shap_complex / shap_norm

        for word in words:
            word_clean = word.lower().strip("?,.")
            if len(word_clean) < 2:
                continue

            # Cache word embeddings for repeated words
            if word_clean not in self._word_cache:
                try:
                    self._word_cache[word_clean] = embed(word_clean)
                except Exception:
                    continue

            word_emb  = self._word_cache[word_clean]
            score     = float(np.dot(word_emb, shap_unit))

            scores.append({
                "word"      : word_clean,
                "score"     : round(score, 4),
               "direction": "→ complex" if score > SHAP_DIRECTION_THRESHOLD
             else ("→ simple" if score < -SHAP_DIRECTION_THRESHOLD
             else "neutral"),
            })

        # Sort by absolute score — most influential words first
        scores.sort(key=lambda x: abs(x["score"]), reverse=True)
        return scores[:10]   # top 10 words


    def _generate_summary(self, label: str, confidence: float,
                          top_dims: list, word_scores: list) -> str:
        """Generates a plain English explanation of the routing decision."""
        direction  = "complex" if label == "complex" else "simple"
        conf_pct   = round(confidence * 100, 1)

        # Find words that pushed toward the winning direction
        if label == "complex":
            drivers = [w for w in word_scores if w["direction"] == "→ complex"]
        else:
            drivers = [w for w in word_scores if w["direction"] == "→ simple"]

        driver_words = [d["word"] for d in drivers[:3]]

        if driver_words:
            word_part = f"Key words driving this: {', '.join(driver_words)}."
        else:
            word_part = "Decision based on overall query structure."

        top_shap     = top_dims[0] if top_dims else None
        dim_part     = (f"Strongest signal from embedding dimension "
                        f"{top_shap['dimension']} "
                        f"(SHAP={top_shap['shap_value']:+.4f})."
                        if top_shap else "")

        return (
            f"Classified as {direction.upper()} with {conf_pct}% confidence. "
            f"{word_part} {dim_part}"
        )


    def _extract_decision_factors(self, word_scores: list,
                                  label: str) -> list:
        """Extracts human-readable decision factors from word scores."""
        factors = []

        complex_words = [w for w in word_scores if w["direction"] == "→ complex"]
        simple_words  = [w for w in word_scores if w["direction"] == "→ simple"]

        if complex_words:
            factors.append({
                "factor"    : "complexity signals",
                "words"     : [w["word"] for w in complex_words[:3]],
                "effect"    : "pushed toward smart model",
                "strength"  : round(sum(w["score"] for w in complex_words[:3]), 3),
            })

        if simple_words:
            factors.append({
                "factor"    : "simplicity signals",
                "words"     : [w["word"] for w in simple_words[:3]],
                "effect"    : "pushed toward fast model",
                "strength"  : round(abs(sum(w["score"] for w in simple_words[:3])), 3),
            })

        return factors


    def explain_batch(self, queries: list) -> list:
        """Explains multiple routing decisions efficiently."""
        return [self.explain(q) for q in queries]


    def print_explanation(self, explanation: dict):
        """Pretty-prints an explanation to terminal."""
        if "error" in explanation:
            print(f"Explanation error: {explanation['error']}")
            return

        print(f"\n{'─'*60}")
        print(f"Query      : {explanation['query']}")
        print(f"Decision   : {explanation['label'].upper()} "
              f"({explanation['confidence']*100:.1f}% confident)")
        print(f"Summary    : {explanation['shap_summary']}")

        print(f"\nWord scores (most influential first):")
        for w in explanation["word_scores"][:6]:
            bar = "█" * min(int(abs(w["score"]) * 20), 20)
            print(f"  {w['word']:<20} {w['score']:+.4f} {bar} {w['direction']}")

        print(f"\nDecision factors:")
        for f in explanation["decision_factors"]:
            print(f"  {f['factor']}: {', '.join(f['words'])} "
                  f"({f['effect']}, strength={f['strength']})")
        print(f"{'─'*60}")