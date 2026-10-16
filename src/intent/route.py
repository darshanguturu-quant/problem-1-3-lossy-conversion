"""
Standalone Fast Inference Router for Task 2 User Intent Classification.
Provides direct command-line query routing and zero-overhead production prediction.
"""

import sys
import os
import json
import argparse
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.intent.cascade import CascadedIntentClassifier
from src.intent.stage1 import Stage1IntentClassifier
from src.intent.stage2 import Stage2IntentClassifier


def load_router(
    stage1_path: str = "artifacts/intent_stage1.joblib",
    stage2_path: str = "artifacts/intent_stage2.joblib",
    escalation_threshold: float = 0.82,
    margin_threshold: float = 0.30,
    ambiguity_threshold: float = 0.45,
    ambiguity_mode: str = "ambiguous"
) -> CascadedIntentClassifier:
    """Load pre-trained Stage 1 and Stage 2 models into CascadedIntentClassifier."""
    if not os.path.exists(stage1_path) or not os.path.exists(stage2_path):
        raise FileNotFoundError(
            f"Trained intent models not found at {stage1_path} or {stage2_path}. "
            "Please run 'python src/main.py train-intent' first."
        )

    s1 = Stage1IntentClassifier.load(stage1_path)
    s2 = Stage2IntentClassifier.load(stage2_path)
    router = CascadedIntentClassifier(
        s1,
        s2,
        escalation_threshold=escalation_threshold,
        margin_threshold=margin_threshold,
        ambiguity_threshold=ambiguity_threshold,
        ambiguity_mode=ambiguity_mode
    )
    return router


def route_query_cli(query: str, model_dir: str = "artifacts") -> Dict[str, Any]:
    """Route a single query and return structured JSON result."""
    s1_path = os.path.join(model_dir, "intent_stage1.joblib")
    s2_path = os.path.join(model_dir, "intent_stage2.joblib")
    router = load_router(s1_path, s2_path)
    result = router.predict(query)
    return result.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Route query to minimum sufficient temporal granularity.")
    parser.add_argument("--query", "-q", type=str, required=True, help="Natural language search query")
    parser.add_argument("--model-dir", type=str, default="artifacts", help="Path to artifacts containing trained models")
    parser.add_argument("--json", action="store_true", help="Output pure JSON format")
    args = parser.parse_args()

    res = route_query_cli(args.query, model_dir=args.model_dir)

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"Query: {res['query']}")
        print(f"Predicted Intent: {res['predicted_intent']} (confidence: {res['confidence']:.4f})")
        print(f"Resolved At: {res['stage_resolved']} (Escalated: {res['escalated_to_stage2']})")
        print(f"Routing Target: {res['routing_target']} (Latency: {res['latency_ms']:.2f} ms)")
        print(f"Probabilities: {res['probabilities']}")


if __name__ == "__main__":
    main()
