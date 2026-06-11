"""LangSmith evaluation entry point.

Run with:
    python -m yt_rag.evaluation.run_eval --dataset yt-rag-PLxxx --user eval_user
"""
from __future__ import annotations

import argparse

from langsmith.evaluation import evaluate

from yt_rag.config import settings
from yt_rag.evaluation.judges import (
    citation_precision_evaluator,
    citation_presence_evaluator,
    faithfulness_evaluator,
    refusal_accuracy_evaluator,
)
from yt_rag.graph.rag_graph import rag_graph


def _build_predict():
    def predict(inputs: dict) -> dict:
        state = {
            "question":         inputs["question"],
            "playlist_id":      inputs.get("playlist_id", "default"),
            "user_id":          inputs.get("user_id", "eval_user"),
            "metadata_filter":  None,
            "retrieved_docs":   [],
            "reranked_docs":    [],
            "min_rerank_score": settings.min_rerank_score,
            "threshold_passed": False,
            "refusal_reason":   None,
            "answer":           None,
            "citations":        [],
            "final_response":   None,
        }
        return rag_graph.invoke(state)

    return predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation against a LangSmith dataset")
    parser.add_argument("--dataset", required=True, help="LangSmith dataset name (e.g. yt-rag-PLxxx)")
    parser.add_argument("--prefix", default="yt-rag-eval", help="Experiment name prefix")
    args = parser.parse_args()

    results = evaluate(
        _build_predict(),
        data=args.dataset,
        evaluators=[
            faithfulness_evaluator(),
            citation_precision_evaluator(),
            citation_presence_evaluator(),
            refusal_accuracy_evaluator(),
        ],
        experiment_prefix=args.prefix,
        metadata={"min_rerank_score": settings.min_rerank_score},
    )
    print(results)


if __name__ == "__main__":
    main()
