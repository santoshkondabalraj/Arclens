"""Evaluation entry point.

Run with:
    python -m yt_rag.evaluation.run_eval
    python -m yt_rag.evaluation.run_eval --dataset yt-rag-golden --user eval_user
"""
from __future__ import annotations

import argparse

from langsmith.evaluation import evaluate

from yt_rag.config import settings
from yt_rag.evaluation.judges import (
    citation_precision_evaluator,
    faithfulness_evaluator,
    relevance_evaluator,
)
from yt_rag.graph.rag_graph import rag_graph


def _build_predict(user_id: str):
    def predict(inputs: dict) -> dict:
        state = {
            "question": inputs["question"],
            "playlist_id": inputs.get("playlist_id", "default"),
            "user_id": user_id,
            "metadata_filter": inputs.get("metadata_filter"),
            "retrieved_docs": [],
            "reranked_docs": [],
            "min_rerank_score": settings.min_rerank_score,
            "threshold_passed": False,
            "refusal_reason": None,
            "answer": None,
            "citations": [],
            "final_response": None,
        }
        return rag_graph.invoke(state)

    return predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation against LangSmith dataset")
    parser.add_argument("--dataset", default="yt-rag-golden", help="LangSmith dataset name")
    parser.add_argument("--user", default="eval_user", help="User ID for namespace resolution")
    parser.add_argument("--prefix", default="yt-rag-eval", help="Experiment name prefix")
    args = parser.parse_args()

    results = evaluate(
        _build_predict(args.user),
        data=args.dataset,
        evaluators=[
            faithfulness_evaluator(),
            relevance_evaluator(),
            citation_precision_evaluator(),
        ],
        experiment_prefix=args.prefix,
        metadata={"min_rerank_score": settings.min_rerank_score},
    )
    print(results)


if __name__ == "__main__":
    main()
