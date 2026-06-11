from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith.evaluation import LangChainStringEvaluator

from yt_rag.config import settings


def _eval_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.gemini_flash_model,
        google_api_key=settings.google_api_key,
        temperature=0.0,
    )


def faithfulness_evaluator() -> LangChainStringEvaluator:
    """LLM-as-a-judge: is the answer fully grounded in the retrieved context?

    Target: ≥ 90% of answers score 1.
    """
    return LangChainStringEvaluator(
        "criteria",
        config={
            "criteria": {
                "faithfulness": (
                    "Does the answer contain ONLY information present in the retrieved "
                    "context? Score 1 if every claim in the answer is supported by the "
                    "context. Score 0 if any claim goes beyond or contradicts the context."
                )
            },
            "llm": _eval_llm(),
        },
        prepare_data=lambda run, example: {
            "prediction": (run.outputs or {}).get("answer", ""),
            "input": (run.inputs or {}).get("question", ""),
            "reference": (run.outputs or {}).get("context", ""),
        },
    )


def relevance_evaluator() -> LangChainStringEvaluator:
    """LLM-as-a-judge: does the answer directly address the question?

    Target: ≥ 85% of answers score 1.
    """
    return LangChainStringEvaluator(
        "criteria",
        config={
            "criteria": {
                "relevance": (
                    "Does the answer directly and completely address the question asked? "
                    "Score 1 if it fully answers the question. Score 0 if it is off-topic, "
                    "incomplete, or a refusal when a real answer was possible."
                )
            },
            "llm": _eval_llm(),
        },
        prepare_data=lambda run, example: {
            "prediction": (run.outputs or {}).get("answer", ""),
            "input": (run.inputs or {}).get("question", ""),
            "reference": (example.outputs or {}).get("answer", ""),
        },
    )


def citation_precision_evaluator():
    """Heuristic evaluator: fraction of citations matching a reranked source title."""

    def evaluate_fn(run, example):
        outputs = run.outputs or {}
        citations = outputs.get("citations", [])
        docs = outputs.get("reranked_docs", [])
        if not citations:
            # No citations — vacuously correct (may also indicate a refusal)
            return {"key": "citation_precision", "score": 1.0}
        source_titles = {d.metadata.get("title", "") for d in docs}
        matches = sum(1 for c in citations if c.get("title") in source_titles)
        return {"key": "citation_precision", "score": matches / len(citations)}

    return evaluate_fn
