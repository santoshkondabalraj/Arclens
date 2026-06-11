from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith.evaluation import LangChainStringEvaluator

from yt_rag.config import settings


def _eval_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.gemini_flash_model,
        google_api_key=settings.google_api_key,
        temperature=0.0,
        max_output_tokens=4096,
        thinking_budget=0,
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


def citation_precision_evaluator():
    """Heuristic evaluator: fraction of citations matching a reranked source title."""

    def evaluate_fn(run, example):
        outputs = run.outputs or {}
        final = outputs.get("final_response") or {}
        citations = final.get("citations") or []
        docs = outputs.get("reranked_docs") or []
        if not citations:
            return {"key": "citation_precision", "score": 1.0}
        # docs may be Document objects or plain dicts depending on LangSmith serialization
        source_titles = set()
        for d in docs:
            if hasattr(d, "metadata"):
                source_titles.add(d.metadata.get("title", ""))
            else:
                source_titles.add((d.get("metadata") or {}).get("title", ""))
        matches = sum(1 for c in citations if c.get("title") in source_titles)
        return {"key": "citation_precision", "score": matches / len(citations)}

    return evaluate_fn


def citation_presence_evaluator():
    """Heuristic evaluator: does every non-refusal answer include at least one citation?"""

    def evaluate_fn(run, example):
        final = (run.outputs or {}).get("final_response") or {}
        if final.get("refusal"):
            return {"key": "citation_presence", "score": 1.0}
        has_citations = len(final.get("citations") or []) >= 1
        return {"key": "citation_presence", "score": int(has_citations)}

    return evaluate_fn


def refusal_accuracy_evaluator():
    """Heuristic evaluator: did the system refuse iff it was expected to?"""

    def evaluate_fn(run, example):
        expected = (example.outputs or {}).get("expected_refusal", False)
        final = (run.outputs or {}).get("final_response") or {}
        actual = bool(final.get("refusal"))
        return {"key": "refusal_accuracy", "score": int(actual == expected)}

    return evaluate_fn
