from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from yt_rag.graph.nodes import (
    format_citations,
    generate,
    rerank,
    refusal,
    retrieve,
    threshold_check,
)
from yt_rag.graph.state import RAGState


def _route_after_threshold(state: RAGState) -> str:
    return "generate" if state["threshold_passed"] else "refusal"


def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("threshold_check", threshold_check)
    graph.add_node("generate", generate)
    graph.add_node("format_citations", format_citations)
    graph.add_node("refusal", refusal)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "threshold_check")

    graph.add_conditional_edges(
        "threshold_check",
        _route_after_threshold,
        {"generate": "generate", "refusal": "refusal"},
    )

    graph.add_edge("generate", "format_citations")
    graph.add_edge("format_citations", END)
    graph.add_edge("refusal", END)

    return graph.compile()


# Module-level singleton used by the API and evaluation runner
rag_graph = build_rag_graph()
