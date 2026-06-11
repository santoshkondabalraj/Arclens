from __future__ import annotations

from functools import lru_cache

from yt_rag.graph.rag_graph import rag_graph as _rag_graph


@lru_cache(maxsize=1)
def get_rag_graph():
    """Return the compiled LangGraph singleton."""
    return _rag_graph
