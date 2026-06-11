from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from yt_rag.config import settings
from yt_rag.generation.prompts import build_rag_prompt


def build_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.gemini_flash_model,
        google_api_key=settings.google_api_key,
        temperature=0.0,
        max_output_tokens=2048,
    )


def format_context(docs: list[Document]) -> str:
    """Render retrieved docs as a numbered context block with citation metadata."""
    parts = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        ts = m.get("timestamp_seconds", 0)
        parts.append(
            f"[{i}] Title: {m.get('title', 'Unknown')} | "
            f"Channel: {m.get('channel', 'Unknown')} | "
            f"Timestamp: ~{ts}s\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def build_generation_chain():
    """LCEL chain: {docs, question} → grounded answer string."""
    prompt = build_rag_prompt()
    llm = build_llm()
    parser = StrOutputParser()

    chain = (
        {
            "context": RunnableLambda(lambda x: format_context(x["docs"])),
            "question": RunnableLambda(lambda x: x["question"]),
        }
        | prompt
        | llm
        | parser
    )
    return chain
