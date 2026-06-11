from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pinecone import Pinecone, ServerlessSpec

from yt_rag.config import settings
from yt_rag.embeddings.gemini import build_query_embeddings


def get_pinecone_client() -> Pinecone:
    return Pinecone(api_key=settings.pinecone_api_key)


def ensure_index_exists() -> None:
    """Create the Pinecone hybrid index if it doesn't already exist."""
    pc = get_pinecone_client()
    existing = [idx.name for idx in pc.list_indexes()]
    if settings.pinecone_index_name not in existing:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=3072,         # gemini-embedding-2 output dimension
            metric="dotproduct",    # required for hybrid search
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


class _PineconeRetriever(BaseRetriever):
    """Minimal LangChain retriever adapter around a PineconeVectorStore."""

    store: object  # PineconeVectorStore; typed as object to avoid forward-ref issues
    k: int = 20
    filter: dict | None = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.store.similarity_search(query, k=self.k, filter=self.filter)  # type: ignore[attr-defined]


class PineconeVectorStore:
    """Lightweight Pinecone vectorstore that works on Python 3.13.

    langchain-pinecone caps at Python <3.13; langchain-community 0.4 dropped
    the bundled Pinecone class.  This wrapper uses the pinecone SDK directly
    and exposes add_documents / similarity_search / as_retriever so the rest
    of the codebase is unaffected.
    """

    def __init__(self, index_name: str, embedding, namespace: str) -> None:
        self._pc = get_pinecone_client()
        self._index = self._pc.Index(index_name)
        self._embedding = embedding
        self._namespace = namespace

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def add_documents(self, docs: list[Document], batch_size: int = 100) -> None:
        texts = [d.page_content for d in docs]
        vectors = self._embedding.embed_documents(texts)

        records = []
        for i, (doc, vec) in enumerate(zip(docs, vectors)):
            vid = doc.metadata.get("video_id", "doc")
            chunk_idx = doc.metadata.get("chunk_index", i)
            records.append({
                "id": f"{vid}_{chunk_idx}",
                "values": vec,
                "metadata": {**doc.metadata, "text": doc.page_content},
            })

        for i in range(0, len(records), batch_size):
            self._index.upsert(
                vectors=records[i : i + batch_size],
                namespace=self._namespace,
            )

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
    ) -> list[Document]:
        query_vec = self._embedding.embed_query(query)
        kwargs: dict = {
            "vector": query_vec,
            "top_k": k,
            "namespace": self._namespace,
            "include_metadata": True,
        }
        if filter:
            kwargs["filter"] = filter

        response = self._index.query(**kwargs)
        docs = []
        for match in response.matches:
            meta = dict(match.metadata)
            text = meta.pop("text", "")
            docs.append(Document(page_content=text, metadata=meta))
        return docs

    def as_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: dict | None = None,
    ) -> _PineconeRetriever:
        kw = search_kwargs or {}
        return _PineconeRetriever(
            store=self,
            k=kw.get("k", settings.dense_top_k),
            filter=kw.get("filter"),
        )


def get_vectorstore(namespace: str) -> PineconeVectorStore:
    """Return a PineconeVectorStore scoped to the given namespace.

    Namespace convention: "{user_id}_{playlist_id}"
    """
    embeddings = build_query_embeddings()
    return PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=embeddings,
        namespace=namespace,
    )
