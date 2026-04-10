from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A robust in-memory vector store for text chunks.
    
    This implementation uses pure Python lists for storage to ensure 
    compatibility and zero-config execution for the Lab.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._store: list[dict[str, Any]] = []
        self._next_index = 0

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Creates a searchable record from a document chunk."""
        embedding = self._embedding_fn(doc.content)
        return {
            "id": doc.id or str(self._next_index),
            "content": doc.content,
            "metadata": doc.metadata or {},
            "embedding": embedding,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Core similarity search logic using dot product."""
        if not records:
            return []
            
        query_vec = self._embedding_fn(query)
        scored_records = []
        for rec in records:
            score = _dot(query_vec, rec["embedding"])
            scored_records.append({**rec, "score": score})
            
        # Sort by similarity score descending
        scored_records.sort(key=lambda x: x["score"], reverse=True)
        return scored_records[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """Embeds and adds multiple documents/chunks to the store."""
        for doc in docs:
            record = self._make_record(doc)
            self._store.append(record)
            self._next_index += 1

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieves top-k most similar chunks for a query."""
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Returns the total number of chunks stored."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """Performs search with pre-filtering based on metadata."""
        if not metadata_filter:
            return self.search(query, top_k)
            
        filtered_records = []
        for rec in self._store:
            match = True
            for key, val in metadata_filter.items():
                if rec["metadata"].get(key) != val:
                    match = False
                    break
            if match:
                filtered_records.append(rec)
                
        return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """Removes a document or all chunks of a document from the store."""
        initial_size = len(self._store)
        self._store = [
            rec for rec in self._store 
            if rec["metadata"].get("doc_id") != doc_id and rec["id"] != doc_id
        ]
        return len(self._store) < initial_size
