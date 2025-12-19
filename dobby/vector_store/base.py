from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Base data class for searchable document Node for vector store.

    Provided the common fields across the document
    """

    id: str
    """Unique identifier for the document"""

    content: str
    """Text of the document"""

    meta: dict[str, Any] | None = None
    """Additional custom metadata for the document. Must be JSON-serializable."""

    embedding: list[float] | None = Field(default=None)
    """dense vector representation of the document."""

    sparse_embedding: None | Any = Field(default=None)  # kept for future reference, not used now
    """sparse vector representation of the document."""

    score: float | None = Field(default=None)
    """Score of the document. Used for ranking, usually assigned by retrievers."""


class BaseVectorStore[T: Document](ABC):
    """Abstract base class for vector store implementations.

    Defines the interface that all vector stores must implement.
    Supports multiple retrieval strategies: vector search, keyword search (BM25), and hybrid.

    Type Parameters:
        T: Document type (must inherit from Document)
    """

    @abstractmethod
    async def get_client(self):
        """Initialize and return the vector store client.

        Returns:
            Client instance for the vector store
        """
        ...

    @abstractmethod
    async def add_document(self, documents: list[T]) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of documents to index
        """
        ...

    @abstractmethod
    async def hybrid_retrieval(
        self,
        query: str,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        filters: str | None = None,
        **kwargs: Any,
    ) -> list[T]:
        """Perform hybrid search (vector + keyword).

        Args:
            query: Search query text
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Optional filter expression
            **kwargs: Additional search parameters

        Returns:
            List of retrieved documents with scores
        """
        ...

    @abstractmethod
    async def embedding_retrieval(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        filters: str | None = None,
        **kwargs: Any,
    ) -> list[T]:
        """Perform vector similarity search.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Optional filter expression
            **kwargs: Additional search parameters

        Returns:
            List of retrieved documents with scores
        """
        ...

    @abstractmethod
    async def bm25_retrieval(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: str | None = None,
        **kwargs: Any,
    ) -> list[T]:
        """Perform BM25 keyword search.

        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Optional filter expression
            **kwargs: Additional search parameters

        Returns:
            List of retrieved documents with scores
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close all connections and clean up resources."""
        ...