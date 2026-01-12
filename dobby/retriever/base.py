from abc import ABC, abstractmethod
from typing import Any

from ..vector_store.base import BaseVectorStore, Document


class BaseRetriever[T: Document](ABC):
    """Abstract base class for retriever implementations.

    Retrievers wrap vector stores and provide a simple interface for document retrieval.
    They handle embedding generation and query processing.

    Type Parameters:
        T: Document type (must inherit from Document)
    """

    def __init__(self, vector_store: BaseVectorStore[T]):
        """Initialize retriever with a vector store.

        Args:
            vector_store: Vector store instance to retrieve from
        """
        self.vector_store = vector_store

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: str | None = None,
        **kwargs: Any,
    ) -> list[T]:
        """Retrieve documents for a query.

        Args:
            query: Search query text
            top_k: Number of documents to return
            filters: Optional filter expression
            **kwargs: Additional retrieval parameters

        Returns:
            List of retrieved documents with scores
        """
        pass
