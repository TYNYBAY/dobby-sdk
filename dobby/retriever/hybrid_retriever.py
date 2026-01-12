from typing import Any

from openai import AsyncAzureOpenAI, AsyncOpenAI

from .._logging import logger
from ..vector_store.base import BaseVectorStore, Document
from .base import BaseRetriever


class HybridRetriever[T: Document](BaseRetriever[T]):
    """Hybrid retriever using vector + keyword search with OpenAI embeddings.

    Combines semantic search (via embeddings) with keyword search (BM25)
    for improved retrieval accuracy using Azure AI Search's RRF (Reciprocal Rank Fusion).

    Type Parameters:
        T: Document type (must inherit from Document)

    Example:
        >>> from dobby.vector_store import AzureAISearchVectorStore, HybridRetriever
        >>>
        >>> vector_store = AzureAISearchVectorStore(...)
        >>> retriever = HybridRetriever(
        ...     vector_store=vector_store,
        ...     api_key="sk-...",
        ...     embedding_model="text-embedding-3-small",
        ... )
        >>>
        >>> results = await retriever.retrieve("What is covered by medical insurance?", top_k=5)
        >>> for doc in results:
        ...     print(f"{doc.content[:100]}... (score: {doc.score})")
    """

    client: AsyncOpenAI | AsyncAzureOpenAI

    def __init__(
        self,
        vector_store: BaseVectorStore[T],
        openai_api_key: str | None = None,
        openai_endpoint: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        api_version: str = "2024-12-01-preview",
    ):
        super().__init__(vector_store)

        if openai_endpoint and "azure" in openai_endpoint:
            self.client = AsyncAzureOpenAI(
                api_key=openai_api_key, api_version=api_version, azure_endpoint=openai_endpoint
            )
        else:
            self.client = AsyncOpenAI(api_key=openai_api_key)

        self.model = embedding_model

    async def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text using OpenAI.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            Exception: If embedding generation fails
        """
        try:
            response = await self.client.embeddings.create(model=self.model, input=text)
            embedding = response.data[0].embedding
            logger.debug(
                f"Generated embedding for text (length: {len(text)}, dim: {len(embedding)})"
            )
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: str | None = None,
        **kwargs: Any,
    ) -> list[T]:
        """Retrieve documents using hybrid search (vector + keyword).

        Generates query embedding using OpenAI and performs hybrid search
        combining semantic similarity (vector) with keyword matching (BM25).

        Args:
            query: Search query text
            top_k: Number of documents to return (default: 10)
            filters: Optional filter expression (OData syntax)
            **kwargs: Additional parameters passed to vector store

        Returns:
            List of retrieved documents with scores (sorted by relevance)

        Example:
            >>> results = await retriever.retrieve(
            ...     "medical coverage", top_k=5, filters="section_number gt 3"
            ... )
        """
        query_embedding = await self._generate_embedding(query)

        results = await self.vector_store.hybrid_retrieval(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            **kwargs,
        )

        logger.debug(f"Retrieved {len(results)} documents")
        return results

    async def close(self):
        """Close OpenAI client and clean up resources."""
        await self.client.close()
        logger.info("Closed HybridRetriever")
