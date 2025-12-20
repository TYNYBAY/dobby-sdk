# Vector Stores

Vector stores provide document storage and retrieval using embeddings for semantic search.

## Interface

All vector stores implement `BaseVectorStore`:

```python
from abc import ABC, abstractmethod

class BaseVectorStore[T: Document](ABC):
    
    async def add_document(self, documents: list[T]) -> None: ...
    
    async def hybrid_retrieval(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
        filters: str | None = None,
    ) -> list[T]: ...
    
    async def embedding_retrieval(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: str | None = None,
    ) -> list[T]: ...
    
    async def bm25_retrieval(
        self,
        query: str,
        top_k: int = 10,
        filters: str | None = None,
    ) -> list[T]: ...
    
    async def close(self) -> None: ...
```

## Document Model

```python
from dobby.vector_store import Document

class Document(BaseModel):
    id: str                              # Unique ID
    content: str                         # Text content
    meta: dict[str, Any] | None = None   # Custom metadata
    embedding: list[float] | None = None # Dense embedding
    score: float | None = None           # Relevance score
```

## Search Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `hybrid_retrieval` | Vector + keyword (RRF) | Best accuracy |
| `embedding_retrieval` | Pure vector search | Semantic only |
| `bm25_retrieval` | Pure keyword search | Exact matches |

## Implementations

- [Azure AI Search](./azure-ai-search.md) - Production-ready
