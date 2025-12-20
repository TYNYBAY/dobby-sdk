# Hybrid Retriever

Combines vector search with keyword search for optimal retrieval.

## Overview

The `HybridRetriever`:
1. Generates query embedding using OpenAI
2. Performs hybrid search (vector + BM25)
3. Returns ranked results using RRF (Reciprocal Rank Fusion)

## Setup

```python
from dobby.vector_store import AzureAISearchVectorStore
from dobby.retriever import HybridRetriever

# Setup vector store
store = AzureAISearchVectorStore(
    endpoint="https://your-search.search.windows.net",
    api_key="search-key",
    index_name="policies",
)

# Setup retriever with OpenAI embeddings
retriever = HybridRetriever(
    vector_store=store,
    openai_api_key="sk-...",
    embedding_model="text-embedding-3-small",
)
```

## Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `vector_store` | `BaseVectorStore` | Underlying store |
| `openai_api_key` | `str` | OpenAI API key |
| `openai_endpoint` | `str` | Azure OpenAI endpoint (optional) |
| `embedding_model` | `str` | Model name (default: `text-embedding-3-small`) |
| `api_version` | `str` | Azure API version |

---

## Azure OpenAI Embeddings

```python
retriever = HybridRetriever(
    vector_store=store,
    openai_api_key="azure-key",
    openai_endpoint="https://your-resource.openai.azure.com",
    embedding_model="text-embedding-3-small",
    api_version="2024-12-01-preview",
)
```

---

## Retrieving Documents

```python
results = await retriever.retrieve(
    query="What does the medical policy cover?",
    top_k=5,
    filters="section_number gt 0",
)

for doc in results:
    print(f"[Score: {doc.score:.3f}] {doc.content[:100]}...")
```

---

## With Tools

Use retriever in a tool for RAG:

```python
from dataclasses import dataclass
from typing import Annotated
from dobby import Tool, Injected

@dataclass
class PolicySearchTool(Tool):
    name = "search_policies"
    description = "Search insurance policy documents"
    
    async def __call__(
        self,
        ctx: Injected[AppContext],
        query: Annotated[str, "Search query"],
    ) -> list[dict]:
        results = await ctx.retriever.retrieve(query, top_k=5)
        return [
            {"content": doc.content, "score": doc.score}
            for doc in results
        ]
```

---

## Cleanup

```python
await retriever.close()
```

This closes both the OpenAI client and any underlying connections.
