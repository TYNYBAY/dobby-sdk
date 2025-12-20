# Azure AI Search

Production-ready vector store using Azure AI Search.

## Setup

```python
from dobby.vector_store import AzureAISearchVectorStore, Document

store = AzureAISearchVectorStore(
    endpoint="https://your-search.search.windows.net",
    api_key="your-api-key",
    index_name="your-index",
)
```

## Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `endpoint` | `str` | Azure Search endpoint URL |
| `api_key` | `str` | Admin or query key |
| `index_name` | `str` | Index name |
| `embedding_dimensions` | `int` | Vector dimensions (default: 1536) |

---

## Custom Documents

Extend the base `Document` for custom fields:

```python
from dobby.vector_store import Document

class PolicyDocument(Document):
    section_number: int
    effective_date: str
    policy_type: str
```

---

## Adding Documents

```python
documents = [
    PolicyDocument(
        id="policy_1",
        content="This policy covers medical expenses...",
        meta={"source": "policy_handbook"},
        section_number=3,
        effective_date="2024-01-01",
        policy_type="medical",
    )
]

await store.add_document(documents)
```

---

## Searching

### Hybrid Search (Recommended)

Combines vector similarity with keyword matching:

```python
results = await store.hybrid_retrieval(
    query="medical coverage for surgery",
    query_embedding=embedding,  # From OpenAI embeddings
    top_k=10,
    filters="policy_type eq 'medical'",
)

for doc in results:
    print(f"{doc.content[:100]}... (score: {doc.score:.3f})")
```

### Vector Search

Pure semantic search:

```python
results = await store.embedding_retrieval(
    query_embedding=embedding,
    top_k=10,
)
```

### Keyword Search

Pure BM25 keyword search:

```python
results = await store.bm25_retrieval(
    query="surgery coverage",
    top_k=10,
)
```

---

## Filtering (OData)

Azure AI Search uses OData syntax for filters:

```python
# Exact match
filters="policy_type eq 'medical'"

# Numeric comparison
filters="section_number gt 5"

# Multiple conditions
filters="policy_type eq 'medical' and section_number ge 3"

# Date comparison
filters="effective_date ge 2024-01-01"
```

---

## Cleanup

```python
await store.close()
```
