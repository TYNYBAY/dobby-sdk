from datetime import datetime
import json
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
)
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SimpleField,
    VectorSearch,
)
from azure.search.documents.models import VectorizedQuery
from pydantic import ValidationError
from pydantic.fields import FieldInfo

from .._logging import logger
from .base import BaseVectorStore, Document

DEFAULT_MAX_BATCH_SIZE = 500

type_mapping = {
    str: SearchFieldDataType.String,
    bool: SearchFieldDataType.Boolean,
    int: SearchFieldDataType.Int32,
    float: SearchFieldDataType.Double,
    datetime: SearchFieldDataType.DateTimeOffset,
}


class AzureAISearchVectorStore[T: Document](BaseVectorStore[T]):
    document_type: type[T]
    endpoint: str
    api_version: str
    vector_search_algorithm: VectorSearch | None

    def __init__(
        self,
        search_service_api_key,
        search_service_endpoint,
        document_type: type[T],
        index_name: str,
        embedding_dimension: int,
        search_service_api_version="2024-07-01",
        vector_search_algorithm: VectorSearch | None = None,
    ):
        self.endpoint = search_service_endpoint
        self.api_version = search_service_api_version
        self.api_key = search_service_api_key
        self.index_name = index_name
        self.embedding_dimension = embedding_dimension

        self.document_type = document_type
        self.vector_search_algorithm = vector_search_algorithm

        self._metadata_fields = self._map_index_fields()

        self.client: SearchClient | None = None
        self.index_client: SearchIndexClient | None = None
        self._index_fields = []

    async def close(self):
        """Close all async clients."""
        if self.client:
            await self.client.close()
        if self.index_client:
            await self.index_client.close()
        logger.info("Closed Azure AI Search clients")

    def _map_index_fields(self) -> dict[str, SimpleField]:
        """Map Document model fields to Azure Search field definitions."""
        index_field_spec: dict[str, SimpleField] = {}

        model_fields: dict[str, FieldInfo] = self.document_type.model_fields
        for field_name, field_info in model_fields.items():
            if field_name in {"id", "content", "meta", "embedding", "score", "sparse_embedding"}:
                # handled separately for now
                continue

            field_type = type_mapping.get(field_info.annotation)
            if not field_type:
                raise ValueError(f"Unsupported field type for key '{field_name}': {field_info}")

            field_config = field_info.json_schema_extra or {}
            if field_config.get("searchable"):
                index_field_spec[field_name] = SearchableField(name=field_name, type=field_type)
            else:
                index_field_spec[field_name] = SimpleField(
                    name=field_name,
                    type=field_type,
                    filterable=field_config.get("filterable", False),
                    sortable=field_config.get("sortable", False),
                )

        return index_field_spec

    async def get_client(self) -> SearchClient:
        if self.client is not None:
            return self.client

        self.credential = AzureKeyCredential(self.api_key)

        try:
            if not self.index_client:
                self.index_client = SearchIndexClient(
                    endpoint=self.endpoint,
                    credential=self.credential,
                )
            # Check if index exists, create if not
            if not await self._index_exists(self.index_name):
                logger.info(
                    f"The index with name: {self.index_name} doesn't exist, new index will be created"
                )
                await self._create_index()

        except (HttpResponseError, ClientAuthenticationError) as e:
            logger.error(e)
            raise

        if self.index_client:
            search_index = await self.index_client.get_index(self.index_name)
            self._index_fields = [field.name for field in search_index.fields]
            self.client = self.index_client.get_search_client(self.index_name)
        else:
            raise ValueError("index_client is not initialized")

        return self.client

    async def _create_index(self) -> None:
        """Create a new Azure AI Search index."""
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration,
            VectorSearchAlgorithmKind,
            VectorSearchProfile,
        )

        default_fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                hidden=False,
                vector_search_dimensions=self.embedding_dimension,
                vector_search_profile_name="default-vector-config",
            ),
            SimpleField(name="meta", type=SearchFieldDataType.String),
        ]

        if self._metadata_fields:
            default_fields.extend(self._metadata_fields.values())

        # Use provided vector search config or create default HNSW
        vector_search = self.vector_search_algorithm
        if vector_search is None:
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="default-hnsw",
                        kind=VectorSearchAlgorithmKind.HNSW,
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="default-vector-config",
                        algorithm_configuration_name="default-hnsw",
                    )
                ],
            )

        index = SearchIndex(
            name=self.index_name,
            fields=default_fields,
            vector_search=vector_search,
        )

        try:
            await self.index_client.create_index(index)
            logger.info(f"Successfully created index: {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to create index {self.index_name}: {e}")
            raise

    async def _index_exists(self, index_name) -> bool:
        if self.index_client:
            index_name_iter = self.index_client.list_index_names()
            async for name in index_name_iter:
                if index_name == name:
                    return True

            return False

        raise ValueError("`index_client` is not defined")

    async def count_documents(self) -> int:
        """Returns how many documents are present in the search index."""
        client = await self.get_client()
        return await client.get_document_count()

    async def delete_document(self, ids: list[str]):
        """Delete documents from the index.

        Args:
            ids: List of document IDs to delete
        """
        try:
            if await self.count_documents() == 0:
                return

            client = await self.get_client()
            documents = [{"id": doc_id} for doc_id in ids]
            await client.delete_documents(documents=documents)
            logger.info(f"Deleted {len(ids)} documents from {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to delete documents from {self.index_name}: {e}")
            raise

    async def delete_index(self, index_name: str) -> None:
        """Delete an entire index.

        Args:
            index_name: Name of the index to delete
        """
        if self.index_client:
            try:
                await self.index_client.delete_index(index_name)
                self.client = None
                logger.info(f"Successfully deleted index: {index_name}")
            except Exception as e:
                logger.error(f"Failed to delete index {index_name}: {e}")
                raise
        else:
            raise ValueError("index_client is not initialized")

    async def list_indexes(self) -> list[str]:
        """List all available indexes.

        Returns:
            List of index names
        """
        if not self.index_client:
            raise ValueError("index_client is not initialized")

        try:
            index_names = []
            async for index in self.index_client.list_indexes():
                index_names.append(index.name)
            logger.info(f"Found {len(index_names)} indexes")
            return index_names
        except Exception as e:
            logger.error(f"Failed to list indexes: {e}")
            raise

    def _document_to_azure(self, document: Document) -> dict[str, Any] | None:
        """Create AI Search index document from embedding result."""
        doc = document.model_dump(exclude_none=True)
        if doc.get("meta"):
            doc["meta"] = json.dumps(doc["meta"])

        if doc.get("embedding") is None:
            logger.warning(f"Cannot index document without embedding, Skipping.. {document.id}")
            return None

        return doc

    async def add_document(self, documents: list[Document]):
        from azure.search.documents import IndexDocumentsBatch

        accumulator = IndexDocumentsBatch()
        ids = []

        for i in range(0, len(documents), DEFAULT_MAX_BATCH_SIZE):
            batch = documents[i : i + DEFAULT_MAX_BATCH_SIZE]
            for document in batch:
                # logger.debug(f"Processing document node: {document.id}")
                ids.append(document.id)

                index_document = self._document_to_azure(document)
                if not index_document:
                    continue

                accumulator.add_upload_actions(index_document)

            logger.info(
                f"Uploading {len(accumulator.actions)} documents"
                f"current progress {len(ids)} of {len(documents)}, "
            )
            await self.client.index_documents(accumulator)
            accumulator.dequeue_actions()

    async def _get_raw_documents_by_id(self, ids: list[str]) -> list[dict]:
        azure_documents = []
        for doc_id in ids:
            try:
                document = await self.client.get_document(doc_id)
                azure_documents.append(document)
            except ResourceNotFoundError:
                logger.warning(f"Document with ID {doc_id} not found.")
        return azure_documents

    def _convert_azure_result_to_document(self, azure_doc: dict[str, Any]) -> T | None:
        score = azure_doc.get("@search.score")
        # to-do need check other internal fields
        meta = azure_doc.get("meta")
        if meta and isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse meta JSON for document {azure_doc.get('id')}")
                meta = {}

        document = {"score": score, "meta": meta}
        for k, v in azure_doc.items():
            if k not in {
                "@search.score",
                "@search.rerankerScore",
                "@search.highlights",
                "@search.captions",
                "meta",  # already handled
            }:
                document[k] = v

        try:
            index_doc = self.document_type.model_validate(document)
        except ValidationError:
            logger.error(f"Failed to Validate Document {azure_doc.get('id')}")
            return None

        return index_doc

    async def get_documents_by_id(self, document_ids: list[str]) -> list[Document]:
        azure_docs = await self._get_raw_documents_by_id(document_ids)
        documents = []
        for azure_doc in azure_docs:
            document = self._convert_azure_result_to_document(azure_doc)
            if document:
                documents.append(document)
        return documents

    async def _process_search_results(self, results) -> list[T]:
        documents = []
        async for result in results:
            document = self._convert_azure_result_to_document(result)
            documents.append(document)

        return documents

    async def search_document(self, search_text: str, top_k: int = 10) -> list[T]:
        """Returns all documents that match the provided search_text.

        Args:
            search_text: the text to search for in the Document list.
            top_k: Maximum number of documents to return.

        Returns:
            A list of Documents that match the given search_text.
        """
        client = await self.get_client()
        results = await client.search(search_text=search_text, top=top_k)

        return await self._process_search_results(results)

    async def embedding_retrieval(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        filters: str | None = None,
        **kwargs: Any,
    ) -> list[T]:
        client = await self.get_client()

        query = VectorizedQuery(
            vector=query_embedding, k_nearest_neighbors=top_k, fields="embedding"
        )
        results = await client.search(vector_queries=[query], filter=filters, **kwargs)
        return await self._process_search_results(results)

    async def bm25_retrieval(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: str | None = None,
        **kwargs: Any,
    ) -> list[T]:
        client = await self.get_client()

        results = await client.search(search_text=query, filter=filters, top=top_k, **kwargs)
        return await self._process_search_results(results)

    async def hybrid_retrieval(
        self,
        query: str,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        filters: str | None = None,
        **kwargs: Any,
    ) -> list[T]:
        client = await self.get_client()

        k_nearest_neighbors = top_k * 3
        vector_query = VectorizedQuery(
            vector=query_embedding, k_nearest_neighbors=k_nearest_neighbors, fields="embedding"
        )
        results = await client.search(
            search_text=query,
            vector_queries=[vector_query],
            filter=filters,
            top=top_k,
            **kwargs,
        )
        return await self._process_search_results(results)
