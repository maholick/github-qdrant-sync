import logging
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from qdrant_client.http import models as qdrant_models

from github_to_qdrant import EmbeddingCache, GitHubToQdrantProcessor


class FakeQdrantClient:
    def __init__(self, collection_exists=False):
        self.collection_exists = collection_exists
        self.create_calls = []
        self.update_calls = []
        self.payload_index_calls = []

    def get_collections(self):
        collections = [SimpleNamespace(name="docs")] if self.collection_exists else []
        return SimpleNamespace(collections=collections)

    def create_collection(self, **kwargs):
        self.create_calls.append(kwargs)
        return True

    def update_collection(self, **kwargs):
        self.update_calls.append(kwargs)
        return True

    def get_collection(self, collection_name):
        return SimpleNamespace(payload_schema={})

    def create_payload_index(self, **kwargs):
        self.payload_index_calls.append(kwargs)
        return True


def make_processor(config, client=None):
    processor = GitHubToQdrantProcessor.__new__(GitHubToQdrantProcessor)
    processor.config = config
    processor.qdrant_client = client or FakeQdrantClient()
    processor.logger = logging.getLogger("test")
    processor.embedding_cache = EmbeddingCache()
    return processor


def base_config():
    return {
        "embedding_provider": "sentence_transformers",
        "qdrant": {
            "collection_name": "docs",
            "vector_size": 3,
            "distance": "Cosine",
            "vector_name": "dense",
            "recreate_collection": False,
            "payload_indexes": {"enabled": False},
            "quantization": {"enabled": False},
            "sparse_vector": {"enabled": False},
        },
        "processing": {
            "embedding_batch_size": 2,
            "batch_delay_seconds": 0,
            "deduplication_enabled": False,
        },
        "payload": {"content_fields": ["content"], "metadata_structure": "nested"},
    }


def test_collection_creation_can_enable_turboquant_and_sparse_vectors():
    cfg = base_config()
    cfg["qdrant"]["quantization"] = {
        "enabled": True,
        "method": "turbo",
        "bits": "bits2",
        "always_ram": True,
        "apply_to_existing_collections": False,
    }
    cfg["qdrant"]["sparse_vector"] = {
        "enabled": True,
        "name": "sparse",
        "model": "qdrant/bm25",
    }
    client = FakeQdrantClient(collection_exists=False)
    processor = make_processor(cfg, client)

    processor._setup_qdrant_collection()

    assert len(client.create_calls) == 1
    call = client.create_calls[0]
    assert (
        call["quantization_config"].turbo.bits == qdrant_models.TurboQuantBitSize.BITS2
    )
    assert "sparse" in call["sparse_vectors_config"]
    assert "dense" in call["vectors_config"]


def test_existing_collection_only_gets_quantization_when_explicitly_enabled():
    cfg = base_config()
    cfg["qdrant"]["quantization"] = {
        "enabled": True,
        "method": "turbo",
        "bits": "bits4",
        "always_ram": True,
        "apply_to_existing_collections": False,
    }
    client = FakeQdrantClient(collection_exists=True)
    make_processor(cfg, client)._setup_qdrant_collection()
    assert client.update_calls == []

    cfg["qdrant"]["quantization"]["apply_to_existing_collections"] = True
    client = FakeQdrantClient(collection_exists=True)
    make_processor(cfg, client)._setup_qdrant_collection()
    assert client.update_calls[0]["quantization_config"].turbo.bits


def test_sparse_vectors_require_named_dense_vector():
    cfg = base_config()
    cfg["qdrant"]["vector_name"] = None
    cfg["qdrant"]["sparse_vector"] = {"enabled": True, "name": "sparse"}

    with pytest.raises(ValueError, match="vector_name"):
        make_processor(cfg)._setup_qdrant_collection()


def test_embedding_validation_catches_count_and_dimension_mismatches():
    processor = make_processor(base_config())
    doc = Document(page_content="hello")

    with pytest.raises(ValueError, match="count mismatch"):
        processor._validate_embeddings([doc, doc], [[0.0, 0.1, 0.2]])

    with pytest.raises(ValueError, match="dimension mismatch"):
        processor._validate_embeddings([doc], [[0.0, 0.1]])


def test_sparse_point_vector_includes_dense_and_bm25_document():
    cfg = base_config()
    cfg["qdrant"]["sparse_vector"] = {
        "enabled": True,
        "name": "sparse",
        "model": "qdrant/bm25",
    }
    processor = make_processor(cfg)

    vector = processor._build_point_vector(Document(page_content="hello"), [0, 1, 2])

    assert vector["dense"] == [0, 1, 2]
    assert vector["sparse"].text == "hello"
    assert vector["sparse"].model == "qdrant/bm25"
