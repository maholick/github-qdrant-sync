from types import SimpleNamespace

from qdrant_client.http import models as qdrant_models

from rag_retrieval import _build_filter, _execute_search


class FakeSearchClient:
    def __init__(self):
        self.calls = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(points=[])


def test_build_filter_maps_metadata_paths_and_excludes_markers():
    flt = _build_filter("nested", {"repository": "demo"})

    assert flt.must[0].key == "metadata.repository"
    assert flt.must[0].match.value == "demo"
    assert flt.must_not[0].key == "metadata.record_type"
    assert flt.must_not[0].match.value == "file_marker"


def test_dense_search_routes_named_vector_with_using():
    client = FakeSearchClient()

    _execute_search(
        client=client,
        collection="docs",
        query_text="hello",
        query_vec=[0.1, 0.2],
        qcfg={"vector_name": "dense"},
        retrieval={"mode": "dense"},
        qdrant_filter=None,
        fetch_k=5,
    )

    call = client.calls[0]
    assert call["query"] == [0.1, 0.2]
    assert call["using"] == "dense"
    assert call["limit"] == 5


def test_hybrid_search_builds_dense_sparse_prefetch_and_rrf_fusion():
    client = FakeSearchClient()
    flt = _build_filter("flat", None)

    _execute_search(
        client=client,
        collection="docs",
        query_text="hello",
        query_vec=[0.1, 0.2],
        qcfg={
            "vector_name": "dense",
            "sparse_vector": {
                "enabled": True,
                "name": "sparse",
                "model": "qdrant/bm25",
            },
        },
        retrieval={"mode": "hybrid", "fusion": "rrf"},
        qdrant_filter=flt,
        fetch_k=20,
    )

    call = client.calls[0]
    assert call["query"].fusion == qdrant_models.Fusion.RRF
    assert len(call["prefetch"]) == 2
    assert call["prefetch"][0].using == "dense"
    assert call["prefetch"][0].query == [0.1, 0.2]
    assert call["prefetch"][1].using == "sparse"
    assert call["prefetch"][1].query.text == "hello"
    assert call["prefetch"][1].query.model == "qdrant/bm25"
    assert call["query_filter"] is flt
