import textwrap

from langchain_core.documents import Document

from github_to_qdrant import (
    ConfigLoader,
    create_payload,
    get_metadata_structure,
    is_excluded_path,
)


def test_config_loader_resolves_env_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("EXISTING_VALUE", "from-env")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            value: ${EXISTING_VALUE}
            fallback: ${MISSING_VALUE:-fallback}
            unresolved: ${MISSING_VALUE}
            """
        ),
        encoding="utf-8",
    )

    cfg = ConfigLoader.load_config(str(config_path))

    assert cfg["value"] == "from-env"
    assert cfg["fallback"] == "fallback"
    assert cfg["unresolved"] == "${MISSING_VALUE}"


def test_metadata_structure_uses_payload_first_and_retrieval_fallback():
    assert get_metadata_structure({"payload": {"metadata_structure": "flat"}}) == "flat"
    assert (
        get_metadata_structure({"retrieval": {"metadata_structure": "flat"}}) == "flat"
    )
    assert (
        get_metadata_structure(
            {
                "payload": {"metadata_structure": "nested"},
                "retrieval": {"metadata_structure": "flat"},
            }
        )
        == "nested"
    )


def test_create_payload_respects_nested_and_flat_metadata():
    doc = Document(
        page_content="Hello from docs/auth.md",
        metadata={"file_path": "docs/auth.md", "repository": "demo"},
    )
    base_cfg = {
        "embedding_provider": "sentence_transformers",
        "sentence_transformers": {"model": "test-model"},
        "payload": {"content_fields": ["content"], "metadata_structure": "nested"},
    }

    nested = create_payload(doc, base_cfg, 0, "demo", "docs/auth.md")
    assert nested["content"] == doc.page_content
    assert nested["metadata"]["file_path"] == "docs/auth.md"
    assert nested["metadata"]["source_type"] == "markdown"

    flat_cfg = {
        **base_cfg,
        "payload": {**base_cfg["payload"], "metadata_structure": "flat"},
    }
    flat = create_payload(doc, flat_cfg, 0, "demo", "docs/auth.md")
    assert "metadata" not in flat
    assert flat["file_path"] == "docs/auth.md"


def test_exclude_patterns_support_globs_segments_and_paths(tmp_path):
    root = tmp_path / "repo"
    patterns = ["node_modules", "__pycache__", "*.pyc", "docs/generated/*"]

    assert is_excluded_path(
        str(root / "node_modules/pkg/index.js"), str(root), patterns
    )
    assert is_excluded_path(str(root / "src/__pycache__/mod.pyc"), str(root), patterns)
    assert is_excluded_path(str(root / "src/mod.pyc"), str(root), patterns)
    assert is_excluded_path(str(root / "docs/generated/api.md"), str(root), patterns)
    assert not is_excluded_path(
        str(root / "docs/node_modules-guide.md"), str(root), patterns
    )
