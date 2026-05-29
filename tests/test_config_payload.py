import textwrap
from pathlib import Path

from langchain_core.documents import Document

from github_to_qdrant import (
    ConfigLoader,
    GitHubToQdrantProcessor,
    create_payload,
    get_metadata_structure,
    is_excluded_path,
    is_likely_text_file,
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


def test_likely_text_detection_rejects_binary_files(tmp_path):
    text_file = tmp_path / "Dockerfile"
    text_file.write_text("FROM python:3.12\nRUN echo hello\n", encoding="utf-8")
    binary_file = tmp_path / "asset.bin"
    binary_file.write_bytes(b"\x00\x01\x02PNG\x00")

    assert is_likely_text_file(str(text_file))
    assert not is_likely_text_file(str(binary_file))


def test_all_text_mode_detects_unlisted_text_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Docs\n", encoding="utf-8")
    (repo / "Dockerfile").write_text("FROM python:3.12\n", encoding="utf-8")
    (repo / "asset.bin").write_bytes(b"\x00\x01\x02PNG\x00")
    ignored = repo / "node_modules"
    ignored.mkdir()
    (ignored / "package.json").write_text('{"name": "ignored"}', encoding="utf-8")

    processor = GitHubToQdrantProcessor.__new__(GitHubToQdrantProcessor)
    processor.config = {
        "processing": {
            "file_mode": "all_text",
            "text_extensions": [".md"],
            "markdown_extensions": [".md"],
            "exclude_patterns": ["node_modules"],
        }
    }

    discovered = {
        path.relative_to(repo).as_posix()
        for path in map(Path, processor._find_text_files(str(repo)))
    }

    assert discovered == {"README.md", "Dockerfile"}
