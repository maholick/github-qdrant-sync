import asyncio
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path

import yaml
from rich.console import Console
from textual.widgets import DataTable, Input, RichLog, Static
from typer.testing import CliRunner

import github_qdrant_cli
import github_qdrant_quality
import github_qdrant_tui
import github_to_qdrant
import rag_retrieval
from github_to_qdrant import ConfigLoader
from github_qdrant_cli import app, validate_config_data
from rag_retrieval import CollectionTarget, QueryHit, QueryResponse, QueryTimings


runner = CliRunner()


def _recording_console(width=80, force_terminal=True):
    return Console(
        record=True,
        force_terminal=force_terminal,
        color_system=None,
        file=StringIO(),
        _environ={"TERM": "xterm-256color", "COLUMNS": str(width)},
    )


def _write_minimal_config(path):
    path.write_text(
        yaml.safe_dump(
            {
                "github": {
                    "repository_url": "https://github.com/example/project.git",
                    "branch": "main",
                },
                "embedding_provider": "sentence_transformers",
                "sentence_transformers": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "dimensions": 384,
                },
                "qdrant": {
                    "url": "http://localhost:6333",
                    "collection_name": "project-docs",
                    "vector_size": 384,
                    "distance": "Cosine",
                },
                "processing": {
                    "file_mode": "all_text",
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "text_extensions": [".md", ".txt"],
                    "markdown_extensions": [".md"],
                },
                "pdf_processing": {"enabled": False, "mode": "local"},
                "logging": {"level": "INFO", "format": "%(levelname)s: %(message)s"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_answer_config(path, provider="mistral_ai"):
    _write_minimal_config(path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    loaded["embedding_provider"] = provider
    if provider == "mistral_ai":
        loaded["mistral_ai"] = {
            "api_key": "${MISTRAL_API_KEY}",
            "model": "codestral-embed",
        }
        loaded["answering"] = {
            "provider": "mistral_ai",
            "model": "mistral-large-2512",
            "temperature": 0.2,
            "max_context_chars": 12000,
        }
    elif provider == "azure_openai":
        loaded["azure_openai"] = {
            "api_key": "${AZURE_OPENAI_API_KEY}",
            "endpoint": "${AZURE_OPENAI_ENDPOINT}",
            "model": "text-embedding-3-large",
            "api_version": "2024-02-01",
        }
        loaded["answering"] = {
            "provider": "azure_openai",
            "model": "${AZURE_OPENAI_CHAT_DEPLOYMENT}",
            "temperature": 0.2,
            "max_context_chars": 12000,
        }
    path.write_text(yaml.safe_dump(loaded, sort_keys=False), encoding="utf-8")


def _write_repo_list(path):
    path.write_text(
        yaml.safe_dump(
            {
                "repositories": [
                    {
                        "name": "Project One",
                        "url": "https://github.com/example/one.git",
                        "branch": "main",
                        "collection_name": "project-one",
                    },
                    {
                        "name": "Project Two",
                        "url": "https://github.com/example/two.git",
                        "branch": "develop",
                        "collection_name": "project-two",
                    },
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_cli_help_lists_core_commands():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "ingest" in result.output
    assert "ask" in result.output
    assert "collections" in result.output
    assert "interactive" in result.output
    assert "query" in result.output
    assert "wizard" in result.output
    assert "validate-config" in result.output


def test_cli_version_option_prints_version():
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "GithubQdrant-Sync v" in result.output


def test_cli_imports_project_modules_from_its_own_directory_when_cwd_shadows(tmp_path):
    (tmp_path / "github_to_qdrant.py").write_text(
        "class ConfigLoader:\n    pass\n",
        encoding="utf-8",
    )
    module_dir = str(Path(github_qdrant_cli.__file__).resolve().parent)
    script = (
        "import sys; "
        f"sys.path.insert(1, {module_dir!r}); "
        "import github_qdrant_cli; "
        "print(github_qdrant_cli.ConfigLoader.__module__)"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "github_to_qdrant"


def test_brand_header_renders_project_metadata(monkeypatch):
    test_console = _recording_console(width=80)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(github_qdrant_cli, "console", test_console)

    github_qdrant_cli._print_startup_screen("Semantic query")

    output = test_console.export_text(styles=False)
    assert "GithubQdrant-Sync" in output
    assert "GitHub repositories -> Qdrant vector knowledge" in output
    assert "maholick/github-qdrant-sync" in output
    assert "v0.5.0" in output
    assert "Qdrat" not in output
    assert max(len(line) for line in output.splitlines()) <= 80


def test_brand_header_suppressed_when_requested(monkeypatch):
    test_console = _recording_console(width=80)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(github_qdrant_cli, "console", test_console)

    github_qdrant_cli._print_startup_screen("Semantic query", no_banner=True)

    assert test_console.export_text(styles=False) == ""


def test_brand_header_suppressed_for_no_color(monkeypatch):
    test_console = _recording_console(width=80)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(github_qdrant_cli, "console", test_console)

    github_qdrant_cli._print_startup_screen("Semantic query")

    assert test_console.export_text(styles=False) == ""


def test_brand_header_suppressed_for_non_terminal(monkeypatch):
    test_console = _recording_console(width=80, force_terminal=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(github_qdrant_cli, "console", test_console)

    github_qdrant_cli._print_startup_screen("Semantic query")

    assert test_console.export_text(styles=False) == ""


def test_ingest_delegates_to_shared_runner(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_minimal_config(config)
    calls = {}

    def fake_run_ingest(config_path, repo_url=None, repo_list=None):
        calls["config_path"] = config_path
        calls["repo_url"] = repo_url
        calls["repo_list"] = repo_list
        return 0

    monkeypatch.setattr(github_qdrant_cli, "run_ingest", fake_run_ingest)

    result = runner.invoke(
        app,
        [
            "ingest",
            str(config),
            "--repo-url",
            "https://github.com/example/override.git",
        ],
    )

    assert result.exit_code == 0
    assert calls == {
        "config_path": str(config),
        "repo_url": "https://github.com/example/override.git",
        "repo_list": None,
    }


def test_ingest_propagates_runner_exit_code(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_minimal_config(config)

    monkeypatch.setattr(github_qdrant_cli, "run_ingest", lambda **_: 7)

    result = runner.invoke(app, ["ingest", str(config)])

    assert result.exit_code == 7


def test_query_delegates_to_shared_runner(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    repo_list = tmp_path / "repositories.yaml"
    _write_minimal_config(config)
    _write_repo_list(repo_list)
    calls = {}

    def fake_run_query(**kwargs):
        calls.update(kwargs)
        return 0

    monkeypatch.setattr(github_qdrant_cli, "run_query", fake_run_query)

    result = runner.invoke(
        app,
        [
            "query",
            str(config),
            "--query",
            "authentication setup",
            "--limit",
            "3",
            "--format",
            "json",
            "--collection",
            "project-two",
            "--repo-list",
            str(repo_list),
            "--quiet",
        ],
    )

    assert result.exit_code == 0
    assert calls["config_path"] == str(config)
    assert calls["query"] == "authentication setup"
    assert calls["limit"] == 3
    assert calls["output_format"] == "json"
    assert calls["collection"] == "project-two"
    assert calls["repo_list"] == str(repo_list)
    assert calls["quiet"] is True


def test_query_text_format_suppresses_logs_by_default(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_minimal_config(config)
    calls = {}

    def fake_run_query(**kwargs):
        calls.update(kwargs)
        return 0

    monkeypatch.setattr(github_qdrant_cli, "run_query", fake_run_query)

    result = runner.invoke(
        app,
        ["query", str(config), "--query", "authentication setup", "--format", "text"],
    )

    assert result.exit_code == 0
    assert calls["quiet"] is True


def test_text_query_response_uses_matched_snippet_label(capsys):
    response = QueryResponse(
        query="authentication setup",
        collection="project-docs",
        hits=[
            QueryHit(
                score=0.91,
                file_path="docs/auth.md",
                content="Use an API key.",
                metadata={"file_path": "docs/auth.md"},
                preview="Authentication setup guide",
            )
        ],
        timings=QueryTimings(0.1, 0.2, 0.0),
        candidates=4,
    )

    rag_retrieval._render_text_response(response)

    output = capsys.readouterr().out
    assert "matched snippet: Authentication setup guide" in output
    assert "preview:" not in output


class _FakePoint:
    def __init__(self, score, file_path, content, preview):
        self.score = score
        self.payload = {
            "page_content": content,
            "metadata": {
                "file_path": file_path,
                "preview": preview,
                "embedding_provider": "sentence_transformers",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            },
        }


class _FakeQueryResult:
    def __init__(self, points):
        self.points = points


class _FakeEmbedder:
    def embed_query(self, _query):
        return [0.1, 0.2, 0.3]


class _FakeVectorParams:
    size = 384
    distance = "Cosine"


class _FakeInfoParams:
    vectors = _FakeVectorParams()


class _FakeInfoConfig:
    params = _FakeInfoParams()


class _FakeCollectionInfo:
    config = _FakeInfoConfig()
    points_count = 1


class _FakeQdrantClient:
    existing = {"project-docs", "project-one", "project-two"}
    points_by_collection = {}
    queried_collections = []
    query_kwargs = []

    def __init__(self, **_kwargs):
        pass

    def collection_exists(self, collection_name):
        return collection_name in self.existing

    def get_collection(self, collection_name):
        if collection_name not in self.existing:
            raise ValueError("missing")
        return _FakeCollectionInfo()

    def scroll(self, collection_name, **_kwargs):
        points = self.points_by_collection.get(collection_name, [])
        return points[:1], None

    def query_points(self, collection_name, **_kwargs):
        self.queried_collections.append(collection_name)
        self.query_kwargs.append(_kwargs)
        return _FakeQueryResult(self.points_by_collection.get(collection_name, []))


def _patch_fake_retrieval(monkeypatch, points_by_collection, existing=None):
    _FakeQdrantClient.points_by_collection = points_by_collection
    _FakeQdrantClient.existing = (
        existing if existing is not None else set(points_by_collection)
    )
    _FakeQdrantClient.queried_collections = []
    _FakeQdrantClient.query_kwargs = []
    monkeypatch.setattr(rag_retrieval, "QdrantClient", _FakeQdrantClient)
    monkeypatch.setattr(rag_retrieval, "_init_embedder", lambda _cfg: _FakeEmbedder())
    return _FakeQdrantClient


def test_resolve_collection_targets_defaults_to_config(tmp_path):
    config = tmp_path / "config.yaml"
    _write_minimal_config(config)
    loaded = yaml.safe_load(config.read_text(encoding="utf-8"))

    targets = rag_retrieval.resolve_collection_targets(loaded)

    assert targets == [CollectionTarget(collection_name="project-docs")]


def test_resolve_collection_targets_supports_override_and_repo_list(tmp_path):
    config = tmp_path / "config.yaml"
    repo_list = tmp_path / "repositories.yaml"
    _write_minimal_config(config)
    _write_repo_list(repo_list)
    loaded = yaml.safe_load(config.read_text(encoding="utf-8"))

    all_targets = rag_retrieval.resolve_collection_targets(
        loaded, repo_list=str(repo_list)
    )
    narrowed = rag_retrieval.resolve_collection_targets(
        loaded,
        collection="project-two",
        repo_list=str(repo_list),
    )
    override = rag_retrieval.resolve_collection_targets(
        loaded, collection="manual-collection"
    )

    assert [target.collection_name for target in all_targets] == [
        "project-one",
        "project-two",
    ]
    assert narrowed[0].collection_name == "project-two"
    assert narrowed[0].repository_name == "Project Two"
    assert override == [CollectionTarget(collection_name="manual-collection")]


def test_execute_query_searches_repo_list_collections_and_merges_by_score(
    tmp_path, monkeypatch
):
    config = tmp_path / "config.yaml"
    repo_list = tmp_path / "repositories.yaml"
    _write_minimal_config(config)
    _write_repo_list(repo_list)
    fake_client = _patch_fake_retrieval(
        monkeypatch,
        {
            "project-one": [
                _FakePoint(0.82, "one.md", "one content", "one snippet"),
            ],
            "project-two": [
                _FakePoint(0.94, "two.md", "two content", "two snippet"),
            ],
        },
    )

    response = rag_retrieval.execute_query(
        config_path=str(config),
        query="auth",
        limit=2,
        quiet=True,
        repo_list=str(repo_list),
    )

    assert fake_client.queried_collections == ["project-one", "project-two"]
    assert response.collection == "2 collections"
    assert response.collections == ["project-one", "project-two"]
    assert [hit.collection for hit in response.hits] == ["project-two", "project-one"]
    assert [hit.file_path for hit in response.hits] == ["two.md", "one.md"]


def test_execute_query_skips_missing_collections_when_one_exists(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    repo_list = tmp_path / "repositories.yaml"
    _write_minimal_config(config)
    _write_repo_list(repo_list)
    _patch_fake_retrieval(
        monkeypatch,
        {
            "project-one": [
                _FakePoint(0.82, "one.md", "one content", "one snippet"),
            ],
        },
        existing={"project-one"},
    )

    response = rag_retrieval.execute_query(
        config_path=str(config),
        query="auth",
        quiet=True,
        repo_list=str(repo_list),
    )

    assert response.collections == ["project-one"]
    assert response.skipped_collections[0].collection_name == "project-two"
    assert response.skipped_collections[0].status == "missing"
    assert "collection does not exist" in response.warnings[0]
    assert response.hits[0].collection == "project-one"


def test_execute_query_fails_when_all_requested_collections_are_missing(
    tmp_path, monkeypatch
):
    config = tmp_path / "config.yaml"
    repo_list = tmp_path / "repositories.yaml"
    _write_minimal_config(config)
    _write_repo_list(repo_list)
    _patch_fake_retrieval(monkeypatch, {}, existing=set())

    try:
        rag_retrieval.execute_query(
            config_path=str(config),
            query="auth",
            quiet=True,
            repo_list=str(repo_list),
        )
    except LookupError as exc:
        assert "None of the requested collections are usable" in str(exc)
        assert "collection does not exist" in str(exc)
    else:
        raise AssertionError("Expected missing collections to fail")


def test_execute_query_skips_embedding_mismatched_collections(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    repo_list = tmp_path / "repositories.yaml"
    _write_minimal_config(config)
    _write_repo_list(repo_list)
    mismatched = _FakePoint(0.94, "two.md", "two content", "two snippet")
    mismatched.payload["metadata"]["embedding_model"] = "different-model"
    fake_client = _patch_fake_retrieval(
        monkeypatch,
        {
            "project-one": [
                _FakePoint(0.82, "one.md", "one content", "one snippet"),
            ],
            "project-two": [mismatched],
        },
    )

    response = rag_retrieval.execute_query(
        config_path=str(config),
        query="auth",
        quiet=True,
        repo_list=str(repo_list),
    )

    assert fake_client.queried_collections == ["project-one"]
    assert response.collections == ["project-one"]
    assert response.skipped_collections[0].collection_name == "project-two"
    assert response.skipped_collections[0].status == "embedding_mismatch"
    assert "different-model" in response.warnings[0]


def test_execute_query_fails_when_all_collections_are_incompatible(
    tmp_path, monkeypatch
):
    config = tmp_path / "config.yaml"
    repo_list = tmp_path / "repositories.yaml"
    _write_minimal_config(config)
    _write_repo_list(repo_list)
    first = _FakePoint(0.82, "one.md", "one content", "one snippet")
    second = _FakePoint(0.94, "two.md", "two content", "two snippet")
    first.payload["metadata"]["embedding_provider"] = "mistral_ai"
    first.payload["metadata"]["embedding_model"] = "codestral-embed"
    second.payload["metadata"]["embedding_model"] = "different-model"
    _patch_fake_retrieval(
        monkeypatch,
        {"project-one": [first], "project-two": [second]},
    )

    try:
        rag_retrieval.execute_query(
            config_path=str(config),
            query="auth",
            quiet=True,
            repo_list=str(repo_list),
        )
    except LookupError as exc:
        assert "None of the requested collections are usable" in str(exc)
        assert "mistral_ai/codestral-embed" in str(exc)
    else:
        raise AssertionError("Expected incompatible collections to fail")


def test_execute_query_allows_unknown_embedding_metadata_with_warning(
    tmp_path, monkeypatch
):
    config = tmp_path / "config.yaml"
    _write_minimal_config(config)
    point = _FakePoint(0.82, "one.md", "one content", "one snippet")
    point.payload["metadata"].pop("embedding_provider")
    point.payload["metadata"].pop("embedding_model")
    fake_client = _patch_fake_retrieval(monkeypatch, {"project-docs": [point]})

    response = rag_retrieval.execute_query(
        config_path=str(config),
        query="auth",
        quiet=True,
    )

    assert fake_client.queried_collections == ["project-docs"]
    assert response.collection_statuses[0].status == "metadata_unknown"
    assert "metadata is missing" in response.warnings[0]


def test_execute_query_passes_quantization_search_params(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_minimal_config(config)
    loaded = yaml.safe_load(config.read_text(encoding="utf-8"))
    loaded["qdrant"]["quantization"] = {
        "enabled": True,
        "method": "turbo",
        "bits": "bits4",
        "search": {"ignore": False, "rescore": True, "oversampling": 2.0},
    }
    config.write_text(yaml.safe_dump(loaded, sort_keys=False), encoding="utf-8")
    fake_client = _patch_fake_retrieval(
        monkeypatch,
        {
            "project-docs": [
                _FakePoint(0.82, "one.md", "one content", "one snippet"),
            ],
        },
    )

    rag_retrieval.execute_query(
        config_path=str(config),
        query="auth",
        quiet=True,
    )

    search_params = fake_client.query_kwargs[0]["search_params"]
    assert search_params.quantization.ignore is False
    assert search_params.quantization.rescore is True
    assert search_params.quantization.oversampling == 2.0


def test_collection_compatibility_reports_vector_mismatch(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_minimal_config(config)
    loaded = yaml.safe_load(config.read_text(encoding="utf-8"))
    targets = [CollectionTarget(collection_name="project-docs")]

    class VectorParams:
        size = 1536
        distance = "Cosine"

    class Params:
        vectors = VectorParams()

    class InfoConfig:
        params = Params()

    class Info:
        config = InfoConfig()

    class Client:
        def collection_exists(self, collection_name):
            return collection_name == "project-docs"

        def get_collection(self, collection_name):
            return Info()

    statuses = rag_retrieval.inspect_collection_compatibility(loaded, targets, Client())

    assert statuses[0].status == "vector_mismatch"
    assert statuses[0].usable is False


def test_query_json_includes_collection_metadata(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    repo_list = tmp_path / "repositories.yaml"
    _write_minimal_config(config)
    _write_repo_list(repo_list)
    _patch_fake_retrieval(
        monkeypatch,
        {
            "project-one": [
                _FakePoint(0.82, "one.md", "one content", "one snippet"),
            ],
            "project-two": [
                _FakePoint(0.94, "two.md", "two content", "two snippet"),
            ],
        },
    )

    result = runner.invoke(
        app,
        [
            "query",
            str(config),
            "--query",
            "auth",
            "--repo-list",
            str(repo_list),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    parsed = yaml.safe_load(result.output)
    assert parsed["collections"] == ["project-one", "project-two"]
    assert parsed["results"][0]["collection"] == "project-two"
    assert parsed["skipped_collections"] == []


def test_query_json_redacts_secret_metadata(capsys):
    response = QueryResponse(
        query="auth",
        collection="project-docs",
        hits=[
            QueryHit(
                score=0.91,
                file_path="docs/auth.md",
                content="content",
                metadata={
                    "api_key": "super-secret",
                    "url": "https://ghp_token@github.com/example/project.git",
                },
            )
        ],
        timings=QueryTimings(0.1, 0.2, 0.0),
        candidates=1,
    )

    rag_retrieval._render_json_response(response)

    output = capsys.readouterr().out
    parsed = yaml.safe_load(output)
    assert parsed["results"][0]["metadata"]["api_key"] == "<redacted>"
    assert "ghp_token" not in output


def test_execute_query_non_verbose_does_not_log_query_text(
    tmp_path, monkeypatch, capsys
):
    config = tmp_path / "config.yaml"
    _write_minimal_config(config)
    _patch_fake_retrieval(
        monkeypatch,
        {
            "project-docs": [
                _FakePoint(0.82, "one.md", "one content", "one snippet"),
            ],
        },
    )

    rag_retrieval.execute_query(
        config_path=str(config),
        query="super-secret-question",
        quiet=False,
        verbose=False,
    )

    captured = capsys.readouterr()
    assert "super-secret-question" not in captured.err


def test_query_defaults_to_rich_renderer(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_minimal_config(config)

    def fake_execute_query(**_kwargs):
        return QueryResponse(
            query="authentication setup",
            collection="project-docs",
            hits=[
                QueryHit(
                    score=0.91,
                    file_path="docs/auth.md",
                    content="Use an API key.",
                    metadata={"file_path": "docs/auth.md"},
                    preview="Authentication setup guide",
                )
            ],
            timings=QueryTimings(0.1, 0.2, 0.0),
            candidates=4,
        )

    monkeypatch.setattr(github_qdrant_cli, "execute_query", fake_execute_query)

    result = runner.invoke(
        app,
        ["query", str(config), "--query", "authentication setup", "--quiet"],
    )

    assert result.exit_code == 0
    assert "Vector Search" in result.output
    assert "Matched snippet" in result.output
    assert "docs/auth.md" in result.output
    assert "Authentication setup guide" in result.output
    assert "GithubQdrant-Sync" not in result.output


def test_rich_query_renderer_keeps_lines_within_terminal_width(monkeypatch):
    test_console = _recording_console(width=60)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(github_qdrant_cli, "console", test_console)
    response = QueryResponse(
        query="How do I configure authentication for a private repository? " * 4,
        collection="project-docs",
        hits=[
            QueryHit(
                score=0.91,
                file_path="docs/security/authentication/private-repositories.md",
                content="Use a repository token and store secrets in environment variables.",
                metadata={"file_path": "docs/auth.md"},
                preview="Use a repository token and store secrets in environment variables.",
            )
        ],
        timings=QueryTimings(0.1, 0.2, 0.0),
        candidates=12,
    )

    github_qdrant_cli._render_rich_query_response(response)

    output = test_console.export_text(styles=False)
    assert "Matched snippet" in output
    assert max(len(line) for line in output.splitlines()) <= 60


def test_generate_answer_emits_progress_in_order(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)
    progress = []

    def fake_execute_query(**kwargs):
        kwargs["progress"]("Encoding query")
        kwargs["progress"]("Searching Qdrant")
        kwargs["progress"]("Ranking matches")
        return QueryResponse(
            query=kwargs["query"],
            collection="project-docs",
            hits=[
                QueryHit(
                    score=0.91,
                    file_path="docs/auth.md",
                    content="Use an API key.",
                    metadata={"file_path": "docs/auth.md"},
                    preview="Authentication setup guide",
                )
            ],
            timings=QueryTimings(0.1, 0.2, 0.0),
            candidates=4,
        )

    monkeypatch.setattr(github_qdrant_cli, "execute_query", fake_execute_query)
    monkeypatch.setattr(
        github_qdrant_cli,
        "_call_answer_model",
        lambda *_args, **_kwargs: "Use the configured API key [1].",
    )

    response = github_qdrant_cli.generate_answer(
        config_path=str(config),
        question="How do I configure authentication?",
        progress=progress.append,
    )

    assert response.answer == "Use the configured API key [1]."
    assert progress == [
        "Encoding question",
        "Searching Qdrant",
        "Ranking matches",
        "Preparing context",
        "Generating answer with Mistral",
    ]


def test_ask_command_renders_rich_answer(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)

    monkeypatch.setattr(
        github_qdrant_cli,
        "execute_query",
        lambda **kwargs: QueryResponse(
            query=kwargs["query"],
            collection="project-docs",
            hits=[
                QueryHit(
                    score=0.91,
                    file_path="docs/auth.md",
                    content="Use an API key.",
                    metadata={"file_path": "docs/auth.md"},
                    preview="Authentication setup guide",
                )
            ],
            timings=QueryTimings(0.1, 0.2, 0.0),
            candidates=4,
        ),
    )
    monkeypatch.setattr(
        github_qdrant_cli,
        "_call_answer_model",
        lambda *_args, **_kwargs: "Use the configured API key [1].",
    )

    result = runner.invoke(
        app,
        [
            "ask",
            str(config),
            "--question",
            "How do I configure authentication?",
            "--quiet",
        ],
    )

    assert result.exit_code == 0
    assert "AI Answer" in result.output
    assert "Use the configured API key [1]." in result.output
    assert "docs/auth.md" in result.output


def test_ask_command_reports_missing_answering_config(tmp_path):
    config = tmp_path / "config.yaml"
    _write_minimal_config(config)

    result = runner.invoke(
        app,
        ["ask", str(config), "--question", "How?", "--quiet"],
    )

    assert result.exit_code == 1
    assert "Missing required 'answering' config" in result.output


def test_ask_command_handles_empty_retrieval(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)
    called = {"answer_model": False}

    monkeypatch.setattr(
        github_qdrant_cli,
        "execute_query",
        lambda **kwargs: QueryResponse(
            query=kwargs["query"],
            collection="project-docs",
            hits=[],
            timings=QueryTimings(0.1, 0.2, 0.0),
            candidates=0,
        ),
    )

    def fake_answer_model(*_args, **_kwargs):
        called["answer_model"] = True
        return "should not be called"

    monkeypatch.setattr(github_qdrant_cli, "_call_answer_model", fake_answer_model)

    result = runner.invoke(
        app,
        ["ask", str(config), "--question", "Unknown?", "--quiet"],
    )

    assert result.exit_code == 0
    assert "could not find matching repository context" in result.output
    assert called["answer_model"] is False


def test_ask_command_supports_text_and_json_formats(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)

    monkeypatch.setattr(
        github_qdrant_cli,
        "execute_query",
        lambda **kwargs: QueryResponse(
            query=kwargs["query"],
            collection="project-docs",
            hits=[
                QueryHit(
                    score=0.91,
                    file_path="docs/auth.md",
                    content="Use an API key.",
                    metadata={"file_path": "docs/auth.md"},
                    preview="Authentication setup guide",
                )
            ],
            timings=QueryTimings(0.1, 0.2, 0.0),
            candidates=4,
        ),
    )
    monkeypatch.setattr(
        github_qdrant_cli,
        "_call_answer_model",
        lambda *_args, **_kwargs: "Use the configured API key [1].",
    )

    text_result = runner.invoke(
        app,
        [
            "ask",
            str(config),
            "--question",
            "How?",
            "--format",
            "text",
            "--hide-sources",
        ],
    )
    json_result = runner.invoke(
        app,
        ["ask", str(config), "--question", "How?", "--format", "json"],
    )

    assert text_result.exit_code == 0
    assert "Use the configured API key [1]." in text_result.output
    assert "Sources:" not in text_result.output
    assert json_result.exit_code == 0
    parsed = yaml.safe_load(json_result.output)
    assert parsed["answer"] == "Use the configured API key [1]."
    assert parsed["sources"][0]["file_path"] == "docs/auth.md"


def test_generate_answer_context_includes_collection_names(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)
    captured = {}

    monkeypatch.setattr(
        github_qdrant_cli,
        "execute_query",
        lambda **kwargs: QueryResponse(
            query=kwargs["query"],
            collection="2 collections",
            collections=["project-one", "project-two"],
            hits=[
                QueryHit(
                    score=0.94,
                    file_path="docs/auth.md",
                    content="Use an API key.",
                    metadata={"file_path": "docs/auth.md"},
                    preview="Authentication setup guide",
                    collection="project-two",
                )
            ],
            timings=QueryTimings(0.1, 0.2, 0.0),
            candidates=4,
        ),
    )

    def fake_answer_model(_config, _question, context):
        captured["context"] = context
        return "Use the configured API key [1]."

    monkeypatch.setattr(github_qdrant_cli, "_call_answer_model", fake_answer_model)

    response = github_qdrant_cli.generate_answer(
        config_path=str(config),
        question="How?",
    )

    assert response.answer == "Use the configured API key [1]."
    assert "Collection: project-two" in captured["context"]


def test_ask_text_format_suppresses_logs_by_default(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)
    calls = {}

    def fake_run_ask(**kwargs):
        calls.update(kwargs)
        return 0

    monkeypatch.setattr(github_qdrant_cli, "run_ask", fake_run_ask)

    result = runner.invoke(
        app,
        ["ask", str(config), "--question", "How?", "--format", "text"],
    )

    assert result.exit_code == 0
    assert calls["quiet"] is True


def test_ask_command_passes_collection_and_repo_list_to_runner(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    repo_list = tmp_path / "repositories.yaml"
    _write_answer_config(config)
    _write_repo_list(repo_list)
    calls = {}

    def fake_run_ask(**kwargs):
        calls.update(kwargs)
        return 0

    monkeypatch.setattr(github_qdrant_cli, "run_ask", fake_run_ask)

    result = runner.invoke(
        app,
        [
            "ask",
            str(config),
            "--question",
            "How?",
            "--format",
            "text",
            "--collection",
            "project-two",
            "--repo-list",
            str(repo_list),
        ],
    )

    assert result.exit_code == 0
    assert calls["collection"] == "project-two"
    assert calls["repo_list"] == str(repo_list)


def test_interactive_menu_can_ask_twice(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)
    questions = []

    def fake_generate_answer(**kwargs):
        questions.append(kwargs["question"])
        return github_qdrant_cli.AnswerResponse(
            question=kwargs["question"],
            answer=f"answer for {kwargs['question']}",
            retrieval=QueryResponse(
                query=kwargs["question"],
                collection="project-docs",
                hits=[],
                timings=QueryTimings(0.1, 0.2, 0.0),
                candidates=0,
            ),
            model="mistral-large-2512",
            timings=github_qdrant_cli.AnswerTimings(0.1, 0.2),
            context_chars=0,
        )

    monkeypatch.setattr(github_qdrant_cli, "generate_answer", fake_generate_answer)

    result = runner.invoke(
        app,
        ["interactive", str(config), "--classic", "--no-banner"],
        input="1\nfirst question\n1\nsecond question\n9\n",
    )

    assert result.exit_code == 0
    assert questions == ["first question", "second question"]
    assert "Goodbye" in result.output


def test_interactive_without_config_uses_default_config(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_execute_query(**kwargs):
        calls.append(kwargs)
        time.sleep(0.05)
        return QueryResponse(
            query=kwargs["query"],
            collection="project-docs",
            hits=[],
            timings=QueryTimings(0.1, 0.2, 0.0),
            candidates=0,
        )

    monkeypatch.setattr(github_qdrant_cli, "execute_query", fake_execute_query)

    result = runner.invoke(
        app,
        ["interactive", "--classic", "--no-banner"],
        input="2\nsetup guide\n9\n",
    )

    assert result.exit_code == 0
    assert calls[0]["config_path"] == "config.yaml"
    assert calls[0]["query"] == "setup guide"


def test_no_args_opens_interactive_with_default_config(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, [], input="9\n")

    assert result.exit_code == 0
    assert "Interactive Menu" in result.output
    assert "Goodbye" in result.output


def test_no_args_routes_to_textual_tui_on_terminal(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)
    monkeypatch.chdir(tmp_path)
    calls = {}

    def fake_run_textual_interactive(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(github_qdrant_cli, "_can_run_tui", lambda: True)
    monkeypatch.setattr(
        github_qdrant_cli,
        "_run_textual_interactive",
        fake_run_textual_interactive,
    )

    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert calls["config"] == Path("config.yaml")
    assert calls["collection"] is None
    assert calls["repo_list"] is None
    assert calls["first_run_setup"] is False


def test_no_args_without_config_starts_textual_setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = {}

    def fake_run_textual_interactive(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(github_qdrant_cli, "_can_run_tui", lambda: True)
    monkeypatch.setattr(
        github_qdrant_cli,
        "_run_textual_interactive",
        fake_run_textual_interactive,
    )

    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert calls["config"] == Path("config.yaml")
    assert calls["first_run_setup"] is True


def test_interactive_missing_config_starts_textual_setup(tmp_path, monkeypatch):
    missing = tmp_path / "custom.yaml"
    calls = {}

    def fake_run_textual_interactive(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(github_qdrant_cli, "_can_run_tui", lambda: True)
    monkeypatch.setattr(
        github_qdrant_cli,
        "_run_textual_interactive",
        fake_run_textual_interactive,
    )

    result = runner.invoke(app, ["interactive", str(missing)])

    assert result.exit_code == 0
    assert calls["config"] == missing
    assert calls["first_run_setup"] is True


def test_interactive_defaults_to_textual_tui_on_terminal(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)
    calls = {}

    def fake_run_textual_interactive(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(github_qdrant_cli, "_can_run_tui", lambda: True)
    monkeypatch.setattr(
        github_qdrant_cli,
        "_run_textual_interactive",
        fake_run_textual_interactive,
    )

    result = runner.invoke(app, ["interactive", str(config)])

    assert result.exit_code == 0
    assert calls["config"] == config
    assert calls["with_parent_window"] is False


def test_interactive_no_tui_alias_routes_to_classic(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)
    calls = {}

    def fake_run_interactive_session(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(github_qdrant_cli, "_can_run_tui", lambda: True)
    monkeypatch.setattr(
        github_qdrant_cli,
        "_run_interactive_session",
        fake_run_interactive_session,
    )

    result = runner.invoke(app, ["interactive", str(config), "--no-tui"])

    assert result.exit_code == 0
    assert calls["config"] == config


def test_interactive_falls_back_to_classic_when_tui_is_unavailable(
    tmp_path, monkeypatch
):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)
    calls = {}

    def fake_run_interactive_session(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(github_qdrant_cli, "_can_run_tui", lambda: False)
    monkeypatch.setattr(
        github_qdrant_cli,
        "_run_interactive_session",
        fake_run_interactive_session,
    )

    result = runner.invoke(app, ["interactive", str(config)])

    assert result.exit_code == 0
    assert calls["config"] == config


def test_interactive_menu_can_query_then_exit(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)
    calls = []

    def fake_execute_query(**kwargs):
        calls.append(kwargs)
        time.sleep(0.05)
        return QueryResponse(
            query=kwargs["query"],
            collection="project-docs",
            hits=[],
            timings=QueryTimings(0.1, 0.2, 0.0),
            candidates=0,
        )

    monkeypatch.setattr(github_qdrant_cli, "execute_query", fake_execute_query)

    result = runner.invoke(
        app,
        ["interactive", str(config), "--classic", "--no-banner"],
        input="2\nsetup guide\n9\n",
    )

    assert result.exit_code == 0
    assert calls[0]["query"] == "setup guide"


def test_interactive_menu_updates_limit_and_parent_window(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)
    calls = []

    def fake_execute_query(**kwargs):
        calls.append(kwargs)
        return QueryResponse(
            query=kwargs["query"],
            collection="project-docs",
            hits=[],
            timings=QueryTimings(0.1, 0.2, 0.0),
            candidates=0,
        )

    monkeypatch.setattr(github_qdrant_cli, "execute_query", fake_execute_query)

    result = runner.invoke(
        app,
        ["interactive", str(config), "--classic", "--no-banner"],
        input="6\n7\n3\n2\nsetup guide\n9\n",
    )

    assert result.exit_code == 0
    assert calls[0]["with_parent_window"] is True
    assert calls[0]["limit"] == 3


def test_interactive_starts_with_all_repo_list_collections(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    repo_list = tmp_path / "repositories.yaml"
    _write_answer_config(config)
    _write_repo_list(repo_list)
    calls = []

    def fake_execute_query(**kwargs):
        calls.append(kwargs)
        return QueryResponse(
            query=kwargs["query"],
            collection="2 collections",
            collections=["project-one", "project-two"],
            hits=[],
            timings=QueryTimings(0.1, 0.2, 0.0),
            candidates=0,
        )

    monkeypatch.setattr(github_qdrant_cli, "execute_query", fake_execute_query)

    result = runner.invoke(
        app,
        [
            "interactive",
            str(config),
            "--repo-list",
            str(repo_list),
            "--classic",
            "--no-banner",
        ],
        input="2\nsetup guide\n9\n",
    )

    assert result.exit_code == 0
    assert calls[0]["repo_list"] == str(repo_list)
    assert calls[0]["collection"] is None
    assert "all repo-list collections" in result.output


def test_interactive_can_switch_to_one_repo_list_collection(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    repo_list = tmp_path / "repositories.yaml"
    _write_answer_config(config)
    _write_repo_list(repo_list)
    calls = []

    def fake_execute_query(**kwargs):
        calls.append(kwargs)
        return QueryResponse(
            query=kwargs["query"],
            collection="project-two",
            collections=["project-two"],
            hits=[],
            timings=QueryTimings(0.1, 0.2, 0.0),
            candidates=0,
        )

    monkeypatch.setattr(github_qdrant_cli, "execute_query", fake_execute_query)

    result = runner.invoke(
        app,
        [
            "interactive",
            str(config),
            "--repo-list",
            str(repo_list),
            "--classic",
            "--no-banner",
        ],
        input="5\nproject-two\n2\nsetup guide\n9\n",
    )

    assert result.exit_code == 0
    assert calls[0]["collection"] == "project-two"
    assert calls[0]["repo_list"] == str(repo_list)


def test_interactive_can_list_repo_list_collections(tmp_path):
    config = tmp_path / "config.yaml"
    repo_list = tmp_path / "repositories.yaml"
    _write_answer_config(config)
    _write_repo_list(repo_list)

    result = runner.invoke(
        app,
        [
            "interactive",
            str(config),
            "--repo-list",
            str(repo_list),
            "--classic",
            "--no-banner",
        ],
        input="4\n9\n",
    )

    assert result.exit_code == 0
    assert "project-one" in result.output
    assert "project-two" in result.output


def test_interactive_menu_can_ingest_default_config(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)
    calls = []

    def fake_run_ingest(config_path, repo_url=None, repo_list=None):
        calls.append(
            {
                "config_path": config_path,
                "repo_url": repo_url,
                "repo_list": repo_list,
            }
        )
        return 0

    monkeypatch.setattr(github_qdrant_cli, "run_ingest", fake_run_ingest)

    result = runner.invoke(
        app,
        ["interactive", str(config), "--classic", "--no-banner"],
        input="3\nconfig\n9\n",
    )

    assert result.exit_code == 0
    assert calls == [{"config_path": str(config), "repo_url": None, "repo_list": None}]
    assert "Ingestion completed successfully" in result.output


def test_interactive_menu_can_ingest_repo_url_override(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)
    calls = []

    def fake_run_ingest(config_path, repo_url=None, repo_list=None):
        calls.append(
            {
                "config_path": config_path,
                "repo_url": repo_url,
                "repo_list": repo_list,
            }
        )
        return 0

    monkeypatch.setattr(github_qdrant_cli, "run_ingest", fake_run_ingest)

    result = runner.invoke(
        app,
        ["interactive", str(config), "--classic", "--no-banner"],
        input="3\nrepo-url\nhttps://github.com/example/override.git\n9\n",
    )

    assert result.exit_code == 0
    assert calls == [
        {
            "config_path": str(config),
            "repo_url": "https://github.com/example/override.git",
            "repo_list": None,
        }
    ]


def test_interactive_menu_can_ingest_repo_list(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    repo_list = tmp_path / "repositories.yaml"
    repo_list.write_text("repositories: []\n", encoding="utf-8")
    _write_answer_config(config)
    calls = []

    def fake_run_ingest(config_path, repo_url=None, repo_list=None):
        calls.append(
            {
                "config_path": config_path,
                "repo_url": repo_url,
                "repo_list": repo_list,
            }
        )
        return 0

    monkeypatch.setattr(github_qdrant_cli, "run_ingest", fake_run_ingest)

    result = runner.invoke(
        app,
        ["interactive", str(config), "--classic", "--no-banner"],
        input=f"3\nrepo-list\n{repo_list}\n9\n",
    )

    assert result.exit_code == 0
    assert calls == [
        {"config_path": str(config), "repo_url": None, "repo_list": str(repo_list)}
    ]


def test_interactive_menu_reports_ingest_failure(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)

    monkeypatch.setattr(github_qdrant_cli, "run_ingest", lambda **_: 2)

    result = runner.invoke(
        app,
        ["interactive", str(config), "--classic", "--no-banner"],
        input="3\nconfig\n9\n",
    )

    assert result.exit_code == 0
    assert "Ingestion failed with exit code 2" in result.output


def test_tui_command_parser_maps_supported_commands():
    assert github_qdrant_tui.parse_tui_command(
        "/ask how?"
    ) == github_qdrant_tui.TuiCommand("ask", "how?")
    assert github_qdrant_tui.parse_tui_command("search auth") == (
        github_qdrant_tui.TuiCommand("search", "auth")
    )
    assert github_qdrant_tui.parse_tui_command("scope all") == (
        github_qdrant_tui.TuiCommand("scope", "all")
    )
    assert github_qdrant_tui.parse_tui_command("limit 5") == (
        github_qdrant_tui.TuiCommand("limit", "5")
    )
    assert github_qdrant_tui.parse_tui_command("parent on") == (
        github_qdrant_tui.TuiCommand("parent", "on")
    )
    assert github_qdrant_tui.parse_tui_command("/load-config config.local.yaml") == (
        github_qdrant_tui.TuiCommand("config", "config.local.yaml")
    )
    assert github_qdrant_tui.parse_tui_command("/wizard") == (
        github_qdrant_tui.TuiCommand("wizard")
    )
    assert github_qdrant_tui.parse_tui_command("/repo-list repositories.yaml") == (
        github_qdrant_tui.TuiCommand("repo-list", "repositories.yaml")
    )
    assert github_qdrant_tui.parse_tui_command("/get qdrant.collection_name") == (
        github_qdrant_tui.TuiCommand("get", "qdrant.collection_name")
    )
    assert github_qdrant_tui.parse_tui_command("/set retrieval.top_k 5") == (
        github_qdrant_tui.TuiCommand("set", "retrieval.top_k 5")
    )
    assert github_qdrant_tui.parse_tui_command("/secret qdrant.api_key QDRANT_KEY") == (
        github_qdrant_tui.TuiCommand("secret", "qdrant.api_key QDRANT_KEY")
    )
    assert github_qdrant_tui.parse_tui_command("/changes") == (
        github_qdrant_tui.TuiCommand("changes")
    )
    assert github_qdrant_tui.parse_tui_command("/save-config --confirm") == (
        github_qdrant_tui.TuiCommand("save-config", "--confirm")
    )
    assert github_qdrant_tui.parse_tui_command("/save-config-as local.yaml") == (
        github_qdrant_tui.TuiCommand("save-config-as", "local.yaml")
    )
    assert github_qdrant_tui.parse_tui_command("/discard-config-changes") == (
        github_qdrant_tui.TuiCommand("discard-config-changes")
    )
    assert github_qdrant_tui.parse_tui_command("/doctor") == (
        github_qdrant_tui.TuiCommand("doctor")
    )
    assert github_qdrant_tui.parse_tui_command("/benchmark eval.yaml") == (
        github_qdrant_tui.TuiCommand("benchmark", "eval.yaml")
    )
    assert github_qdrant_tui.parse_tui_command("/improve eval.yaml --apply") == (
        github_qdrant_tui.TuiCommand("improve", "eval.yaml --apply")
    )
    assert github_qdrant_tui.parse_tui_command("/help config") == (
        github_qdrant_tui.TuiCommand("help", "config")
    )
    assert github_qdrant_tui.parse_tui_command("validate") == (
        github_qdrant_tui.TuiCommand("validate")
    )
    assert github_qdrant_tui.parse_tui_command("quit") == (
        github_qdrant_tui.TuiCommand("quit")
    )


def test_textual_app_mounts_fixed_panes(tmp_path):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)

    async def run_app():
        tui = github_qdrant_tui.GithubQdrantSyncApp(config=config)
        async with tui.run_test(size=(100, 30)):
            assert tui.query_one("#status", Static)
            assert tui.query_one("#main", RichLog)
            assert tui.query_one("#activity", RichLog)
            assert tui.query_one("#sources", DataTable)
            assert tui.query_one("#collections", DataTable)
            assert tui.query_one("#command", Input)

    asyncio.run(run_app())


def test_textual_first_run_setup_writes_config(tmp_path):
    config = tmp_path / "config.yaml"

    async def run_app():
        tui = github_qdrant_tui.GithubQdrantSyncApp(config=config, first_run_setup=True)
        async with tui.run_test(size=(100, 30)):
            for answer in [
                "https://github.com/example/project.git",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "disabled",
                "",
            ]:
                tui.handle_command(answer)

    asyncio.run(run_app())

    generated = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert generated["github"]["repository_url"] == (
        "https://github.com/example/project.git"
    )
    assert generated["embedding_provider"] == "mistral_ai"
    assert generated["qdrant"]["collection_name"] == "project"
    assert generated["pdf_processing"]["enabled"] is False


def test_textual_slash_input_opens_command_palette(tmp_path):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)
    filters = []

    async def run_app():
        tui = github_qdrant_tui.GithubQdrantSyncApp(config=config)
        async with tui.run_test(size=(100, 30)) as pilot:
            tui._render_command_palette = lambda filter_text="": filters.append(
                filter_text
            )
            command = tui.query_one("#command", Input)
            command.value = "/a"
            await pilot.pause()

    asyncio.run(run_app())

    assert filters == ["a"]


def test_textual_grouped_palette_has_categories(tmp_path):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)

    async def run_app():
        tui = github_qdrant_tui.GithubQdrantSyncApp(config=config)
        async with tui.run_test(size=(100, 30)):
            all_specs = tui._matching_command_specs("")
            quality_specs = tui._matching_command_specs("quality")
            benchmark_specs = tui._matching_command_specs("ben")
            assert {spec.category for spec in all_specs} >= {
                "Search & Ask",
                "Collections",
                "Config",
                "Quality",
                "Ingest",
                "Session",
            }
            assert any(spec.command.startswith("/doctor") for spec in quality_specs)
            assert [spec.command for spec in benchmark_specs] == [
                "/benchmark [eval.yaml]"
            ]
            tui.handle_command("/help config")

    asyncio.run(run_app())


def test_tui_config_value_parsing_and_secret_guards():
    assert github_qdrant_tui._coerce_config_value("retrieval.top_k", "7") == 7
    assert github_qdrant_tui._coerce_config_value("answering.temperature", "0.4") == 0.4
    assert (
        github_qdrant_tui._coerce_config_value("pdf_processing.enabled", "on") is True
    )
    assert github_qdrant_tui._coerce_config_value("qdrant.vector_name", "null") is None
    assert (
        github_qdrant_tui._secret_placeholder("MISTRAL_API_KEY") == "${MISTRAL_API_KEY}"
    )

    try:
        github_qdrant_tui._coerce_config_value("mistral_ai.api_key", "raw-key")
    except ValueError as exc:
        assert "Use `/secret" in str(exc)
    else:
        raise AssertionError("Expected raw secret values to be rejected")

    try:
        github_qdrant_tui._coerce_config_value("unknown.path", "value")
    except ValueError as exc:
        assert "not editable" in str(exc)
    else:
        raise AssertionError("Expected unknown config paths to be rejected")


def test_textual_search_command_updates_results(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)
    calls = []

    def fake_execute_query(**kwargs):
        calls.append(kwargs)
        return QueryResponse(
            query=kwargs["query"],
            collection="project-docs",
            collections=["project-docs"],
            hits=[
                QueryHit(
                    score=0.91,
                    file_path="README.md",
                    content="authentication setup",
                    metadata={},
                    preview="authentication setup",
                    collection="project-docs",
                )
            ],
            timings=QueryTimings(0.1, 0.2, 0.0),
            candidates=1,
        )

    monkeypatch.setattr(github_qdrant_tui, "execute_query", fake_execute_query)

    async def run_app():
        tui = github_qdrant_tui.GithubQdrantSyncApp(config=config)
        async with tui.run_test(size=(110, 32)) as pilot:
            tui.handle_command("/search auth")
            assert tui.operation_running is True
            await tui.active_worker.wait()
            await pilot.pause()
            sources = tui.query_one("#sources", DataTable)
            assert sources.row_count == 1
            assert tui.operation_running is False

    asyncio.run(run_app())

    assert calls[0]["query"] == "auth"
    assert calls[0]["collection"] is None


def test_textual_search_updates_collection_compatibility_sidebar(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    repo_list = tmp_path / "repositories.yaml"
    _write_answer_config(config)
    _write_repo_list(repo_list)

    def fake_execute_query(**kwargs):
        skipped = rag_retrieval.CollectionCompatibility(
            target=CollectionTarget(
                collection_name="project-two", repository_name="Project Two"
            ),
            status="embedding_mismatch",
            reason="embedded with other-model.",
            exists=True,
            usable=False,
        )
        usable = rag_retrieval.CollectionCompatibility(
            target=CollectionTarget(
                collection_name="project-one", repository_name="Project One"
            ),
            status="usable",
            reason="collection is compatible.",
            exists=True,
            usable=True,
        )
        return QueryResponse(
            query=kwargs["query"],
            collection="project-one",
            collections=["project-one"],
            hits=[
                QueryHit(
                    score=0.91,
                    file_path="README.md",
                    content="authentication setup",
                    metadata={},
                    preview="authentication setup",
                    collection="project-one",
                )
            ],
            timings=QueryTimings(0.1, 0.2, 0.0),
            candidates=1,
            warnings=[skipped.warning_message()],
            skipped_collections=[skipped],
            collection_statuses=[usable, skipped],
        )

    monkeypatch.setattr(github_qdrant_tui, "execute_query", fake_execute_query)

    async def run_app():
        tui = github_qdrant_tui.GithubQdrantSyncApp(config=config, repo_list=repo_list)
        async with tui.run_test(size=(110, 32)) as pilot:
            tui.handle_command("/search auth")
            await tui.active_worker.wait()
            await pilot.pause()
            assert [status.status for status in tui.collection_statuses] == [
                "usable",
                "embedding_mismatch",
            ]
            collections = tui.query_one("#collections", DataTable)
            assert collections.row_count == 2

    asyncio.run(run_app())


def test_textual_ask_command_updates_answer_and_sources(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)
    calls = []

    def fake_generate_answer(**kwargs):
        calls.append(kwargs)
        time.sleep(0.05)
        retrieval = QueryResponse(
            query=kwargs["question"],
            collection="project-docs",
            collections=["project-docs"],
            hits=[
                QueryHit(
                    score=0.88,
                    file_path="config/auth.php",
                    content="auth config",
                    metadata={},
                    preview="auth config",
                    collection="project-docs",
                )
            ],
            timings=QueryTimings(0.1, 0.2, 0.0),
            candidates=1,
        )
        return github_qdrant_cli.AnswerResponse(
            question=kwargs["question"],
            answer="Configure authentication in config/auth.php [1].",
            retrieval=retrieval,
            model="mistral-large-2512",
            timings=github_qdrant_cli.AnswerTimings(0.1, 0.2),
            context_chars=256,
        )

    monkeypatch.setattr(github_qdrant_tui, "generate_answer", fake_generate_answer)

    async def run_app():
        tui = github_qdrant_tui.GithubQdrantSyncApp(config=config)
        async with tui.run_test(size=(110, 32)) as pilot:
            tui.handle_command("/ask How do I configure auth?")
            assert tui.operation_running is True
            await tui.active_worker.wait()
            await pilot.pause()
            sources = tui.query_one("#sources", DataTable)
            assert sources.row_count == 1
            assert tui.operation_running is False

    asyncio.run(run_app())

    assert calls[0]["question"] == "How do I configure auth?"


def test_textual_scope_limit_and_parent_commands_update_state(tmp_path):
    config = tmp_path / "config.yaml"
    repo_list = tmp_path / "repositories.yaml"
    _write_answer_config(config)
    _write_repo_list(repo_list)

    async def run_app():
        tui = github_qdrant_tui.GithubQdrantSyncApp(
            config=config,
            repo_list=repo_list,
        )
        async with tui.run_test(size=(100, 30)):
            assert tui.collection is None
            tui.handle_command("scope project-two")
            tui.handle_command("limit 5")
            tui.handle_command("parent on")
            assert tui.collection == "project-two"
            assert tui.limit == 5
            assert tui.with_parent_window is True

    asyncio.run(run_app())


def test_textual_scope_all_without_repo_list_resets_to_config_default(tmp_path):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)

    async def run_app():
        tui = github_qdrant_tui.GithubQdrantSyncApp(
            config=config,
            collection="manual-override",
        )
        async with tui.run_test(size=(100, 30)):
            assert tui.collection == "manual-override"
            tui.handle_command("/scope all")
            assert tui.collection is None
            assert tui._scope_label() == "project-docs"

    asyncio.run(run_app())


def test_textual_can_load_different_config(tmp_path):
    config = tmp_path / "config.yaml"
    alternate = tmp_path / "alternate.yaml"
    _write_answer_config(config)
    _write_answer_config(alternate)
    loaded = yaml.safe_load(alternate.read_text(encoding="utf-8"))
    loaded["qdrant"]["collection_name"] = "alternate-docs"
    alternate.write_text(yaml.safe_dump(loaded, sort_keys=False), encoding="utf-8")

    async def run_app():
        tui = github_qdrant_tui.GithubQdrantSyncApp(config=config)
        async with tui.run_test(size=(100, 30)):
            tui.handle_command(f"/config {alternate}")
            assert tui.config == alternate
            assert tui.collection is None
            assert tui._scope_label() == "alternate-docs"

    asyncio.run(run_app())


def test_textual_config_editing_stages_secret_and_discard(tmp_path):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)

    async def run_app():
        tui = github_qdrant_tui.GithubQdrantSyncApp(config=config)
        async with tui.run_test(size=(100, 30)):
            tui.handle_command("/set retrieval.top_k 5")
            tui.handle_command("/secret qdrant.api_key LOCAL_QDRANT_KEY")
            assert tui.working_config["retrieval"]["top_k"] == 5
            assert tui.working_config["qdrant"]["api_key"] == "${LOCAL_QDRANT_KEY}"
            assert tui.dirty_paths == {"retrieval.top_k", "qdrant.api_key"}
            tui.handle_command("/discard-config-changes")
            assert not tui.dirty_paths
            assert "retrieval" not in tui.working_config

    asyncio.run(run_app())


def test_textual_validate_uses_working_config(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)
    seen = {}

    def fake_validate_config_data(loaded):
        seen["top_k"] = loaded["retrieval"]["top_k"]
        return github_qdrant_cli.ValidationReport(errors=[], warnings=[])

    monkeypatch.setattr(
        github_qdrant_tui, "validate_config_data", fake_validate_config_data
    )

    async def run_app():
        tui = github_qdrant_tui.GithubQdrantSyncApp(config=config)
        async with tui.run_test(size=(100, 30)):
            tui.handle_command("/set retrieval.top_k 4")
            tui.handle_command("/validate")

    asyncio.run(run_app())

    assert seen["top_k"] == 4


def test_textual_search_uses_temp_config_for_unsaved_changes(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)
    calls = {}

    def fake_execute_query(**kwargs):
        config_path = Path(kwargs["config_path"])
        calls["config_path"] = config_path
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        calls["collection_name"] = loaded["qdrant"]["collection_name"]
        assert config_path != config
        assert config_path.exists()
        return QueryResponse(
            query=kwargs["query"],
            collection="changed-docs",
            collections=["changed-docs"],
            hits=[],
            timings=QueryTimings(0.1, 0.2, 0.0),
            candidates=0,
        )

    monkeypatch.setattr(github_qdrant_tui, "execute_query", fake_execute_query)

    async def run_app():
        tui = github_qdrant_tui.GithubQdrantSyncApp(config=config)
        async with tui.run_test(size=(100, 30)):
            tui.handle_command("/set qdrant.collection_name changed-docs")
            tui.handle_command("/search auth")
            await tui.active_worker.wait()

    asyncio.run(run_app())

    assert calls["collection_name"] == "changed-docs"
    assert not calls["config_path"].exists()


def test_textual_config_load_guard_and_force(tmp_path):
    config = tmp_path / "config.yaml"
    alternate = tmp_path / "alternate.yaml"
    _write_answer_config(config)
    _write_answer_config(alternate)
    loaded = yaml.safe_load(alternate.read_text(encoding="utf-8"))
    loaded["qdrant"]["collection_name"] = "alternate-docs"
    alternate.write_text(yaml.safe_dump(loaded, sort_keys=False), encoding="utf-8")

    async def run_app():
        tui = github_qdrant_tui.GithubQdrantSyncApp(config=config)
        async with tui.run_test(size=(100, 30)):
            tui.handle_command("/set retrieval.top_k 4")
            tui.handle_command(f"/config {alternate}")
            assert tui.config == config
            assert tui.dirty_paths
            tui.handle_command(f"/config {alternate} --force")
            assert tui.config == alternate
            assert tui._scope_label() == "alternate-docs"
            assert not tui.dirty_paths

    asyncio.run(run_app())


def test_textual_save_config_creates_backup_and_preserves_comments(tmp_path):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)
    config.write_text(
        "# keep me\n" + config.read_text(encoding="utf-8"), encoding="utf-8"
    )

    async def run_app():
        tui = github_qdrant_tui.GithubQdrantSyncApp(config=config)
        async with tui.run_test(size=(100, 30)):
            tui.handle_command("/set retrieval.top_k 6")
            tui.handle_command("/save-config --confirm")
            assert not tui.dirty_paths

    asyncio.run(run_app())

    saved = config.read_text(encoding="utf-8")
    assert "# keep me" in saved
    assert "top_k: 6" in saved
    backups = list(tmp_path.glob("config.yaml.*.bak"))
    assert len(backups) == 1


def test_textual_save_config_blocks_validation_errors(tmp_path):
    config = tmp_path / "config.yaml"
    _write_answer_config(config)

    async def run_app():
        tui = github_qdrant_tui.GithubQdrantSyncApp(config=config)
        async with tui.run_test(size=(100, 30)):
            tui.handle_command("/set qdrant.vector_size 0")
            tui.handle_command("/save-config --confirm")
            assert tui.dirty_paths == {"qdrant.vector_size"}

    asyncio.run(run_app())

    saved = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert saved["qdrant"]["vector_size"] == 384
    assert not list(tmp_path.glob("config.yaml.*.bak"))


def test_textual_quality_commands_run_workers(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    cases = tmp_path / "eval.yaml"
    _write_answer_config(config)
    cases.write_text("cases:\n  - query: sso\n", encoding="utf-8")
    calls = []

    def fake_run_doctor(**kwargs):
        calls.append(("doctor", kwargs))
        return _doctor_report()

    def fake_run_benchmark(**kwargs):
        calls.append(("benchmark", kwargs))
        return _benchmark_report()

    def fake_run_improve(**kwargs):
        calls.append(("improve", kwargs))
        return _improve_report(applied=kwargs.get("apply", False))

    monkeypatch.setattr(github_qdrant_tui, "run_doctor", fake_run_doctor)
    monkeypatch.setattr(github_qdrant_tui, "run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(github_qdrant_tui, "run_improve", fake_run_improve)

    async def run_app():
        tui = github_qdrant_tui.GithubQdrantSyncApp(config=config)
        async with tui.run_test(size=(110, 32)) as pilot:
            tui.handle_command("/doctor")
            await tui.active_worker.wait()
            await pilot.pause()
            tui.handle_command("/benchmark")
            await tui.active_worker.wait()
            await pilot.pause()
            tui.handle_command("/improve")
            await tui.active_worker.wait()
            await pilot.pause()

    asyncio.run(run_app())

    assert [name for name, _kwargs in calls] == ["doctor", "benchmark", "improve"]
    assert calls[1][1]["cases_path"] == str(cases)


def test_textual_improve_apply_requires_saved_config(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    cases = tmp_path / "eval.yaml"
    _write_answer_config(config)
    cases.write_text("cases:\n  - query: sso\n", encoding="utf-8")

    async def run_app():
        tui = github_qdrant_tui.GithubQdrantSyncApp(config=config)
        async with tui.run_test(size=(100, 30)):
            tui.handle_command("/set retrieval.top_k 5")
            tui.handle_command(f"/improve {cases} --apply")
            assert tui.active_worker is None
            assert tui.dirty_paths

    asyncio.run(run_app())


def test_collections_command_lists_repo_list_targets(tmp_path):
    config = tmp_path / "config.yaml"
    repo_list = tmp_path / "repositories.yaml"
    _write_minimal_config(config)
    _write_repo_list(repo_list)

    result = runner.invoke(
        app,
        [
            "collections",
            str(config),
            "--repo-list",
            str(repo_list),
            "--no-banner",
        ],
    )

    assert result.exit_code == 0
    assert "project-one" in result.output
    assert "Project One" in result.output
    assert "project-two" in result.output


def test_collections_command_can_check_qdrant_and_render_json(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    repo_list = tmp_path / "repositories.yaml"
    _write_minimal_config(config)
    _write_repo_list(repo_list)

    def fake_inspect(_config_path, targets, **_kwargs):
        return [
            rag_retrieval.CollectionCompatibility(
                target=targets[0],
                status="usable",
                reason="collection is compatible.",
                exists=True,
                usable=True,
            ),
            rag_retrieval.CollectionCompatibility(
                target=targets[1],
                status="missing",
                reason="collection does not exist.",
                exists=False,
                usable=False,
            ),
        ]

    monkeypatch.setattr(github_qdrant_cli, "inspect_collection_targets", fake_inspect)

    result = runner.invoke(
        app,
        [
            "collections",
            str(config),
            "--repo-list",
            str(repo_list),
            "--check-qdrant",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    parsed = yaml.safe_load(result.output)
    assert parsed["collections"][0]["collection"] == "project-one"
    assert parsed["collections"][0]["exists"] is True
    assert parsed["collections"][0]["status"] == "usable"
    assert parsed["collections"][1]["exists"] is False
    assert parsed["collections"][1]["status"] == "missing"


def test_wizard_writes_config_with_env_placeholders(tmp_path, monkeypatch):
    output = tmp_path / "generated.yaml"
    answers = iter(
        [
            "https://github.com/example/project.git",
            "main",
            "Project Docs",
            "sentence_transformers",
            "sentence-transformers/all-MiniLM-L6-v2",
            "384",
            "http://localhost:6333",
            "QDRANT_API_KEY",
            "project-docs",
            "all_text",
            "disabled",
        ]
    )

    monkeypatch.setattr(
        github_qdrant_cli.typer,
        "prompt",
        lambda *_args, **_kwargs: next(answers),
    )

    result = runner.invoke(app, ["wizard", "--output", str(output), "--no-run"])

    assert result.exit_code == 0
    generated = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert generated["embedding_provider"] == "sentence_transformers"
    assert generated["qdrant"]["api_key"] == "${QDRANT_API_KEY}"
    assert generated["qdrant"]["vector_size"] == 384
    assert generated["qdrant"]["quantization"]["enabled"] is False
    assert generated["qdrant"]["quantization"]["method"] == "turbo"
    assert generated["answering"]["provider"] == "mistral_ai"
    assert generated["answering"]["model"] == "mistral-large-2512"
    assert generated["answering"]["temperature"] == 0.2
    assert generated["answering"]["max_context_chars"] == 12000
    assert generated["pdf_processing"]["enabled"] is False
    assert "supersecret" not in output.read_text(encoding="utf-8")


def test_wizard_prompts_for_alternate_filename_when_output_exists(
    tmp_path, monkeypatch
):
    existing = tmp_path / "config.yaml"
    existing.write_text("existing: true\n", encoding="utf-8")
    alternate = tmp_path / "custom-config.yaml"
    answers = iter(
        [
            str(alternate),
            "https://github.com/example/project.git",
            "main",
            "Project Docs",
            "sentence_transformers",
            "sentence-transformers/all-MiniLM-L6-v2",
            "384",
            "http://localhost:6333",
            "QDRANT_API_KEY",
            "project-docs",
            "all_text",
            "disabled",
        ]
    )

    monkeypatch.setattr(github_qdrant_cli.typer, "confirm", lambda *_args, **_: False)
    monkeypatch.setattr(
        github_qdrant_cli.typer,
        "prompt",
        lambda *_args, **_kwargs: next(answers),
    )

    result = runner.invoke(app, ["wizard", "--output", str(existing), "--no-run"])

    assert result.exit_code == 0
    assert existing.read_text(encoding="utf-8") == "existing: true\n"
    assert alternate.exists()
    generated = yaml.safe_load(alternate.read_text(encoding="utf-8"))
    assert generated["qdrant"]["collection_name"] == "project-docs"


def test_validate_config_command_accepts_minimal_valid_config(tmp_path):
    config = tmp_path / "config.yaml"
    _write_minimal_config(config)

    result = runner.invoke(app, ["validate-config", str(config)])

    assert result.exit_code == 0
    assert "Config structure looks good" in result.output


def test_validate_config_data_reports_provider_specific_missing_fields():
    report = validate_config_data(
        {
            "github": {"repository_url": "https://github.com/example/project.git"},
            "embedding_provider": "azure_openai",
            "azure_openai": {"endpoint": "${AZURE_OPENAI_ENDPOINT}"},
            "qdrant": {
                "collection_name": "docs",
                "vector_size": 3072,
                "distance": "Cosine",
            },
            "processing": {
                "file_mode": "markdown_only",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "markdown_extensions": [".md"],
            },
            "logging": {"level": "INFO", "format": "%(levelname)s: %(message)s"},
        }
    )

    assert not report.ok
    assert "Missing required config field: azure_openai.api_key" in report.errors
    assert "Missing required config field: azure_openai.model" in report.errors
    assert "Missing required config field: azure_openai.api_version" in report.errors


def test_validate_config_data_accepts_mistral_answering_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_minimal_config(config_path)
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    loaded["mistral_ai"] = {
        "api_key": "${MISTRAL_API_KEY}",
        "model": "codestral-embed",
    }
    loaded["answering"] = {
        "provider": "mistral_ai",
        "model": "mistral-large-2512",
        "temperature": 0.2,
        "max_context_chars": 12000,
    }

    report = validate_config_data(loaded)

    assert report.ok


def test_validate_config_data_accepts_qdrant_turboquant_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_minimal_config(config_path)
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    loaded["qdrant"]["quantization"] = {
        "enabled": True,
        "method": "turbo",
        "bits": "bits4",
        "always_ram": True,
        "apply_to_existing_collections": False,
        "search": {"ignore": False, "rescore": True, "oversampling": 2.0},
    }

    report = validate_config_data(loaded)

    assert report.ok


def test_validate_config_data_rejects_invalid_qdrant_turboquant_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_minimal_config(config_path)
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    loaded["qdrant"]["quantization"] = {
        "enabled": "yes",
        "method": "scalar",
        "bits": "bits3",
        "search": {"rescore": "true", "oversampling": 0},
    }

    report = validate_config_data(loaded)

    assert not report.ok
    assert "qdrant.quantization.method currently supports only turbo" in report.errors
    assert any(
        "qdrant.quantization.bits must be one of" in error for error in report.errors
    )
    assert "qdrant.quantization.enabled must be true or false" in report.errors
    assert "qdrant.quantization.search.rescore must be true or false" in report.errors
    assert (
        "qdrant.quantization.search.oversampling must be greater than zero"
        in report.errors
    )


def test_tui_can_coerce_qdrant_turboquant_config_values():
    assert (
        github_qdrant_tui._coerce_config_value("qdrant.quantization.enabled", "true")
        is True
    )
    assert (
        github_qdrant_tui._coerce_config_value("qdrant.quantization.bits", "bits1_5")
        == "bits1_5"
    )
    assert (
        github_qdrant_tui._coerce_config_value(
            "qdrant.quantization.search.oversampling", "2.5"
        )
        == 2.5
    )


def test_build_qdrant_turboquant_config():
    config = {
        "quantization": {
            "enabled": True,
            "method": "turbo",
            "bits": "bits2",
            "always_ram": True,
        }
    }

    quantization = github_to_qdrant.build_qdrant_quantization_config(config)

    assert quantization.turbo.bits.value == "bits2"
    assert quantization.turbo.always_ram is True


def test_validate_config_data_reports_invalid_answering_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_minimal_config(config_path)
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    loaded["answering"] = {
        "provider": "sentence_transformers",
        "temperature": "warm",
        "max_context_chars": 0,
    }

    report = validate_config_data(loaded)

    assert not report.ok
    assert "answering.provider must be azure_openai or mistral_ai" in report.errors
    assert "Missing required config field: answering.model" in report.errors
    assert "answering.temperature must be a number" in report.errors
    assert "answering.max_context_chars must be greater than zero" in report.errors


def _quality_hit(file_path, score=0.8, content="sso authentication"):
    return QueryHit(
        score=score,
        file_path=file_path,
        content=content,
        metadata={"source": file_path},
        preview=content,
        collection="project-docs",
    )


def _doctor_report(level="OK"):
    return github_qdrant_quality.DoctorReport(
        config_path="config.yaml",
        targets=[CollectionTarget(collection_name="project-docs")],
        findings=[
            github_qdrant_quality.DoctorFinding(
                level=level,
                check="collection",
                message="Collection exists.",
                collection="project-docs",
            )
        ],
    )


def _benchmark_report(passed=True):
    result = github_qdrant_quality.BenchmarkCaseResult(
        case_id="sso",
        query="sso",
        collection="project-docs",
        passed=passed,
        top_score=0.8,
        hit_at_1=passed,
        hit_at_5=passed,
        hit_at_10=passed,
        mrr=1.0 if passed else 0.0,
        expected_source_rank=1 if passed else None,
        keyword_coverage=1.0 if passed else 0.0,
        latency_seconds=0.01,
        sources=["app/Http/Controllers/Auth/Sso.php"] if passed else [],
    )
    return github_qdrant_quality.BenchmarkReport(
        cases_path="eval.yaml",
        thresholds=github_qdrant_quality.QualityThresholds(),
        results=[result],
        pass_rate=1.0 if passed else 0.0,
        passed=passed,
    )


def _improve_report(applied=False):
    return github_qdrant_quality.ImproveReport(
        config_path="config.yaml",
        applied=applied,
        backup_path="config.yaml.20260101-000000.bak" if applied else None,
        actions=[
            github_qdrant_quality.ImproveAction(
                category="retrieval",
                action="retrieval.fetch_k",
                status="applied" if applied else "preview",
                message="Set retrieval.fetch_k to 80.",
            )
        ],
        doctor=_doctor_report(),
    )


def test_quality_benchmark_yaml_parsing_and_rubric(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    cases = tmp_path / "eval.yaml"
    _write_minimal_config(config)
    cases.write_text(
        yaml.safe_dump(
            {
                "thresholds": {
                    "pass_rate": 0.8,
                    "expected_source_top_k": 5,
                    "min_top_score": 0.4,
                    "min_keyword_coverage": 0.5,
                },
                "cases": [
                    {
                        "id": "sso",
                        "query": "sso",
                        "expected_sources": ["app/Http/Controllers/Auth/Sso"],
                        "keywords": ["sso", "authentication"],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    def fake_execute_query(**kwargs):
        assert kwargs["limit"] == 10
        return QueryResponse(
            query=kwargs["query"],
            collection="project-docs",
            collections=["project-docs"],
            hits=[_quality_hit("app/Http/Controllers/Auth/Sso.php")],
            timings=QueryTimings(0.01, 0.01, 0.0),
            candidates=1,
        )

    monkeypatch.setattr(github_qdrant_quality, "execute_query", fake_execute_query)

    report = github_qdrant_quality.run_benchmark(str(config), str(cases))

    assert report.passed is True
    assert report.pass_rate == 1.0
    assert report.results[0].hit_at_1 is True
    assert report.results[0].keyword_coverage == 1.0


def test_quality_benchmark_invalid_cases_file(tmp_path):
    cases = tmp_path / "eval.yaml"
    cases.write_text("cases: []\n", encoding="utf-8")

    try:
        github_qdrant_quality.load_benchmark_file(str(cases))
    except ValueError as exc:
        assert "non-empty cases list" in str(exc)
    else:
        raise AssertionError("Expected invalid benchmark file to fail")


def test_quality_benchmark_default_cases_resolves_next_to_config(tmp_path):
    config = tmp_path / "config.yaml"
    cases = tmp_path / "eval.yml"
    _write_minimal_config(config)
    cases.write_text("cases:\n  - query: auth\n", encoding="utf-8")

    resolved = github_qdrant_quality.resolve_benchmark_cases(None, config)

    assert resolved == cases


def test_quality_doctor_reports_healthy_collection(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_minimal_config(config)
    loaded = yaml.safe_load(config.read_text(encoding="utf-8"))
    loaded["qdrant"]["payload_indexes"] = {
        "enabled": True,
        "fields": [{"name": "source", "type": "keyword"}],
    }
    config.write_text(yaml.safe_dump(loaded, sort_keys=False), encoding="utf-8")

    class VectorParams:
        size = 384
        distance = "Cosine"

    class Params:
        vectors = VectorParams()

    class InfoConfig:
        params = Params()

    class Info:
        config = InfoConfig()
        points_count = 3
        payload_schema = {"metadata.source": "keyword"}

    class Record:
        payload = {
            "page_content": "auth docs",
            "metadata": {"source": "README.md"},
        }

    class Client:
        def collection_exists(self, collection_name):
            assert collection_name == "project-docs"
            return True

        def get_collection(self, collection_name):
            return Info()

        def scroll(self, **_kwargs):
            return [Record()], None

    monkeypatch.setattr(
        github_qdrant_quality, "_init_qdrant_client", lambda _cfg: Client()
    )

    report = github_qdrant_quality.run_doctor(str(config))

    assert report.ok is True
    assert any(finding.check == "payload_indexes" for finding in report.findings)
    assert not any(finding.level == "ERROR" for finding in report.findings)


def test_quality_doctor_missing_collection_is_error(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_minimal_config(config)

    class Client:
        def collection_exists(self, collection_name):
            return False

    monkeypatch.setattr(
        github_qdrant_quality, "_init_qdrant_client", lambda _cfg: Client()
    )

    report = github_qdrant_quality.run_doctor(str(config))

    assert report.ok is False
    assert "does not exist" in report.findings[0].message


def test_quality_doctor_apply_indexes_requires_yes(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_minimal_config(config)
    loaded = yaml.safe_load(config.read_text(encoding="utf-8"))
    loaded["qdrant"]["payload_indexes"] = {
        "enabled": True,
        "fields": [{"name": "source", "type": "keyword"}],
    }
    config.write_text(yaml.safe_dump(loaded, sort_keys=False), encoding="utf-8")
    created = []

    class VectorParams:
        size = 384
        distance = "Cosine"

    class Params:
        vectors = VectorParams()

    class InfoConfig:
        params = Params()

    class Info:
        config = InfoConfig()
        points_count = 1
        payload_schema = {}

    class Record:
        payload = {"page_content": "docs", "metadata": {"source": "README.md"}}

    class Client:
        def collection_exists(self, collection_name):
            return True

        def get_collection(self, collection_name):
            return Info()

        def scroll(self, **_kwargs):
            return [Record()], None

        def create_payload_index(self, **kwargs):
            created.append(kwargs["field_name"])

    monkeypatch.setattr(
        github_qdrant_quality, "_init_qdrant_client", lambda _cfg: Client()
    )

    no_confirm = github_qdrant_quality.run_doctor(
        str(config), apply_indexes=True, yes=False
    )
    confirmed = github_qdrant_quality.run_doctor(
        str(config), apply_indexes=True, yes=True
    )

    assert created == ["metadata.source"]
    assert any("not created" in finding.message for finding in no_confirm.findings)
    assert any("Created 1" in finding.message for finding in confirmed.findings)


def test_quality_improve_preview_and_apply_safe_config_changes(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_minimal_config(config)
    loaded = yaml.safe_load(config.read_text(encoding="utf-8"))
    loaded["retrieval"] = {"top_k": 3}
    config.write_text(yaml.safe_dump(loaded, sort_keys=False), encoding="utf-8")

    monkeypatch.setattr(
        github_qdrant_quality,
        "run_doctor",
        lambda *_args, **_kwargs: github_qdrant_quality.DoctorReport(
            config_path=str(config),
            targets=[CollectionTarget(collection_name="project-docs")],
            findings=[
                github_qdrant_quality.DoctorFinding(
                    level="WARN",
                    check="payload_indexes",
                    message="Payload indexes are disabled.",
                    collection="project-docs",
                    fixable=True,
                )
            ],
        ),
    )

    preview = github_qdrant_quality.run_improve(str(config))
    applied = github_qdrant_quality.run_improve(str(config), apply=True, yes=True)

    saved = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert preview.applied is False
    assert applied.applied is True
    assert applied.backup_path
    assert saved["qdrant"]["payload_indexes"]["enabled"] is True
    assert saved["retrieval"]["fetch_k"] == 80
    assert saved["retrieval"]["top_k"] == 10
    assert list(tmp_path.glob("config.yaml.*.bak"))


def test_doctor_command_renders_json(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_minimal_config(config)
    monkeypatch.setattr(
        github_qdrant_cli, "run_doctor", lambda **_kwargs: _doctor_report()
    )

    result = runner.invoke(app, ["doctor", str(config), "--format", "json"])

    assert result.exit_code == 0
    parsed = yaml.safe_load(result.output)
    assert parsed["ok"] is True
    assert parsed["findings"][0]["check"] == "collection"


def test_benchmark_command_improve_on_fail(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    cases = tmp_path / "eval.yaml"
    _write_minimal_config(config)
    cases.write_text("cases:\n  - query: sso\n", encoding="utf-8")
    monkeypatch.setattr(
        github_qdrant_cli, "run_benchmark", lambda **_kwargs: _benchmark_report(False)
    )
    monkeypatch.setattr(
        github_qdrant_cli, "run_improve", lambda **_kwargs: _improve_report(False)
    )

    result = runner.invoke(
        app,
        [
            "benchmark",
            str(config),
            "--improve-on-fail",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    parsed = yaml.safe_load(result.output)
    assert parsed["benchmark"]["passed"] is False
    assert parsed["improve"]["actions"][0]["action"] == "retrieval.fetch_k"


def test_improve_command_can_apply_with_confirmation(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    _write_minimal_config(config)
    calls = {}

    def fake_run_improve(**kwargs):
        calls.update(kwargs)
        return _improve_report(applied=kwargs["apply"] and kwargs["yes"])

    monkeypatch.setattr(github_qdrant_cli, "run_improve", fake_run_improve)

    result = runner.invoke(app, ["improve", str(config), "--apply", "--yes"])

    assert result.exit_code == 0
    assert calls["apply"] is True
    assert calls["yes"] is True
    assert "Backup created" in result.output


def test_validate_config_data_accepts_azure_answering_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    _write_answer_config(config_path, provider="azure_openai")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    loaded["answering"]["model"] = "docs-chat-deployment"

    report = validate_config_data(loaded)

    assert report.ok


def test_config_loader_uses_env_file_next_to_config(tmp_path, monkeypatch):
    config_dir = tmp_path / "project"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text("qdrant:\n  url: ${QDRANT_URL}\n", encoding="utf-8")
    (config_dir / ".env").write_text(
        "QDRANT_URL=http://localhost:6333\n", encoding="utf-8"
    )
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.chdir(tmp_path)

    cfg = ConfigLoader.load_config(str(config_path))

    assert cfg["qdrant"]["url"] == "http://localhost:6333"
