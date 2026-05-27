#!/usr/bin/env python3
"""Typer-based command line interface for GitHub to Qdrant sync."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import typer
import yaml
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

MODULE_DIR = Path(__file__).resolve().parent
if sys.path[0] != str(MODULE_DIR):
    sys.path.insert(0, str(MODULE_DIR))

# pylint: disable=wrong-import-position
from github_to_qdrant import ConfigLoader, GitHubToQdrantProcessor, run_ingest  # noqa: E402
from github_qdrant_quality import (  # noqa: E402
    benchmark_table,
    benchmark_to_dict,
    doctor_table,
    doctor_to_dict,
    improve_table,
    improve_to_dict,
    resolve_benchmark_cases,
    run_benchmark,
    run_doctor,
    run_improve,
)
from rag_retrieval import (  # noqa: E402
    CollectionCompatibility,
    CollectionTarget,
    QueryResponse,
    execute_query,
    inspect_collection_targets,
    load_repository_targets,
    redact_metadata,
    resolve_collection_targets,
    run_query,
)
# pylint: enable=wrong-import-position


console = Console()
PROJECT_NAME = "GithubQdrant-Sync"
PROJECT_REPO_LABEL = "maholick/github-qdrant-sync"
PROJECT_REPO_URL = "https://github.com/maholick/github-qdrant-sync"
FALLBACK_VERSION = "0.5.0"
DEFAULT_MISTRAL_CHAT_MODEL = "mistral-large-2512"
DEFAULT_AZURE_CHAT_MODEL = "${AZURE_OPENAI_CHAT_DEPLOYMENT}"
DEFAULT_ANSWER_SYSTEM_PROMPT = (
    "Answer using only the retrieved repository context. If the context is "
    "insufficient, say what is missing and cite the closest matching sources."
)
ANSWER_PROVIDERS = {"azure_openai", "mistral_ai"}
SECRET_CONFIG_PATHS = (
    "github.token",
    "mistral_ai.api_key",
    "azure_openai.api_key",
    "qdrant.api_key",
)
DEFAULT_CONFIG_CANDIDATES = (
    Path("config.yaml"),
    Path("config.yml"),
    Path("config.json"),
)
app = typer.Typer(
    help=(
        "[bold cyan]Process GitHub repositories into Qdrant[/bold cyan] "
        "and query the resulting vectors."
    ),
    no_args_is_help=False,
    rich_markup_mode="rich",
)


class OutputFormat(str, Enum):
    """Supported query output formats."""

    RICH = "rich"
    TEXT = "text"
    JSON = "json"


class CollectionOutputFormat(str, Enum):
    """Supported collection listing output formats."""

    RICH = "rich"
    JSON = "json"


@dataclass
class ValidationReport:
    """Validation findings for one config file."""

    errors: List[str]
    warnings: List[str]

    @property
    def ok(self) -> bool:
        """Return whether validation found no errors."""
        return not self.errors


@dataclass
class WizardConfigAnswers:
    """Collected wizard answers used to generate a config file."""

    repository_url: str
    branch: str
    name: str
    embedding_provider: str
    provider_model: str
    provider_dimensions: int
    qdrant_url: str
    qdrant_api_key_env: str
    collection_name: str
    file_mode: str
    pdf_mode: str
    provider_secret_env: Optional[str] = None
    azure_endpoint: str = "${AZURE_OPENAI_ENDPOINT}"


@dataclass
class AnswerTimings:
    """Timing measurements for answer generation."""

    retrieval_seconds: float
    answer_seconds: float


@dataclass
class AnswerResponse:
    """Display-ready AI answer response."""

    question: str
    answer: str
    retrieval: QueryResponse
    model: str
    timings: AnswerTimings
    context_chars: int


def _version_callback(value: bool) -> None:
    """Print the CLI version and exit."""
    if value:
        console.print(f"{PROJECT_NAME} v{_app_version()}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def cli_callback(
    ctx: typer.Context,
    version_option: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the installed CLI version and exit.",
        rich_help_panel="Output",
    ),
) -> None:
    """Process GitHub repositories into Qdrant and query the resulting vectors."""
    _ = version_option
    if ctx.invoked_subcommand is None:
        try:
            resolved_config = _resolve_default_config(None)
        except (LookupError, ValueError) as exc:
            if _can_run_tui():
                _run_textual_interactive(
                    config=_initial_config_target(None),
                    limit=None,
                    with_parent_window=False,
                    collection=None,
                    repo_list=None,
                    first_run_setup=True,
                )
                raise typer.Exit()
            console.print(f"[bold red]Interactive startup failed:[/bold red] {exc}")
            raise typer.Exit(1) from exc
        if _can_run_tui():
            _run_textual_interactive(
                config=resolved_config,
                limit=None,
                with_parent_window=False,
                collection=None,
                repo_list=None,
                first_run_setup=False,
            )
        else:
            _run_interactive_session(
                config=resolved_config,
                limit=None,
                with_parent_window=False,
                no_banner=False,
                collection=None,
                repo_list=None,
            )
        raise typer.Exit()


def _is_missing(value: Any) -> bool:
    """Return true when a config value is absent or blank."""
    return value is None or value == ""


def _get_nested(config: Dict[str, Any], path: str) -> Any:
    """Return a dotted config value, or None when absent."""
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _secret_values(config: Dict[str, Any]) -> List[str]:
    """Return concrete config secret values that should be redacted."""
    values: List[str] = []
    for path in SECRET_CONFIG_PATHS:
        value = _get_nested(config, path)
        if not isinstance(value, str) or not value or value.startswith("${"):
            continue
        values.append(value)
    return values


def _redact_text(text: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Redact known secrets from user-visible text."""
    redacted = str(text)
    for value in _secret_values(config or {}):
        redacted = redacted.replace(value, "<redacted>")
    return re.sub(
        r"https://[^\s/@]+@github\.com",
        "https://<redacted>@github.com",
        redacted,
    )


def _redacted_exception(exc: BaseException, config_path: Optional[Path] = None) -> str:
    """Return a display-safe exception string."""
    config: Dict[str, Any] = {}
    if config_path is not None and config_path.exists():
        try:
            config = _load_runtime_config(str(config_path), quiet=True)
        except Exception:  # pragma: no cover - defensive redaction fallback
            config = {}
    return _redact_text(str(exc), config)


def _is_unresolved_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("${") and value.endswith("}")


def _placeholder(env_name: str) -> str:
    env_name = env_name.strip()
    if env_name.startswith("${") and env_name.endswith("}"):
        return env_name
    return "${" + env_name + "}"


def _resolve_default_config(config: Optional[Path]) -> Path:
    """Resolve an explicit config or find the default config in the cwd."""
    if config is not None:
        resolved = config
        if not resolved.exists():
            raise ValueError(f"Config file not found: {resolved}")
        if not resolved.is_file():
            raise ValueError(f"Config path is not a file: {resolved}")
        return resolved

    for candidate in DEFAULT_CONFIG_CANDIDATES:
        if candidate.exists() and candidate.is_file():
            return candidate

    candidates = ", ".join(str(candidate) for candidate in DEFAULT_CONFIG_CANDIDATES)
    raise ValueError(
        f"No default config found. Expected one of: {candidates}. "
        "Pass a config path or run github-qdrant-sync wizard."
    )


def _initial_config_target(config: Optional[Path]) -> Path:
    """Return the config path a first-run setup flow should create."""
    return config if config is not None else DEFAULT_CONFIG_CANDIDATES[0]


def _app_version() -> str:
    try:
        return version("github-qdrant-sync")
    except PackageNotFoundError:
        return FALLBACK_VERSION


def _terminal_width(max_width: int = 96) -> int:
    """Return a decorative width that fits the active terminal."""
    return max(20, min(max_width, console.width - 2))


def _supports_decorative_output(no_banner: bool = False) -> bool:
    """Return whether the current output should include rich decorative UI."""
    if no_banner:
        return False
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return bool(console.is_terminal)


def _can_run_tui() -> bool:
    """Return whether the current terminal can host the Textual app."""
    if os.environ.get("TERM") == "dumb":
        return False
    return bool(console.is_terminal)


def _run_textual_interactive(
    config: Path,
    limit: Optional[int],
    with_parent_window: bool,
    collection: Optional[str],
    repo_list: Optional[Path],
    first_run_setup: bool = False,
) -> None:
    """Launch the Textual UI through a lazy import."""
    from github_qdrant_tui import run_tui  # pylint: disable=import-outside-toplevel

    run_tui(
        config=config,
        limit=limit,
        with_parent_window=with_parent_window,
        collection=collection,
        repo_list=repo_list,
        first_run_setup=first_run_setup,
    )


def _print_startup_screen(mode: str, no_banner: bool = False) -> None:
    """Render the compact brand header for human terminal sessions."""
    if not _supports_decorative_output(no_banner):
        return

    width = _terminal_width()
    repo_link = f"[link={PROJECT_REPO_URL}]{PROJECT_REPO_LABEL}[/link]"
    details = Table.grid(expand=True)
    details.add_column(ratio=1)
    details.add_column(justify="right", no_wrap=True)
    details.add_row(
        "[bold white]GitHub repositories[/bold white] "
        "[cyan]->[/cyan] "
        "[bold white]Qdrant vector knowledge[/bold white]",
        f"[bold cyan]{mode}[/bold cyan]",
    )
    details.add_row(f"[dim]{repo_link} · v{_app_version()}[/dim]", "")

    console.print(
        Panel(
            details,
            title=f" {PROJECT_NAME} ",
            title_align="left",
            border_style="cyan",
            box=box.ROUNDED,
            safe_box=True,
            width=width,
            padding=(1, 2),
        )
    )


class _NoopStatus:
    """Status stand-in used when animation should be suppressed."""

    def update(self, *_args: Any, **_kwargs: Any) -> None:
        """Match rich.status.Status.update without doing anything."""


class _PhaseStatus:
    """Human-visible phase tracker with optional spinner animation."""

    def __init__(self, initial_message: str, enabled: bool) -> None:
        self.initial_message = initial_message
        self.enabled = enabled
        self.completed: List[str] = []
        self._current: Optional[str] = None
        self._status_context: Any = None
        self._status: Any = None

    def __enter__(self) -> "_PhaseStatus":
        if self.enabled:
            self._status_context = console.status(
                f"[bold cyan]{self.initial_message}[/bold cyan]",
                spinner="dots",
                spinner_style="cyan",
            )
            self._status = self._status_context.__enter__()
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._current and self._current not in self.completed:
            self.completed.append(self._current)
        if self._status_context is not None:
            self._status_context.__exit__(None, None, None)

    def update(self, message: str) -> None:
        """Update the active phase and keep each phase visible briefly."""
        if message == self._current:
            return

        if self._current and self._current not in self.completed:
            self.completed.append(self._current)
        self._current = message

        if not self.enabled:
            return

        if self._status is not None:
            self._status.update(f"[bold cyan]{message}[/bold cyan]")
            time.sleep(0.15)


@contextmanager
def _status(message: str, enabled: bool) -> Iterator[Any]:
    """Yield a Rich status spinner only when terminal animation is appropriate."""
    if enabled:
        with console.status(
            f"[bold cyan]{message}[/bold cyan]",
            spinner="dots",
            spinner_style="cyan",
        ) as status:
            yield status
        return

    yield _NoopStatus()


@contextmanager
def _phase_status(message: str, enabled: bool) -> Iterator[_PhaseStatus]:
    """Yield a status helper that also records completed phases."""
    tracker = _PhaseStatus(message, enabled)
    with tracker:
        yield tracker


def _clip(value: str, limit: int = 160) -> str:
    clean = " ".join(str(value or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "..."


def _phase_text(phases: Optional[List[str]]) -> str:
    """Return a compact completed-phase display string."""
    if not phases:
        return ""
    unique_phases = list(dict.fromkeys(phases))
    return " -> ".join(unique_phases)


def _skipped_collection_text(response: QueryResponse, limit: int = 180) -> str:
    """Return a compact display string for unusable collections."""
    if not response.skipped_collections:
        return ""
    skipped = [
        f"{status.collection_name} ({status.status}: {status.reason})"
        for status in response.skipped_collections
    ]
    return _clip("; ".join(skipped), limit)


def _response_is_multi_collection(response: QueryResponse) -> bool:
    collections = response.collections or [response.collection]
    return len({collection for collection in collections if collection}) > 1


def _render_rich_query_response(
    response: QueryResponse, phases: Optional[List[str]] = None
) -> None:
    if not response.hits:
        body = f"[bold yellow]No results found[/bold yellow]\n\n{response.query}"
        if response.warnings:
            body += "\n\n[dim]" + "\n".join(response.warnings) + "[/dim]"
        console.print(
            Panel(
                body,
                border_style="yellow",
                box=box.ROUNDED,
                safe_box=True,
                width=_terminal_width(),
            )
        )
        return

    width = _terminal_width()
    summary = Table.grid(padding=(0, 2), expand=True)
    summary.add_column("Label", style="bold", no_wrap=True)
    summary.add_column("Value", ratio=1)
    summary.add_row("Query", _clip(response.query, 180))
    if _response_is_multi_collection(response):
        summary.add_row(
            "Collections",
            f"{len(response.collections)} searched · {', '.join(response.collections)}",
        )
    else:
        summary.add_row("Collection", response.collection)
    summary.add_row(
        "Results",
        f"{len(response.hits)} matches from {response.candidates} candidates",
    )
    if response.warnings:
        summary.add_row("Warnings", _clip("; ".join(response.warnings), 180))
    skipped_text = _skipped_collection_text(response)
    if skipped_text:
        summary.add_row("Skipped", skipped_text)
    summary.add_row(
        "Timing",
        (
            f"[dim]embed {response.timings.embed_seconds:.2f}s · "
            f"search {response.timings.search_seconds:.2f}s · "
            f"rank {response.timings.group_seconds:.2f}s[/dim]"
        ),
    )
    phase_line = _phase_text(phases)
    if phase_line:
        summary.add_row("Phases", f"[dim]{phase_line}[/dim]")
    console.print(
        Panel(
            summary,
            title=" Vector Search ",
            border_style="cyan",
            box=box.ROUNDED,
            safe_box=True,
            width=width,
            padding=(1, 2),
        )
    )

    table = Table(
        show_header=True,
        header_style="bold cyan",
        expand=False,
        width=width,
        box=box.SIMPLE_HEAD,
        safe_box=True,
    )
    table.add_column("#", justify="right", width=3)
    table.add_column("Score", justify="right", width=7)
    show_collection = _response_is_multi_collection(response)
    if show_collection:
        table.add_column("Collection", style="cyan", min_width=12, max_width=24)
    table.add_column("Source", style="green", min_width=16, max_width=34)
    table.add_column("Matched snippet", overflow="fold")

    for index, hit in enumerate(response.hits, 1):
        row = [
            str(index),
            f"{hit.score:.4f}",
        ]
        if show_collection:
            row.append(_clip(hit.collection or response.collection, 24))
        row.extend(
            [
                _clip(hit.file_path, 42),
                _clip(hit.preview or hit.content, 220),
            ]
        )
        table.add_row(*row)

    console.print(table)

    expanded_hits = [hit for hit in response.hits if hit.expanded_context]
    for index, hit in enumerate(expanded_hits, 1):
        console.print(
            Panel(
                _clip(hit.expanded_context, 2000),
                title=(
                    f"Expanded Context {index}: "
                    f"{hit.collection + ' / ' if hit.collection else ''}{hit.file_path}"
                ),
                border_style="blue",
                box=box.ROUNDED,
                safe_box=True,
                width=_terminal_width(),
            )
        )


def _load_runtime_config(config_path: str, quiet: bool = False) -> Dict[str, Any]:
    """Load a config without the user-facing ConfigLoader status prints."""
    ConfigLoader.load_env_for_config(config_path, quiet=quiet)
    suffix = Path(config_path).suffix.lower()
    with open(config_path, "r", encoding="utf-8") as config_file:
        if suffix in {".yaml", ".yml"}:
            loaded = yaml.safe_load(config_file)
        elif suffix == ".json":
            loaded = json.load(config_file)
        else:
            raise ValueError(f"Unsupported configuration format: {suffix}")
    return ConfigLoader._resolve_env_vars(loaded)


def _answering_config(config: Dict[str, Any]) -> Dict[str, Any]:
    answering = config.get("answering")
    if not isinstance(answering, dict):
        raise ValueError(
            "Missing required 'answering' config. Add answering.provider and "
            "answering.model to enable AI answers."
        )

    provider = answering.get("provider")
    if provider not in ANSWER_PROVIDERS:
        raise ValueError("answering.provider must be azure_openai or mistral_ai")
    if _is_missing(answering.get("model")) or _is_unresolved_placeholder(
        answering.get("model")
    ):
        raise ValueError("answering.model is not configured")
    return answering


def _ensure_configured(value: Any, path: str) -> str:
    if _is_missing(value) or _is_unresolved_placeholder(value):
        raise ValueError(f"{path} is not configured")
    return str(value)


def _build_answer_context(
    response: QueryResponse, max_context_chars: int
) -> tuple[str, int]:
    blocks: List[str] = []
    used_chars = 0
    for index, hit in enumerate(response.hits, 1):
        body = (hit.expanded_context or hit.content or hit.preview or "").strip()
        if not body:
            continue

        collection_line = f"Collection: {hit.collection}\n" if hit.collection else ""
        block = (
            f"[{index}] Source: {hit.file_path}\n"
            f"{collection_line}"
            f"Score: {hit.score:.4f}\n{body}"
        )
        remaining = max_context_chars - used_chars
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = block[:remaining].rstrip()

        blocks.append(block)
        used_chars += len(block)

    return "\n\n---\n\n".join(blocks), used_chars


def _answer_messages(
    question: str, context: str, system_prompt: str
) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                f"{system_prompt}\n\n"
                "Cite sources inline with bracket numbers like [1] when possible."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\nRetrieved repository context:\n{context}"
            ),
        },
    ]


def _extract_chat_content(response: Any) -> str:
    if not response or not getattr(response, "choices", None):
        return ""
    message = response.choices[0].message
    content = getattr(message, "content", "")
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content or "")


def _call_mistral_answer_model(
    config: Dict[str, Any], question: str, context: str
) -> str:
    try:
        from mistralai import Mistral
    except ImportError as exc:
        raise RuntimeError("Mistral AI library is not installed") from exc

    answering = _answering_config(config)
    mcfg = config.get("mistral_ai", {})
    api_key = _ensure_configured(mcfg.get("api_key"), "mistral_ai.api_key")
    model = _ensure_configured(answering.get("model"), "answering.model")
    temperature = float(answering.get("temperature", 0.2))
    system_prompt = str(answering.get("system_prompt") or DEFAULT_ANSWER_SYSTEM_PROMPT)

    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model=model,
        messages=_answer_messages(question, context, system_prompt),
        temperature=temperature,
    )
    return _extract_chat_content(response)


def _call_azure_answer_model(
    config: Dict[str, Any], question: str, context: str
) -> str:
    try:
        from openai import AzureOpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI library is not installed") from exc

    answering = _answering_config(config)
    acfg = config.get("azure_openai", {})
    api_key = _ensure_configured(acfg.get("api_key"), "azure_openai.api_key")
    endpoint = _ensure_configured(acfg.get("endpoint"), "azure_openai.endpoint")
    api_version = _ensure_configured(
        acfg.get("api_version"), "azure_openai.api_version"
    )
    model = _ensure_configured(answering.get("model"), "answering.model")
    temperature = float(answering.get("temperature", 0.2))
    system_prompt = str(answering.get("system_prompt") or DEFAULT_ANSWER_SYSTEM_PROMPT)

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )
    response = client.chat.completions.create(
        model=model,
        messages=_answer_messages(question, context, system_prompt),
        temperature=temperature,
    )
    return _extract_chat_content(response)


def _call_answer_model(config: Dict[str, Any], question: str, context: str) -> str:
    answering = _answering_config(config)
    provider = answering["provider"]
    if provider == "azure_openai":
        return _call_azure_answer_model(config, question, context)
    return _call_mistral_answer_model(config, question, context)


def generate_answer(
    config_path: str,
    question: str,
    limit: Optional[int] = None,
    with_parent_window: bool = False,
    verbose: bool = False,
    quiet: bool = False,
    progress: Optional[Callable[[str], None]] = None,
    collection: Optional[str] = None,
    repo_list: Optional[str] = None,
) -> AnswerResponse:
    """Retrieve context and generate an AI answer."""
    config = _load_runtime_config(config_path, quiet=True)
    answering = _answering_config(config)
    provider = answering["provider"]
    model = str(answering["model"])
    max_context_chars = int(answering.get("max_context_chars", 12000))

    def retrieval_progress(step: str) -> None:
        if not progress:
            return
        progress("Encoding question" if step == "Encoding query" else step)

    retrieval_start = time.time()
    retrieval = execute_query(
        config_path=config_path,
        query=question,
        limit=limit,
        with_parent_window=with_parent_window,
        verbose=verbose,
        quiet=quiet,
        progress=retrieval_progress,
        collection=collection,
        repo_list=repo_list,
    )
    retrieval_seconds = time.time() - retrieval_start

    if not retrieval.hits:
        return AnswerResponse(
            question=question,
            answer=(
                "I could not find matching repository context for this question, "
                "so I cannot generate a grounded answer yet."
            ),
            retrieval=retrieval,
            model=model,
            timings=AnswerTimings(
                retrieval_seconds=retrieval_seconds, answer_seconds=0
            ),
            context_chars=0,
        )

    if progress:
        progress("Preparing context")
    context, context_chars = _build_answer_context(retrieval, max_context_chars)

    if progress:
        provider_name = "Azure OpenAI" if provider == "azure_openai" else "Mistral"
        progress(f"Generating answer with {provider_name}")
    answer_start = time.time()
    answer = _call_answer_model(config, question, context)
    answer_seconds = time.time() - answer_start

    return AnswerResponse(
        question=question,
        answer=answer.strip(),
        retrieval=retrieval,
        model=model,
        timings=AnswerTimings(
            retrieval_seconds=retrieval_seconds,
            answer_seconds=answer_seconds,
        ),
        context_chars=context_chars,
    )


def _sources_table(response: QueryResponse) -> Table:
    width = _terminal_width()
    table = Table(
        show_header=True,
        header_style="bold cyan",
        expand=False,
        width=width,
        box=box.SIMPLE_HEAD,
        safe_box=True,
    )
    table.add_column("#", justify="right", width=3)
    table.add_column("Score", justify="right", width=7)
    show_collection = _response_is_multi_collection(response)
    if show_collection:
        table.add_column("Collection", style="cyan", min_width=12, max_width=24)
    table.add_column("Source", style="green", min_width=16, max_width=34)
    table.add_column("Matched snippet", overflow="fold")
    for index, hit in enumerate(response.hits, 1):
        row = [str(index), f"{hit.score:.4f}"]
        if show_collection:
            row.append(_clip(hit.collection or response.collection, 24))
        row.extend(
            [
                _clip(hit.file_path, 42),
                _clip(hit.preview or hit.content, 220),
            ]
        )
        table.add_row(*row)
    return table


def _render_rich_answer_response(
    response: AnswerResponse,
    show_sources: bool = True,
    phases: Optional[List[str]] = None,
) -> None:
    width = _terminal_width()
    summary = Table.grid(padding=(0, 2), expand=True)
    summary.add_column("Label", style="bold", no_wrap=True)
    summary.add_column("Value", ratio=1)
    summary.add_row("Question", _clip(response.question, 180))
    summary.add_row("Model", response.model)
    summary.add_row(
        "Context",
        f"{len(response.retrieval.hits)} sources · {response.context_chars} chars",
    )
    skipped_text = _skipped_collection_text(response.retrieval)
    if skipped_text:
        summary.add_row("Skipped", skipped_text)
    summary.add_row(
        "Timing",
        (
            f"[dim]retrieve {response.timings.retrieval_seconds:.2f}s · "
            f"answer {response.timings.answer_seconds:.2f}s[/dim]"
        ),
    )
    phase_line = _phase_text(phases)
    if phase_line:
        summary.add_row("Phases", f"[dim]{phase_line}[/dim]")

    console.print(
        Panel(
            summary,
            title=" AI Answer ",
            border_style="cyan",
            box=box.ROUNDED,
            safe_box=True,
            width=width,
            padding=(1, 2),
        )
    )
    console.print(
        Panel(
            response.answer or "No answer was returned by the chat model.",
            title=" Answer ",
            border_style="green" if response.retrieval.hits else "yellow",
            box=box.ROUNDED,
            safe_box=True,
            width=width,
            padding=(1, 2),
        )
    )

    if show_sources and response.retrieval.hits:
        console.print(_sources_table(response.retrieval))


def _render_json_answer_response(response: AnswerResponse) -> None:
    output = {
        "question": response.question,
        "answer": response.answer,
        "model": response.model,
        "context_chars": response.context_chars,
        "collections": response.retrieval.collections
        or [response.retrieval.collection],
        "warnings": response.retrieval.warnings,
        "skipped_collections": [
            status.to_dict() for status in response.retrieval.skipped_collections
        ],
        "collection_statuses": [
            status.to_dict() for status in response.retrieval.collection_statuses
        ],
        "sources": [
            {
                "score": hit.score,
                "file_path": hit.file_path,
                "collection": hit.collection or response.retrieval.collection,
                "preview": hit.preview,
                "content": hit.content,
                "metadata": redact_metadata(hit.metadata),
            }
            for hit in response.retrieval.hits
        ],
        "timing": {
            "retrieval_seconds": response.timings.retrieval_seconds,
            "answer_seconds": response.timings.answer_seconds,
        },
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


def _render_text_answer_response(
    response: AnswerResponse, show_sources: bool = True
) -> None:
    for warning in response.retrieval.warnings:
        print(f"warning: {warning}")
    for status in response.retrieval.skipped_collections:
        print(
            "skipped collection: "
            f"{status.collection_name} ({status.status}) {status.reason}"
        )
    print(response.answer)
    if show_sources and response.retrieval.hits:
        print("\nSources:")
        for index, hit in enumerate(response.retrieval.hits, 1):
            collection_text = f" collection={hit.collection}" if hit.collection else ""
            print(f"{index}. {hit.file_path} score={hit.score:.4f}{collection_text}")


def run_ask(
    config_path: str,
    question: str,
    limit: Optional[int] = None,
    with_parent_window: bool = False,
    verbose: bool = False,
    quiet: bool = False,
    output_format: str = "rich",
    show_sources: bool = True,
    collection: Optional[str] = None,
    repo_list: Optional[str] = None,
) -> int:
    """Run a retrieval-augmented answer request."""
    try:
        response = generate_answer(
            config_path=config_path,
            question=question,
            limit=limit,
            with_parent_window=with_parent_window,
            verbose=verbose,
            quiet=quiet,
            collection=collection,
            repo_list=repo_list,
        )
    except (LookupError, ValueError, RuntimeError, SystemExit) as exc:
        console.print(
            f"[bold red]Ask failed:[/bold red] {_redacted_exception(exc, Path(config_path))}"
        )
        return 1

    if output_format == "json":
        _render_json_answer_response(response)
    elif output_format == "text":
        _render_text_answer_response(response, show_sources=show_sources)
    else:
        _render_rich_answer_response(response, show_sources=show_sources)
    return 0


def _resolve_cli_collection_targets(
    config: Path,
    collection: Optional[str] = None,
    repo_list: Optional[Path] = None,
) -> List[CollectionTarget]:
    loaded = _load_runtime_config(str(config), quiet=True)
    return resolve_collection_targets(
        loaded,
        collection=collection,
        repo_list=str(repo_list) if repo_list else None,
    )


def _collection_scope_label(
    config: Path,
    collection: Optional[str],
    repo_list: Optional[Path],
) -> str:
    if repo_list and not collection:
        try:
            count = len(load_repository_targets(str(repo_list)))
            return f"all repo-list collections ({count})"
        except (FileNotFoundError, ValueError):
            return "all repo-list collections"
    if collection:
        return collection
    try:
        target = _resolve_cli_collection_targets(config)[0]
        return target.collection_name
    except (LookupError, ValueError, FileNotFoundError):
        return "config default"


def _render_collection_targets(
    targets: List[CollectionTarget],
    compatibilities: Optional[List[CollectionCompatibility]] = None,
) -> None:
    compatibility_by_name = {
        status.collection_name: status for status in compatibilities or []
    }
    table = Table(
        title="Collections",
        show_header=True,
        header_style="bold cyan",
        box=box.SIMPLE_HEAD,
        safe_box=True,
        expand=False,
        width=_terminal_width(),
    )
    table.add_column("#", justify="right", width=3)
    table.add_column("Collection", style="cyan", min_width=16)
    table.add_column("Repository", style="green", min_width=18)
    table.add_column("Branch", min_width=8)
    if compatibilities is not None:
        table.add_column("Status", min_width=12)
        table.add_column("Reason", min_width=24)

    for index, target in enumerate(targets, 1):
        row = [
            str(index),
            target.collection_name,
            _clip(target.repository_name or target.repository_url or "config", 36),
            target.branch or "default",
        ]
        if compatibilities is not None:
            status = compatibility_by_name.get(target.collection_name)
            row.extend(
                [
                    status.status if status else "unknown",
                    _clip(status.reason if status else "not checked", 80),
                ]
            )
        table.add_row(*row)
    console.print(table)


def _render_collection_targets_json(
    targets: List[CollectionTarget],
    compatibilities: Optional[List[CollectionCompatibility]] = None,
) -> None:
    compatibility_by_name = {
        status.collection_name: status for status in compatibilities or []
    }
    output = {
        "collections": [
            {
                "collection": target.collection_name,
                "repository_url": target.repository_url,
                "repository_name": target.repository_name,
                "branch": target.branch,
                "exists": (
                    compatibility_by_name[target.collection_name].exists
                    if target.collection_name in compatibility_by_name
                    else None
                ),
                "usable": (
                    compatibility_by_name[target.collection_name].usable
                    if target.collection_name in compatibility_by_name
                    else None
                ),
                "status": (
                    compatibility_by_name[target.collection_name].status
                    if target.collection_name in compatibility_by_name
                    else None
                ),
                "reason": (
                    compatibility_by_name[target.collection_name].reason
                    if target.collection_name in compatibility_by_name
                    else None
                ),
            }
            for target in targets
        ]
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


def _prompt_choice(label: str, choices: List[str], default: str) -> str:
    choices_display = ", ".join(choices)
    while True:
        answer = typer.prompt(f"{label} [{choices_display}]", default=default)
        normalized = str(answer).strip()
        if normalized in choices:
            return normalized
        console.print(
            f"[red]Invalid choice:[/red] {normalized}. Choose one of: {choices_display}"
        )


def _prompt_int(label: str, default: int) -> int:
    while True:
        answer = typer.prompt(label, default=str(default))
        try:
            return int(str(answer).strip())
        except ValueError:
            console.print("[red]Enter a whole number.[/red]")


def _default_text_extensions() -> List[str]:
    return [
        ".md",
        ".markdown",
        ".txt",
        ".rst",
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".java",
        ".go",
        ".rs",
        ".php",
        ".rb",
        ".c",
        ".cpp",
        ".cs",
        ".html",
        ".css",
        ".scss",
        ".vue",
        ".svelte",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".env",
        ".sh",
        ".bash",
        ".dockerfile",
        ".gitignore",
    ]


def _default_config(answers: WizardConfigAnswers) -> Dict[str, Any]:
    pdf_enabled = answers.pdf_mode != "disabled"
    normalized_pdf_mode = (
        "local" if answers.pdf_mode == "disabled" else answers.pdf_mode
    )
    answer_provider = (
        answers.embedding_provider
        if answers.embedding_provider in ANSWER_PROVIDERS
        else "mistral_ai"
    )
    answer_model = (
        DEFAULT_AZURE_CHAT_MODEL
        if answer_provider == "azure_openai"
        else DEFAULT_MISTRAL_CHAT_MODEL
    )

    config: Dict[str, Any] = {
        "github": {
            "repository_url": answers.repository_url,
            "branch": answers.branch,
            "name": answers.name,
            "clone_depth": 1,
            "cleanup_after_processing": True,
            "token": "${GITHUB_TOKEN}",
        },
        "embedding_provider": answers.embedding_provider,
        "azure_openai": {
            "api_key": "${AZURE_OPENAI_API_KEY}",
            "endpoint": answers.azure_endpoint,
            "model": "text-embedding-3-large",
            "api_version": "2024-02-01",
            "dimensions": 3072,
        },
        "mistral_ai": {
            "api_key": "${MISTRAL_API_KEY}",
            "model": "codestral-embed",
            "dimensions": 3072,
            "api_base": "https://api.mistral.ai/v1",
        },
        "answering": {
            "provider": answer_provider,
            "model": answer_model,
            "temperature": 0.2,
            "max_context_chars": 12000,
            "system_prompt": DEFAULT_ANSWER_SYSTEM_PROMPT,
        },
        "sentence_transformers": {
            "model": "intfloat/multilingual-e5-large",
            "dimensions": 1024,
        },
        "qdrant": {
            "url": answers.qdrant_url,
            "api_key": _placeholder(answers.qdrant_api_key_env),
            "collection_name": answers.collection_name,
            "vector_size": answers.provider_dimensions,
            "distance": "Cosine",
            "vector_name": None,
            "recreate_collection": False,
            "connection_method": "auto",
            "timeout": 30,
            "quantization": {
                "enabled": False,
                "method": "turbo",
                "bits": "bits4",
                "always_ram": True,
                "apply_to_existing_collections": False,
                "search": {
                    "ignore": False,
                    "rescore": True,
                    "oversampling": 2.0,
                },
            },
            "payload_indexes": {
                "enabled": False,
                "apply_to_existing_collections": True,
                "fields": [
                    {"name": "repository", "type": "keyword"},
                    {"name": "source", "type": "keyword"},
                    {"name": "source_type", "type": "keyword"},
                    {"name": "content_hash", "type": "keyword"},
                ],
            },
        },
        "processing": {
            "file_mode": answers.file_mode,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "chunking_strategy": "recursive",
            "chunk_size_tokens": 512,
            "chunk_overlap_tokens": 64,
            "tiktoken_encoding": "cl100k_base",
            "markdown_extensions": [".md", ".markdown", ".mdown", ".mkd", ".mdx"],
            "text_extensions": _default_text_extensions(),
            "exclude_patterns": [
                "node_modules",
                ".git",
                "__pycache__",
                "*.pyc",
                ".DS_Store",
                "dist",
                "build",
                "*.min.js",
                "*.min.css",
                "vendor",
            ],
            "combine_documents": False,
            "track_file_changes": False,
            "cleanup_orphaned_markers": False,
            "legacy_cleanup_delete_by_file_path": False,
            "deduplication_enabled": True,
            "similarity_threshold": 0.95,
            "embedding_batch_size": 50,
            "max_retries": 3,
            "batch_delay_seconds": 1,
        },
        "pdf_processing": {
            "enabled": pdf_enabled,
            "mode": normalized_pdf_mode,
            "extract_images": False,
            "image_processing_mode": "none",
            "local": {
                "primary_method": "pymupdf",
                "fallback_method": "pypdfloader",
                "min_text_per_page": 50,
                "preserve_layout": True,
            },
            "cloud": {
                "enabled": normalized_pdf_mode in {"cloud", "hybrid"},
                "provider": "mistral_ocr",
                "max_pages_per_doc": 100,
                "use_for_quality": False,
            },
            "hybrid": {
                "prefer_local": True,
                "quality_threshold": 0.7,
                "force_cloud_patterns": ["*scan*.pdf", "*scanned*.pdf", "*ocr*.pdf"],
            },
        },
        "output": {
            "base_directory": "markdown",
            "combined_filename": "__combined_markdown.md",
            "preserve_structure": True,
        },
        "payload": {
            "content_fields": ["content", "page_content"],
            "preview_length": 200,
            "minimal_mode": False,
            "metadata_denylist": ["page_content", "content", "text", "document"],
        },
        "retrieval": {
            "top_k": 10,
            "fetch_k": 40,
            "max_chunks_per_file": 3,
            "parent_window": 2,
            "metadata_structure": "nested",
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    }

    if answers.embedding_provider == "azure_openai":
        config["azure_openai"]["model"] = answers.provider_model
        config["azure_openai"]["dimensions"] = answers.provider_dimensions
        if answers.provider_secret_env:
            config["azure_openai"]["api_key"] = _placeholder(
                answers.provider_secret_env
            )
    elif answers.embedding_provider == "mistral_ai":
        config["mistral_ai"]["model"] = answers.provider_model
        config["mistral_ai"]["dimensions"] = answers.provider_dimensions
        if answers.provider_secret_env:
            config["mistral_ai"]["api_key"] = _placeholder(answers.provider_secret_env)
    else:
        config["sentence_transformers"]["model"] = answers.provider_model
        config["sentence_transformers"]["dimensions"] = answers.provider_dimensions

    return config


def _provider_answers(provider: str) -> Dict[str, Any]:
    if provider == "azure_openai":
        endpoint = typer.prompt(
            "Azure OpenAI endpoint", default="${AZURE_OPENAI_ENDPOINT}"
        )
        key_env = typer.prompt(
            "Azure OpenAI API key environment variable",
            default="AZURE_OPENAI_API_KEY",
        )
        model = typer.prompt(
            "Azure OpenAI deployment/model", default="text-embedding-3-large"
        )
        dimensions = _prompt_int("Embedding dimensions", 3072)
        return {
            "model": model,
            "dimensions": dimensions,
            "secret_env": key_env,
            "azure_endpoint": endpoint,
        }

    if provider == "mistral_ai":
        key_env = typer.prompt(
            "Mistral API key environment variable", default="MISTRAL_API_KEY"
        )
        model = typer.prompt("Mistral embedding model", default="codestral-embed")
        dimensions = _prompt_int("Embedding dimensions", 3072)
        return {"model": model, "dimensions": dimensions, "secret_env": key_env}

    model = typer.prompt(
        "Sentence Transformers model", default="intfloat/multilingual-e5-large"
    )
    dimensions = _prompt_int("Embedding dimensions", 1024)
    return {"model": model, "dimensions": dimensions, "secret_env": None}


def _collect_wizard_config() -> Dict[str, Any]:
    repository_url = typer.prompt("GitHub repository URL")
    branch = typer.prompt("Target branch", default="main")
    name = typer.prompt("Repository display name", default="")
    provider = _prompt_choice(
        "Embedding provider",
        ["mistral_ai", "azure_openai", "sentence_transformers"],
        "mistral_ai",
    )
    provider_details = _provider_answers(provider)
    qdrant_url = typer.prompt("Qdrant URL", default="${QDRANT_URL}")
    qdrant_api_key_env = typer.prompt(
        "Qdrant API key environment variable", default="QDRANT_API_KEY"
    )
    collection_name = typer.prompt("Qdrant collection name")
    file_mode = _prompt_choice(
        "File processing mode", ["all_text", "markdown_only"], "all_text"
    )
    pdf_mode = _prompt_choice(
        "PDF processing mode", ["local", "cloud", "hybrid", "disabled"], "local"
    )

    return _default_config(
        WizardConfigAnswers(
            repository_url=repository_url,
            branch=branch,
            name=name,
            embedding_provider=provider,
            provider_model=provider_details["model"],
            provider_dimensions=provider_details["dimensions"],
            qdrant_url=qdrant_url,
            qdrant_api_key_env=qdrant_api_key_env,
            collection_name=collection_name,
            file_mode=file_mode,
            pdf_mode=pdf_mode,
            provider_secret_env=provider_details["secret_env"],
            azure_endpoint=provider_details.get(
                "azure_endpoint", "${AZURE_OPENAI_ENDPOINT}"
            ),
        )
    )


def _required_path(config: Dict[str, Any], path: str, errors: List[str]) -> Any:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            errors.append(f"Missing required config field: {path}")
            return None
        current = current[part]

    if _is_missing(current):
        errors.append(f"Missing required config field: {path}")
    return current


def _find_unresolved_placeholders(obj: Any, prefix: str = "") -> List[str]:
    unresolved: List[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            unresolved.extend(_find_unresolved_placeholders(value, child_prefix))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            child_prefix = f"{prefix}[{index}]"
            unresolved.extend(_find_unresolved_placeholders(value, child_prefix))
    elif _is_unresolved_placeholder(obj):
        unresolved.append(prefix)
    return unresolved


def validate_config_data(config: Dict[str, Any]) -> ValidationReport:
    """Validate config shape without requiring live network connections."""
    errors: List[str] = []
    warnings: List[str] = []

    for section in ["github", "qdrant", "processing", "logging"]:
        if section not in config or not isinstance(config[section], dict):
            errors.append(f"Missing required config section: {section}")

    provider = config.get("embedding_provider")
    if provider not in {"azure_openai", "mistral_ai", "sentence_transformers"}:
        errors.append(
            "embedding_provider must be one of: azure_openai, mistral_ai, "
            "sentence_transformers"
        )

    _required_path(config, "github.repository_url", errors)
    _required_path(config, "qdrant.collection_name", errors)
    _required_path(config, "qdrant.vector_size", errors)
    _required_path(config, "qdrant.distance", errors)
    _required_path(config, "processing.chunk_size", errors)
    _required_path(config, "processing.chunk_overlap", errors)

    qdrant = config.get("qdrant", {})
    quantization = qdrant.get("quantization")
    if quantization is not None:
        if not isinstance(quantization, dict):
            errors.append("qdrant.quantization must be a mapping")
        else:
            method = str(quantization.get("method", "turbo")).strip().lower()
            if method not in {"turbo", "turboquant", "turbo_quant", "turbo-quant"}:
                errors.append(
                    "qdrant.quantization.method currently supports only turbo"
                )
            bits = str(quantization.get("bits", "bits4")).strip().lower()
            if bits not in {"bits1", "bits1_5", "bits2", "bits4"}:
                errors.append(
                    "qdrant.quantization.bits must be one of: "
                    "bits1, bits1_5, bits2, bits4"
                )
            for path in [
                "qdrant.quantization.enabled",
                "qdrant.quantization.always_ram",
                "qdrant.quantization.apply_to_existing_collections",
            ]:
                value = _get_nested(config, path)
                if value is not None and not isinstance(value, bool):
                    errors.append(f"{path} must be true or false")
            search = quantization.get("search")
            if search is not None:
                if not isinstance(search, dict):
                    errors.append("qdrant.quantization.search must be a mapping")
                else:
                    for path in [
                        "qdrant.quantization.search.ignore",
                        "qdrant.quantization.search.rescore",
                    ]:
                        value = _get_nested(config, path)
                        if value is not None and not isinstance(value, bool):
                            errors.append(f"{path} must be true or false")
                    oversampling = search.get("oversampling")
                    if oversampling is not None:
                        try:
                            if float(oversampling) <= 0:
                                errors.append(
                                    "qdrant.quantization.search.oversampling must be greater than zero"
                                )
                        except (TypeError, ValueError):
                            errors.append(
                                "qdrant.quantization.search.oversampling must be a number"
                            )

    file_mode = config.get("processing", {}).get("file_mode", "markdown_only")
    if file_mode not in {"all_text", "markdown_only"}:
        errors.append("processing.file_mode must be all_text or markdown_only")

    if file_mode == "all_text":
        _required_path(config, "processing.text_extensions", errors)
    else:
        _required_path(config, "processing.markdown_extensions", errors)

    pdf_config = config.get("pdf_processing", {})
    pdf_mode = pdf_config.get("mode", "local")
    if pdf_mode not in {"local", "cloud", "hybrid"}:
        errors.append("pdf_processing.mode must be local, cloud, or hybrid")
    if pdf_config.get("enabled") and pdf_mode in {"cloud", "hybrid"}:
        if provider != "mistral_ai":
            warnings.append(
                "Cloud or hybrid PDF processing requires Mistral configuration."
            )

    if provider == "azure_openai":
        _required_path(config, "azure_openai.api_key", errors)
        _required_path(config, "azure_openai.endpoint", errors)
        model = config.get("azure_openai", {}).get("model") or config.get(
            "azure_openai", {}
        ).get("deployment_name")
        if _is_missing(model):
            errors.append("Missing required config field: azure_openai.model")
        _required_path(config, "azure_openai.api_version", errors)
    elif provider == "mistral_ai":
        _required_path(config, "mistral_ai.api_key", errors)
        _required_path(config, "mistral_ai.model", errors)
    elif provider == "sentence_transformers":
        _required_path(config, "sentence_transformers.model", errors)

    answering = config.get("answering")
    if answering is not None:
        if not isinstance(answering, dict):
            errors.append("answering must be a mapping")
        else:
            answering_provider = answering.get("provider")
            if answering_provider not in ANSWER_PROVIDERS:
                errors.append("answering.provider must be azure_openai or mistral_ai")
            _required_path(config, "answering.model", errors)
            if answering_provider == "mistral_ai":
                _required_path(config, "mistral_ai.api_key", errors)
            elif answering_provider == "azure_openai":
                _required_path(config, "azure_openai.api_key", errors)
                _required_path(config, "azure_openai.endpoint", errors)
                _required_path(config, "azure_openai.api_version", errors)

            temperature = answering.get("temperature", 0.2)
            try:
                if not 0 <= float(temperature) <= 2:
                    errors.append("answering.temperature must be between 0 and 2")
            except (TypeError, ValueError):
                errors.append("answering.temperature must be a number")

            max_context_chars = answering.get("max_context_chars", 12000)
            try:
                if int(max_context_chars) <= 0:
                    errors.append(
                        "answering.max_context_chars must be greater than zero"
                    )
            except (TypeError, ValueError):
                errors.append("answering.max_context_chars must be an integer")

    vector_size = config.get("qdrant", {}).get("vector_size")
    try:
        if int(vector_size) <= 0:
            errors.append("qdrant.vector_size must be greater than zero")
    except (TypeError, ValueError):
        errors.append("qdrant.vector_size must be an integer")

    return ValidationReport(errors=errors, warnings=warnings)


def _render_validation_report(report: ValidationReport) -> None:
    table = Table(title="Configuration validation")
    table.add_column("Level")
    table.add_column("Message")

    if not report.errors and not report.warnings:
        table.add_row("OK", "Config structure looks good.")
    for error in report.errors:
        table.add_row("ERROR", error)
    for warning in report.warnings:
        table.add_row("WARN", warning)

    console.print(table)


@app.command()
def ingest(
    config: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to YAML or JSON config.",
        rich_help_panel="Input",
    ),
    repo_url: Optional[str] = typer.Option(
        None,
        "--repo-url",
        help="GitHub repository URL override.",
        rich_help_panel="Input",
    ),
    repo_list: Optional[Path] = typer.Option(
        None,
        "--repo-list",
        exists=True,
        readable=True,
        help="YAML file containing repositories to process.",
        rich_help_panel="Input",
    ),
    no_banner: bool = typer.Option(
        False,
        "--no-banner",
        help="Hide decorative terminal banner.",
        rich_help_panel="Output",
    ),
) -> None:
    """Process one repository or a repository list into Qdrant."""
    _print_startup_screen("Repository ingestion", no_banner=no_banner)
    exit_code = run_ingest(
        config_path=str(config),
        repo_url=repo_url,
        repo_list=str(repo_list) if repo_list else None,
    )
    raise typer.Exit(exit_code)


@app.command()
def query(
    config: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to ingestion config.",
        rich_help_panel="Input",
    ),
    query_text: str = typer.Option(
        ..., "--query", help="Query text.", rich_help_panel="Input"
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        help="Override retrieval.top_k.",
        rich_help_panel="Retrieval",
    ),
    with_parent_window: bool = typer.Option(
        False,
        "--with-parent-window",
        help="Include expanded context around selected hits.",
        rich_help_panel="Retrieval",
    ),
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        help="Search a specific Qdrant collection.",
        rich_help_panel="Retrieval",
    ),
    repo_list: Optional[Path] = typer.Option(
        None,
        "--repo-list",
        exists=True,
        readable=True,
        help="Search all collections listed in repositories.yaml.",
        rich_help_panel="Retrieval",
    ),
    no_banner: bool = typer.Option(
        False,
        "--no-banner",
        help="Hide decorative terminal banner.",
        rich_help_panel="Output",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose logging.",
        rich_help_panel="Logging",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress informational logging.",
        rich_help_panel="Logging",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.RICH,
        "--format",
        help="Output format.",
        rich_help_panel="Output",
    ),
) -> None:
    """Query a populated Qdrant collection."""
    if output_format == OutputFormat.RICH:
        show_decoration = _supports_decorative_output(no_banner or quiet)
        _print_startup_screen("Semantic query", no_banner=no_banner or quiet)
        try:
            with _phase_status("Encoding query", enabled=show_decoration) as status:
                response = execute_query(
                    config_path=str(config),
                    query=query_text,
                    limit=limit,
                    with_parent_window=with_parent_window,
                    verbose=verbose,
                    quiet=quiet or not verbose,
                    progress=lambda step: status.update(step),
                    collection=collection,
                    repo_list=str(repo_list) if repo_list else None,
                )
        except (LookupError, ValueError, SystemExit) as exc:
            console.print(
                f"[bold red]Query failed:[/bold red] {_redacted_exception(exc, config)}"
            )
            raise typer.Exit(1) from exc

        _render_rich_query_response(response, phases=status.completed)
        raise typer.Exit(0)

    exit_code = run_query(
        config_path=str(config),
        query=query_text,
        limit=limit,
        with_parent_window=with_parent_window,
        verbose=verbose,
        quiet=quiet or not verbose,
        output_format=output_format.value,
        collection=collection,
        repo_list=str(repo_list) if repo_list else None,
    )
    raise typer.Exit(exit_code)


@app.command()
def ask(
    config: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to ingestion config.",
        rich_help_panel="Input",
    ),
    question: str = typer.Option(
        ..., "--question", help="Question to answer.", rich_help_panel="Input"
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        help="Override retrieval.top_k.",
        rich_help_panel="Retrieval",
    ),
    with_parent_window: bool = typer.Option(
        False,
        "--with-parent-window",
        help="Include expanded context around selected hits.",
        rich_help_panel="Retrieval",
    ),
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        help="Search a specific Qdrant collection.",
        rich_help_panel="Retrieval",
    ),
    repo_list: Optional[Path] = typer.Option(
        None,
        "--repo-list",
        exists=True,
        readable=True,
        help="Search all collections listed in repositories.yaml.",
        rich_help_panel="Retrieval",
    ),
    show_sources: bool = typer.Option(
        True,
        "--show-sources/--hide-sources",
        help="Show source matches below the answer.",
        rich_help_panel="Output",
    ),
    no_banner: bool = typer.Option(
        False,
        "--no-banner",
        help="Hide decorative terminal banner.",
        rich_help_panel="Output",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose logging.",
        rich_help_panel="Logging",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress informational logging.",
        rich_help_panel="Logging",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.RICH,
        "--format",
        help="Output format.",
        rich_help_panel="Output",
    ),
) -> None:
    """Generate an AI answer from retrieved repository context."""
    if output_format == OutputFormat.RICH:
        show_decoration = _supports_decorative_output(no_banner or quiet)
        _print_startup_screen("AI answer", no_banner=no_banner or quiet)
        try:
            with _phase_status("Encoding question", enabled=show_decoration) as status:
                response = generate_answer(
                    config_path=str(config),
                    question=question,
                    limit=limit,
                    with_parent_window=with_parent_window,
                    verbose=verbose,
                    quiet=quiet or not verbose,
                    progress=lambda step: status.update(step),
                    collection=collection,
                    repo_list=str(repo_list) if repo_list else None,
                )
        except (LookupError, ValueError, RuntimeError, SystemExit) as exc:
            console.print(
                f"[bold red]Ask failed:[/bold red] {_redacted_exception(exc, config)}"
            )
            raise typer.Exit(1) from exc

        _render_rich_answer_response(
            response, show_sources=show_sources, phases=status.completed
        )
        raise typer.Exit(0)

    exit_code = run_ask(
        config_path=str(config),
        question=question,
        limit=limit,
        with_parent_window=with_parent_window,
        verbose=verbose,
        quiet=quiet or not verbose,
        output_format=output_format.value,
        show_sources=show_sources,
        collection=collection,
        repo_list=str(repo_list) if repo_list else None,
    )
    raise typer.Exit(exit_code)


@app.command("collections")
def collections_command(
    config: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to ingestion config.",
        rich_help_panel="Input",
    ),
    repo_list: Optional[Path] = typer.Option(
        None,
        "--repo-list",
        exists=True,
        readable=True,
        help="Show collections from repositories.yaml instead of only config default.",
        rich_help_panel="Input",
    ),
    check_qdrant: bool = typer.Option(
        False,
        "--check-qdrant",
        help="Check existence and embedding/vector compatibility in Qdrant.",
        rich_help_panel="Validation",
    ),
    output_format: CollectionOutputFormat = typer.Option(
        CollectionOutputFormat.RICH,
        "--format",
        help="Output format.",
        rich_help_panel="Output",
    ),
    no_banner: bool = typer.Option(
        False,
        "--no-banner",
        help="Hide decorative terminal banner.",
        rich_help_panel="Output",
    ),
) -> None:
    """List retrievable collections from config or repositories.yaml."""
    try:
        targets = _resolve_cli_collection_targets(config, repo_list=repo_list)
        compatibilities = (
            inspect_collection_targets(str(config), targets, quiet=True)
            if check_qdrant
            else None
        )
    except (FileNotFoundError, ValueError, SystemExit) as exc:
        console.print(
            "[bold red]Collection listing failed:[/bold red] "
            f"{_redacted_exception(exc, config)}"
        )
        raise typer.Exit(1) from exc

    if output_format == CollectionOutputFormat.JSON:
        _render_collection_targets_json(targets, compatibilities=compatibilities)
        raise typer.Exit(0)

    _print_startup_screen("Collections", no_banner=no_banner)
    _render_collection_targets(targets, compatibilities=compatibilities)
    raise typer.Exit(0)


@app.command()
def doctor(
    config: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to ingestion config.",
        rich_help_panel="Input",
    ),
    repo_list: Optional[Path] = typer.Option(
        None,
        "--repo-list",
        exists=True,
        readable=True,
        help="Check all collections listed in repositories.yaml.",
        rich_help_panel="Input",
    ),
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        help="Check a specific Qdrant collection.",
        rich_help_panel="Input",
    ),
    apply_indexes: bool = typer.Option(
        False,
        "--apply-indexes",
        help="Create missing payload indexes idempotently.",
        rich_help_panel="Repair",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Confirm safe index creation when used with --apply-indexes.",
        rich_help_panel="Repair",
    ),
    output_format: CollectionOutputFormat = typer.Option(
        CollectionOutputFormat.RICH,
        "--format",
        help="Output format.",
        rich_help_panel="Output",
    ),
    no_banner: bool = typer.Option(
        False,
        "--no-banner",
        help="Hide decorative terminal banner.",
        rich_help_panel="Output",
    ),
) -> None:
    """Check Qdrant collection, vector, payload, and index health."""
    try:
        report = run_doctor(
            config_path=str(config),
            repo_list=str(repo_list) if repo_list else None,
            collection=collection,
            apply_indexes=apply_indexes,
            yes=yes,
        )
    except (LookupError, ValueError, SystemExit) as exc:
        console.print(
            f"[bold red]Doctor failed:[/bold red] {_redacted_exception(exc, config)}"
        )
        raise typer.Exit(1) from exc

    if output_format == CollectionOutputFormat.JSON:
        print(json.dumps(doctor_to_dict(report), indent=2, ensure_ascii=False))
    else:
        _print_startup_screen("Index doctor", no_banner=no_banner)
        console.print(doctor_table(report))

    raise typer.Exit(0 if report.ok else 1)


@app.command()
def benchmark(
    config: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to ingestion config.",
        rich_help_panel="Input",
    ),
    cases: Optional[Path] = typer.Option(
        None,
        "--cases",
        exists=True,
        readable=True,
        help="Benchmark YAML file. Defaults to eval.yaml/eval.yml.",
        rich_help_panel="Input",
    ),
    repo_list: Optional[Path] = typer.Option(
        None,
        "--repo-list",
        exists=True,
        readable=True,
        help="Search all collections listed in repositories.yaml.",
        rich_help_panel="Retrieval",
    ),
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        help="Search a specific Qdrant collection.",
        rich_help_panel="Retrieval",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        help="Override benchmark retrieval limit.",
        rich_help_panel="Retrieval",
    ),
    fail_under: float = typer.Option(
        0.8,
        "--fail-under",
        help="Required aggregate pass rate.",
        rich_help_panel="Quality",
    ),
    improve_on_fail: bool = typer.Option(
        False,
        "--improve-on-fail",
        help="Render a safe improvement report when benchmark fails.",
        rich_help_panel="Quality",
    ),
    output_format: CollectionOutputFormat = typer.Option(
        CollectionOutputFormat.RICH,
        "--format",
        help="Output format.",
        rich_help_panel="Output",
    ),
    no_banner: bool = typer.Option(
        False,
        "--no-banner",
        help="Hide decorative terminal banner.",
        rich_help_panel="Output",
    ),
) -> None:
    """Run retrieval benchmark cases from YAML."""
    try:
        cases_path = resolve_benchmark_cases(cases, config)
        report = run_benchmark(
            config_path=str(config),
            cases_path=str(cases_path),
            repo_list=str(repo_list) if repo_list else None,
            collection=collection,
            limit=limit,
            fail_under=fail_under,
        )
        improve_report = (
            run_improve(
                config_path=str(config),
                cases_path=str(cases_path),
                repo_list=str(repo_list) if repo_list else None,
                collection=collection,
            )
            if improve_on_fail and not report.passed
            else None
        )
    except (LookupError, ValueError, SystemExit) as exc:
        console.print(
            f"[bold red]Benchmark failed:[/bold red] {_redacted_exception(exc, config)}"
        )
        raise typer.Exit(1) from exc

    if output_format == CollectionOutputFormat.JSON:
        output = {"benchmark": benchmark_to_dict(report)}
        if improve_report is not None:
            output["improve"] = improve_to_dict(improve_report)
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        _print_startup_screen("Retrieval benchmark", no_banner=no_banner)
        console.print(benchmark_table(report))
        if improve_report is not None:
            console.print(improve_table(improve_report))

    raise typer.Exit(0 if report.passed else 1)


@app.command()
def improve(
    config: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to ingestion config.",
        rich_help_panel="Input",
    ),
    cases: Optional[Path] = typer.Option(
        None,
        "--cases",
        exists=True,
        readable=True,
        help="Optional benchmark YAML file.",
        rich_help_panel="Input",
    ),
    repo_list: Optional[Path] = typer.Option(
        None,
        "--repo-list",
        exists=True,
        readable=True,
        help="Analyze all collections listed in repositories.yaml.",
        rich_help_panel="Input",
    ),
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        help="Analyze a specific Qdrant collection.",
        rich_help_panel="Input",
    ),
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Apply safe config/index improvements.",
        rich_help_panel="Repair",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Confirm safe apply actions.",
        rich_help_panel="Repair",
    ),
    output_format: CollectionOutputFormat = typer.Option(
        CollectionOutputFormat.RICH,
        "--format",
        help="Output format.",
        rich_help_panel="Output",
    ),
    no_banner: bool = typer.Option(
        False,
        "--no-banner",
        help="Hide decorative terminal banner.",
        rich_help_panel="Output",
    ),
) -> None:
    """Suggest or apply safe retrieval/index quality improvements."""
    try:
        cases_path = (
            resolve_benchmark_cases(cases, config)
            if cases is not None
            else _optional_default_benchmark_cases(config)
        )
        report = run_improve(
            config_path=str(config),
            cases_path=str(cases_path) if cases_path else None,
            repo_list=str(repo_list) if repo_list else None,
            collection=collection,
            apply=apply,
            yes=yes,
        )
    except (LookupError, ValueError, SystemExit) as exc:
        console.print(
            f"[bold red]Improve failed:[/bold red] {_redacted_exception(exc, config)}"
        )
        raise typer.Exit(1) from exc

    if output_format == CollectionOutputFormat.JSON:
        print(json.dumps(improve_to_dict(report), indent=2, ensure_ascii=False))
    else:
        _print_startup_screen("Safe improvements", no_banner=no_banner)
        console.print(improve_table(report))
        if report.backup_path:
            console.print(f"[green]Backup created:[/green] {report.backup_path}")

    raise typer.Exit(0)


def _optional_default_benchmark_cases(config: Path) -> Optional[Path]:
    """Return default benchmark cases if present, otherwise None."""
    try:
        return resolve_benchmark_cases(None, config)
    except ValueError:
        return None


def _render_interactive_menu(
    limit: Optional[int],
    with_parent_window: bool,
    scope_label: str,
) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column("Key", style="bold cyan", no_wrap=True)
    table.add_column("Action")
    table.add_row("1", "Ask a question")
    table.add_row("2", "Search matching snippets")
    table.add_row("3", "Ingest repository into Qdrant")
    table.add_row("4", "List collections")
    table.add_row("5", f"Change collection scope ({scope_label})")
    table.add_row(
        "6",
        f"Toggle parent-window context ({'on' if with_parent_window else 'off'})",
    )
    table.add_row("7", f"Change result limit ({limit or 'config default'})")
    table.add_row("8", "Validate config")
    table.add_row("9", "Exit")
    console.print(
        Panel(
            table,
            title=" Interactive Menu ",
            border_style="cyan",
            box=box.ROUNDED,
            safe_box=True,
            width=_terminal_width(),
            padding=(1, 2),
        )
    )


def _run_interactive_ingest(config: Path) -> None:
    """Prompt for ingestion options and run the shared ingest runner."""
    mode = _prompt_choice(
        "Ingest mode",
        ["config", "repo-url", "repo-list"],
        "config",
    )
    repo_url = None
    repo_list = None

    if mode == "repo-url":
        repo_url = str(typer.prompt("Repository URL override")).strip()
        if not repo_url:
            console.print("[yellow]Repository URL cannot be empty.[/yellow]")
            return
    elif mode == "repo-list":
        repo_list_path = str(typer.prompt("Repository list YAML path")).strip()
        if not repo_list_path:
            console.print("[yellow]Repository list path cannot be empty.[/yellow]")
            return
        repo_list_file = Path(repo_list_path)
        if not repo_list_file.exists() or not repo_list_file.is_file():
            console.print(f"[red]Repository list not found:[/red] {repo_list_file}")
            return
        repo_list = str(repo_list_file)

    console.print(
        Panel(
            "Starting ingestion. This can take a while for larger repositories.",
            title=" Repository Ingestion ",
            border_style="cyan",
            box=box.ROUNDED,
            safe_box=True,
            width=_terminal_width(),
            padding=(1, 2),
        )
    )
    exit_code = run_ingest(
        config_path=str(config),
        repo_url=repo_url,
        repo_list=repo_list,
    )
    if exit_code == 0:
        console.print("[green]Ingestion completed successfully.[/green]")
    else:
        console.print(f"[red]Ingestion failed with exit code {exit_code}.[/red]")


def _run_interactive_session(
    config: Path,
    limit: Optional[int],
    with_parent_window: bool,
    no_banner: bool,
    collection: Optional[str] = None,
    repo_list: Optional[Path] = None,
) -> None:
    """Run the reusable interactive prompt loop."""
    _print_startup_screen("Interactive session", no_banner=no_banner)
    current_limit = limit
    current_parent_window = with_parent_window
    current_collection = collection

    while True:
        scope_label = _collection_scope_label(config, current_collection, repo_list)
        _render_interactive_menu(current_limit, current_parent_window, scope_label)
        choice = str(typer.prompt("Select an option", default="1")).strip()

        if choice == "1":
            question = str(typer.prompt("Question")).strip()
            if not question:
                console.print("[yellow]Question cannot be empty.[/yellow]")
                continue

            try:
                with _phase_status(
                    "Encoding question",
                    enabled=_supports_decorative_output(no_banner),
                ) as status:
                    answer_response = generate_answer(
                        config_path=str(config),
                        question=question,
                        limit=current_limit,
                        with_parent_window=current_parent_window,
                        quiet=True,
                        collection=current_collection,
                        repo_list=str(repo_list) if repo_list else None,
                        progress=lambda step: status.update(step),
                    )
                _render_rich_answer_response(
                    answer_response, show_sources=True, phases=status.completed
                )
            except (LookupError, ValueError, RuntimeError, SystemExit) as exc:
                console.print(
                    f"[bold red]Ask failed:[/bold red] {_redacted_exception(exc, config)}"
                )

        elif choice == "2":
            query_text = str(typer.prompt("Search query")).strip()
            if not query_text:
                console.print("[yellow]Search query cannot be empty.[/yellow]")
                continue

            try:
                with _phase_status(
                    "Encoding query",
                    enabled=_supports_decorative_output(no_banner),
                ) as status:
                    query_response = execute_query(
                        config_path=str(config),
                        query=query_text,
                        limit=current_limit,
                        with_parent_window=current_parent_window,
                        quiet=True,
                        collection=current_collection,
                        repo_list=str(repo_list) if repo_list else None,
                        progress=lambda step: status.update(step),
                    )
                _render_rich_query_response(query_response, phases=status.completed)
            except (LookupError, ValueError, SystemExit) as exc:
                console.print(
                    f"[bold red]Query failed:[/bold red] {_redacted_exception(exc, config)}"
                )

        elif choice == "3":
            _run_interactive_ingest(config)

        elif choice == "4":
            try:
                targets = _resolve_cli_collection_targets(
                    config,
                    collection=current_collection,
                    repo_list=repo_list,
                )
                _render_collection_targets(targets)
            except (FileNotFoundError, ValueError) as exc:
                console.print(
                    "[bold red]Collection list failed:[/bold red] "
                    f"{_redacted_exception(exc, config)}"
                )

        elif choice == "5":
            if repo_list:
                try:
                    targets = load_repository_targets(str(repo_list))
                except (FileNotFoundError, ValueError) as exc:
                    console.print(
                        "[bold red]Collection list failed:[/bold red] "
                        f"{_redacted_exception(exc, config)}"
                    )
                    continue
                choices = ["all"] + [target.collection_name for target in targets]
                selected = _prompt_choice("Collection scope", choices, "all")
                current_collection = None if selected == "all" else selected
            else:
                default_scope = current_collection or _collection_scope_label(
                    config, current_collection, repo_list
                )
                selected = str(
                    typer.prompt("Collection name", default=default_scope)
                ).strip()
                if not selected:
                    console.print("[yellow]Collection name cannot be empty.[/yellow]")
                    continue
                current_collection = selected
            console.print(
                f"[green]Collection scope set to "
                f"{_collection_scope_label(config, current_collection, repo_list)}.[/green]"
            )

        elif choice == "6":
            current_parent_window = not current_parent_window
            state = "enabled" if current_parent_window else "disabled"
            console.print(f"[green]Parent-window context {state}.[/green]")

        elif choice == "7":
            current_limit = _prompt_int("Result limit", current_limit or 10)
            console.print(f"[green]Result limit set to {current_limit}.[/green]")

        elif choice == "8":
            loaded = _load_runtime_config(str(config), quiet=True)
            report = validate_config_data(loaded)
            _render_validation_report(report)

        elif choice == "9":
            console.print("[cyan]Goodbye.[/cyan]")
            break

        else:
            console.print("[red]Choose a number from 1 to 9.[/red]")


@app.command()
def interactive(
    config: Optional[Path] = typer.Argument(
        None,
        help="Path to ingestion config. Defaults to config.yaml/config.yml/config.json.",
        rich_help_panel="Input",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        help="Initial retrieval limit.",
        rich_help_panel="Retrieval",
    ),
    with_parent_window: bool = typer.Option(
        False,
        "--with-parent-window",
        help="Start with expanded context enabled.",
        rich_help_panel="Retrieval",
    ),
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        help="Start with a specific Qdrant collection selected.",
        rich_help_panel="Retrieval",
    ),
    repo_list: Optional[Path] = typer.Option(
        None,
        "--repo-list",
        exists=True,
        readable=True,
        help="Use repositories.yaml as the interactive collection list.",
        rich_help_panel="Retrieval",
    ),
    no_banner: bool = typer.Option(
        False,
        "--no-banner",
        help="Hide decorative terminal banner.",
        rich_help_panel="Output",
    ),
    classic: bool = typer.Option(
        False,
        "--classic",
        "--no-tui",
        help="Use the classic prompt-loop menu instead of the Textual UI.",
        rich_help_panel="Output",
    ),
) -> None:
    """Open the terminal UI for asking and searching repeatedly."""
    try:
        resolved_config = _resolve_default_config(config)
    except ValueError as exc:
        if not classic and _can_run_tui():
            _run_textual_interactive(
                config=_initial_config_target(config),
                limit=limit,
                with_parent_window=with_parent_window,
                collection=collection,
                repo_list=repo_list,
                first_run_setup=True,
            )
            return
        console.print(f"[bold red]Interactive startup failed:[/bold red] {exc}")
        raise typer.Exit(1) from exc

    if not classic and _can_run_tui():
        _run_textual_interactive(
            config=resolved_config,
            limit=limit,
            with_parent_window=with_parent_window,
            collection=collection,
            repo_list=repo_list,
            first_run_setup=False,
        )
        return

    _run_interactive_session(
        config=resolved_config,
        limit=limit,
        with_parent_window=with_parent_window,
        no_banner=no_banner,
        collection=collection,
        repo_list=repo_list,
    )


@app.command("validate-config")
def validate_config(
    config: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to YAML or JSON config.",
        rich_help_panel="Input",
    ),
    check_connections: bool = typer.Option(
        False,
        "--check-connections",
        help="Also initialize embedding and Qdrant clients.",
        rich_help_panel="Validation",
    ),
    no_banner: bool = typer.Option(
        False,
        "--no-banner",
        help="Hide decorative terminal banner.",
        rich_help_panel="Output",
    ),
) -> None:
    """Validate config structure, optionally checking live connections."""
    _print_startup_screen("Config validation", no_banner=no_banner)
    loaded = ConfigLoader.load_config(str(config))
    report = validate_config_data(loaded)
    _render_validation_report(report)

    if not report.ok:
        raise typer.Exit(1)

    if check_connections:
        unresolved = _find_unresolved_placeholders(loaded)
        if unresolved:
            console.print("[red]Cannot check connections with unresolved values:[/red]")
            for path in unresolved:
                console.print(f"  - {path}")
            raise typer.Exit(1)

        try:
            GitHubToQdrantProcessor(str(config))
        except Exception as exc:
            console.print(
                f"[red]Connection check failed:[/red] {_redacted_exception(exc, config)}"
            )
            raise typer.Exit(1) from exc

        console.print("[green]Connection check succeeded.[/green]")


@app.command()
def wizard(
    output: Path = typer.Option(
        Path("config.yaml"),
        "--output",
        "-o",
        help="Where to write the generated YAML config.",
        rich_help_panel="Output",
    ),
    run_after: bool = typer.Option(
        False,
        "--run/--no-run",
        help="Run ingestion after writing and validating the config.",
        rich_help_panel="Execution",
    ),
    no_banner: bool = typer.Option(
        False,
        "--no-banner",
        help="Hide decorative terminal banner.",
        rich_help_panel="Output",
    ),
) -> None:
    """Interactively create a config file, then optionally run ingestion."""
    _print_startup_screen("Config wizard", no_banner=no_banner)

    if output.exists():
        overwrite = typer.confirm(f"{output} exists. Overwrite it?", default=False)
        if not overwrite:
            alternative = typer.prompt(
                "Choose a different config filename", default="config.local.yaml"
            )
            output = Path(alternative)
            if output.exists():
                overwrite_alternative = typer.confirm(
                    f"{output} also exists. Overwrite it?", default=False
                )
                if not overwrite_alternative:
                    console.print(
                        "[yellow]Wizard cancelled without writing a config.[/yellow]"
                    )
                    raise typer.Exit(1)

    config = _collect_wizard_config()
    report = validate_config_data(config)
    _render_validation_report(report)
    if not report.ok:
        raise typer.Exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    console.print(f"[green]Config written:[/green] {output}")

    if run_after:
        exit_code = run_ingest(config_path=str(output))
        raise typer.Exit(exit_code)


def main() -> None:
    """Console script entrypoint."""
    app()


if __name__ == "__main__":
    main()
