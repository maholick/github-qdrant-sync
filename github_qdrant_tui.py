#!/usr/bin/env python3
"""Textual terminal UI for GitHub Qdrant Sync."""

from __future__ import annotations

import shlex
import shutil
import tempfile
import time
import re
import json
import os
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from rich import box
from rich.panel import Panel
from rich.table import Table
from ruamel.yaml import YAML
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Input, RichLog, Static

from github_qdrant_cli import (  # pylint: disable=protected-access
    AnswerResponse,
    WizardConfigAnswers,
    _answering_config,
    _app_version,
    _clip,
    _default_config,
    generate_answer,
    validate_config_data,
)
from github_qdrant_quality import (
    benchmark_table,
    doctor_table,
    improve_table,
    resolve_benchmark_cases,
    run_benchmark,
    run_doctor,
    run_improve,
)
from github_to_qdrant import run_ingest
from rag_retrieval import (
    CollectionCompatibility,
    CollectionTarget,
    QueryResponse,
    execute_query,
    inspect_collection_targets,
    resolve_collection_targets,
)


PROJECT_TITLE = "GithubQdrant-Sync"
PROJECT_REPO_LABEL = "maholick/github-qdrant-sync"
SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


@dataclass(frozen=True)
class SlashCommandSpec:
    """Display metadata for one TUI slash command."""

    category: str
    command: str
    description: str
    example: str = ""
    aliases: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


COMMAND_CATEGORIES = (
    "Search & Ask",
    "Collections",
    "Config",
    "Quality",
    "Ingest",
    "Session",
)
SETUP_PROVIDER_DEFAULTS = {
    "mistral_ai": ("codestral-embed", 3072, "MISTRAL_API_KEY"),
    "azure_openai": ("text-embedding-3-large", 3072, "AZURE_OPENAI_API_KEY"),
    "sentence_transformers": (
        "intfloat/multilingual-e5-large",
        1024,
        "",
    ),
}
SLASH_COMMANDS = (
    SlashCommandSpec(
        "Search & Ask",
        "/ask <question>",
        "Generate an AI answer with sources.",
        "/ask How do I configure authentication?",
        tags=("answer", "chat"),
    ),
    SlashCommandSpec(
        "Search & Ask",
        "/search <query>",
        "Show matching snippets from Qdrant.",
        "/search auth middleware",
        tags=("query", "retrieve"),
    ),
    SlashCommandSpec(
        "Search & Ask",
        "/limit <n>",
        "Change result count.",
        "/limit 5",
    ),
    SlashCommandSpec(
        "Search & Ask",
        "/parent on|off",
        "Toggle expanded parent-window context.",
        "/parent on",
    ),
    SlashCommandSpec(
        "Collections",
        "/collections",
        "Show configured collections.",
        "/collections",
    ),
    SlashCommandSpec(
        "Collections",
        "/scope all",
        "Use all repo-list collections or config default.",
        "/scope all",
    ),
    SlashCommandSpec(
        "Collections",
        "/scope <collection>",
        "Search one collection.",
        "/scope veviad_app",
    ),
    SlashCommandSpec(
        "Collections",
        "/repo-list <path>",
        "Load a repository list.",
        "/repo-list repositories.yaml",
    ),
    SlashCommandSpec(
        "Collections",
        "/repo-list clear",
        "Clear the repository list.",
        "/repo-list clear",
    ),
    SlashCommandSpec(
        "Config",
        "/config",
        "Show editable config summary.",
        "/config",
    ),
    SlashCommandSpec(
        "Config",
        "/wizard",
        "Create an initial config in the TUI.",
        "/wizard",
        tags=("setup", "first-run"),
    ),
    SlashCommandSpec(
        "Config",
        "/config <path>",
        "Load a different config file.",
        "/config config.local.yaml",
        aliases=("/load-config <path>",),
    ),
    SlashCommandSpec(
        "Config",
        "/get <path>",
        "Show one config value.",
        "/get qdrant.collection_name",
    ),
    SlashCommandSpec(
        "Config",
        "/set <path> <value>",
        "Stage a non-secret config value.",
        "/set retrieval.top_k 5",
    ),
    SlashCommandSpec(
        "Config",
        "/secret <path> <ENV_VAR>",
        "Stage a secret as an env placeholder.",
        "/secret qdrant.api_key QDRANT_API_KEY",
    ),
    SlashCommandSpec(
        "Config",
        "/changes",
        "Show unsaved config changes.",
        "/changes",
    ),
    SlashCommandSpec(
        "Config",
        "/save-config",
        "Preview save with diff and validation.",
        "/save-config",
    ),
    SlashCommandSpec(
        "Config",
        "/save-config --confirm",
        "Write changes and create a backup.",
        "/save-config --confirm",
    ),
    SlashCommandSpec(
        "Config",
        "/save-config-as <path>",
        "Preview saving to a new config path.",
        "/save-config-as config.local.yaml",
    ),
    SlashCommandSpec(
        "Config",
        "/discard-config-changes",
        "Reset unsaved changes.",
        "/discard-config-changes",
    ),
    SlashCommandSpec(
        "Quality",
        "/doctor",
        "Check Qdrant collection and index health.",
        "/doctor",
        tags=("health", "index"),
    ),
    SlashCommandSpec(
        "Quality",
        "/benchmark [eval.yaml]",
        "Run retrieval quality checks.",
        "/benchmark",
        tags=("eval", "score"),
    ),
    SlashCommandSpec(
        "Quality",
        "/improve [eval.yaml]",
        "Suggest safe retrieval/index improvements.",
        "/improve",
        tags=("repair", "tune"),
    ),
    SlashCommandSpec(
        "Quality",
        "/improve [eval.yaml] --apply",
        "Apply safe improvements after review.",
        "/improve --apply",
    ),
    SlashCommandSpec(
        "Quality",
        "/validate",
        "Validate the active config.",
        "/validate",
    ),
    SlashCommandSpec(
        "Ingest",
        "/ingest config",
        "Ingest the active config repository.",
        "/ingest config",
    ),
    SlashCommandSpec(
        "Ingest",
        "/ingest repo-url <url>",
        "Ingest a repository URL override.",
        "/ingest repo-url https://github.com/example/project.git",
    ),
    SlashCommandSpec(
        "Ingest",
        "/ingest repo-list <path>",
        "Ingest repositories from a list.",
        "/ingest repo-list repositories.yaml",
    ),
    SlashCommandSpec("Session", "/clear", "Clear output.", "/clear"),
    SlashCommandSpec(
        "Session",
        "/help <category>",
        "Show command groups or a category page.",
        "/help config",
    ),
    SlashCommandSpec("Session", "/quit", "Exit.", "/quit", aliases=("/exit", "/q")),
)
SECRET_PATHS = {
    "github.token",
    "mistral_ai.api_key",
    "azure_openai.api_key",
    "qdrant.api_key",
}
INDEX_AFFECTING_PATHS = {
    "embedding_provider",
    "mistral_ai.dimensions",
    "azure_openai.dimensions",
    "sentence_transformers.dimensions",
    "qdrant.collection_name",
    "qdrant.vector_size",
    "qdrant.distance",
    "qdrant.vector_name",
    "qdrant.quantization.enabled",
    "qdrant.quantization.method",
    "qdrant.quantization.bits",
    "qdrant.quantization.always_ram",
    "qdrant.quantization.apply_to_existing_collections",
}
CONFIG_PATH_TYPES = {
    "github.repository_url": ("string", None),
    "github.branch": ("string", None),
    "github.name": ("string", None),
    "github.token": ("secret", None),
    "embedding_provider": (
        "enum",
        {"azure_openai", "mistral_ai", "sentence_transformers"},
    ),
    "mistral_ai.model": ("string", None),
    "mistral_ai.dimensions": ("int", None),
    "mistral_ai.api_key": ("secret", None),
    "azure_openai.endpoint": ("string", None),
    "azure_openai.model": ("string", None),
    "azure_openai.api_version": ("string", None),
    "azure_openai.dimensions": ("int", None),
    "azure_openai.api_key": ("secret", None),
    "sentence_transformers.model": ("string", None),
    "sentence_transformers.dimensions": ("int", None),
    "answering.provider": ("enum", {"azure_openai", "mistral_ai"}),
    "answering.model": ("string", None),
    "answering.temperature": ("float", None),
    "answering.max_context_chars": ("int", None),
    "answering.system_prompt": ("string", None),
    "qdrant.url": ("string", None),
    "qdrant.collection_name": ("string", None),
    "qdrant.vector_size": ("int", None),
    "qdrant.distance": ("enum", {"Cosine", "Euclidean", "Dot"}),
    "qdrant.vector_name": ("optional_string", None),
    "qdrant.timeout": ("int", None),
    "qdrant.api_key": ("secret", None),
    "qdrant.quantization.enabled": ("bool", None),
    "qdrant.quantization.method": ("enum", {"turbo"}),
    "qdrant.quantization.bits": (
        "enum",
        {"bits1", "bits1_5", "bits2", "bits4"},
    ),
    "qdrant.quantization.always_ram": ("bool", None),
    "qdrant.quantization.apply_to_existing_collections": ("bool", None),
    "qdrant.quantization.search.ignore": ("bool", None),
    "qdrant.quantization.search.rescore": ("bool", None),
    "qdrant.quantization.search.oversampling": ("float", None),
    "processing.file_mode": ("enum", {"all_text", "markdown_only"}),
    "processing.chunk_size": ("int", None),
    "processing.chunk_overlap": ("int", None),
    "retrieval.top_k": ("int", None),
    "retrieval.fetch_k": ("int", None),
    "retrieval.max_chunks_per_file": ("int", None),
    "retrieval.parent_window": ("int", None),
    "pdf_processing.enabled": ("bool", None),
    "pdf_processing.mode": ("enum", {"local", "cloud", "hybrid"}),
}
CONFIG_SUMMARY_GROUPS = {
    "GitHub": ["github.repository_url", "github.branch", "github.name", "github.token"],
    "Providers": [
        "embedding_provider",
        "mistral_ai.model",
        "mistral_ai.dimensions",
        "azure_openai.model",
        "azure_openai.dimensions",
        "sentence_transformers.model",
        "sentence_transformers.dimensions",
    ],
    "Answering": [
        "answering.provider",
        "answering.model",
        "answering.temperature",
        "answering.max_context_chars",
    ],
    "Qdrant": [
        "qdrant.url",
        "qdrant.collection_name",
        "qdrant.vector_size",
        "qdrant.distance",
        "qdrant.vector_name",
        "qdrant.timeout",
        "qdrant.api_key",
        "qdrant.quantization.enabled",
        "qdrant.quantization.method",
        "qdrant.quantization.bits",
        "qdrant.quantization.always_ram",
        "qdrant.quantization.apply_to_existing_collections",
        "qdrant.quantization.search.ignore",
        "qdrant.quantization.search.rescore",
        "qdrant.quantization.search.oversampling",
    ],
    "Processing": [
        "processing.file_mode",
        "processing.chunk_size",
        "processing.chunk_overlap",
    ],
    "Retrieval": [
        "retrieval.top_k",
        "retrieval.fetch_k",
        "retrieval.max_chunks_per_file",
        "retrieval.parent_window",
    ],
    "PDF": ["pdf_processing.enabled", "pdf_processing.mode"],
}
yaml_rt = YAML()
yaml_rt.preserve_quotes = True


@dataclass(frozen=True)
class TuiCommand:
    """Parsed command submitted in the Textual command input."""

    name: str
    argument: str = ""


@dataclass(frozen=True)
class SetupStep:
    """One first-run setup wizard prompt."""

    key: str
    prompt: str
    default: str = ""
    value_type: str = "string"
    choices: tuple[str, ...] = ()
    required: bool = True


def parse_tui_command(raw_command: str) -> TuiCommand:
    """Parse a natural command line into an internal TUI command."""
    command = raw_command.strip()
    if command.startswith("/"):
        command = command[1:].lstrip()
    if not command:
        return TuiCommand("help")

    lowered = command.lower()
    exact_commands = {
        "?": "help",
        "help": "help",
        "clear": "clear",
        "collections": "collections",
        "config": "config",
        "wizard": "wizard",
        "setup": "wizard",
        "load-config": "config",
        "list collections": "collections",
        "repo-list": "repo-list",
        "get": "get",
        "set": "set",
        "secret": "secret",
        "changes": "changes",
        "save-config": "save-config",
        "save-config-as": "save-config-as",
        "discard-config-changes": "discard-config-changes",
        "doctor": "doctor",
        "benchmark": "benchmark",
        "improve": "improve",
        "validate": "validate",
        "validate config": "validate",
        "quit": "quit",
        "exit": "quit",
        "q": "quit",
        "ingest": "ingest",
    }
    if lowered in exact_commands:
        return TuiCommand(exact_commands[lowered])

    command_aliases = {
        "ask": "ask",
        "search": "search",
        "scope": "scope",
        "limit": "limit",
        "parent": "parent",
        "ingest": "ingest",
        "help": "help",
        "config": "config",
        "wizard": "wizard",
        "setup": "wizard",
        "load-config": "config",
        "repo-list": "repo-list",
        "get": "get",
        "set": "set",
        "secret": "secret",
        "save-config": "save-config",
        "save-config-as": "save-config-as",
        "benchmark": "benchmark",
        "doctor": "doctor",
        "improve": "improve",
    }
    for name, parsed_name in command_aliases.items():
        prefix = f"{name} "
        if lowered.startswith(prefix):
            return TuiCommand(parsed_name, command[len(prefix) :].strip())

    return TuiCommand("unknown", command)


def _split_args(value: str) -> List[str]:
    """Split command arguments using shell-style quotes."""
    return shlex.split(value)


def _load_yaml_config(path: Path) -> Any:
    """Load YAML while preserving comments and order."""
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as config_file:
            return json.load(config_file)
    with path.open("r", encoding="utf-8") as config_file:
        return yaml_rt.load(config_file) or {}


def _write_yaml_config(path: Path, config: Any) -> None:
    """Write YAML while preserving comments and order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        with path.open("w", encoding="utf-8") as config_file:
            json.dump(_plain_config(config), config_file, indent=2, ensure_ascii=False)
            config_file.write("\n")
        return
    with path.open("w", encoding="utf-8") as config_file:
        yaml_rt.dump(config, config_file)


def _plain_config(value: Any) -> Any:
    """Convert ruamel containers to plain Python data for JSON serialization."""
    if isinstance(value, dict):
        return {key: _plain_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_config(item) for item in value]
    return value


def _get_config_path(config: Any, dotted_path: str) -> Any:
    """Return a nested config value by dotted path."""
    current = config
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_path)
        current = current[part]
    return current


def _set_config_path(config: Any, dotted_path: str, value: Any) -> None:
    """Set a nested config value by dotted path."""
    parts = dotted_path.split(".")
    current = config
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _redact_config_value(path: str, value: Any) -> str:
    """Return a display-safe config value."""
    if path in SECRET_PATHS:
        if value in {None, ""}:
            return "<empty>"
        return "<secret env placeholder>"
    if value is None:
        return "null"
    return str(value)


def _collect_secret_values(config: Any) -> List[str]:
    """Collect concrete secret values that should never be displayed."""
    values: List[str] = []
    for path in SECRET_PATHS:
        try:
            value = _get_config_path(config, path)
        except KeyError:
            continue
        if not isinstance(value, str) or not value or value.startswith("${"):
            continue
        values.append(value)
    return values


def _redact_text(text: str, config: Any) -> str:
    """Redact known config secrets from arbitrary text."""
    redacted = str(text)
    for value in _collect_secret_values(config):
        redacted = redacted.replace(value, "<redacted>")
    redacted = re.sub(
        r"https://[^\s/@]+@github\.com",
        "https://<redacted>@github.com",
        redacted,
    )
    return redacted


def _format_change(path: str, old: Any, new: Any) -> str:
    """Return a one-line display of a config change."""
    warning = (
        " [yellow](index-affecting)[/yellow]" if path in INDEX_AFFECTING_PATHS else ""
    )
    return (
        f"[cyan]{path}[/cyan]: "
        f"[dim]{_redact_config_value(path, old)}[/dim] -> "
        f"[bold]{_redact_config_value(path, new)}[/bold]{warning}"
    )


def _coerce_config_value(path: str, raw_value: str) -> Any:
    """Coerce a user-entered value according to the editable path spec."""
    if path not in CONFIG_PATH_TYPES:
        raise ValueError(f"Config path is not editable: {path}")
    value_type, choices = CONFIG_PATH_TYPES[path]
    if value_type == "secret":
        raise ValueError(f"{path} is secret. Use `/secret {path} ENV_VAR`.")

    normalized = raw_value.strip()
    if value_type == "string":
        if not normalized:
            raise ValueError(f"{path} cannot be blank")
        return normalized
    if value_type == "optional_string":
        if normalized.lower() in {"null", "none", ""}:
            return None
        return normalized
    if value_type == "int":
        return int(normalized)
    if value_type == "float":
        return float(normalized)
    if value_type == "bool":
        if normalized.lower() in {"true", "yes", "y", "1", "on"}:
            return True
        if normalized.lower() in {"false", "no", "n", "0", "off"}:
            return False
        raise ValueError(f"{path} must be true or false")
    if value_type == "enum":
        assert choices is not None
        if normalized not in choices:
            allowed = ", ".join(sorted(choices))
            raise ValueError(f"{path} must be one of: {allowed}")
        return normalized
    raise ValueError(f"Unsupported config value type for {path}: {value_type}")


def _secret_placeholder(env_name: str) -> str:
    """Return an env placeholder for a secret variable name."""
    normalized = env_name.strip()
    if normalized.startswith("${") and normalized.endswith("}"):
        normalized = normalized[2:-1]
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", normalized):
        raise ValueError("Secret environment variable must look like ENV_VAR_NAME")
    return "${" + normalized + "}"


class GithubQdrantSyncApp(App[None]):
    """Fixed-area terminal UI for repeated repository search and answers."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #status {
        height: 5;
        border: round $primary;
        padding: 0 1;
    }

    #body {
        height: 1fr;
    }

    #content-column {
        width: 2fr;
        min-width: 45;
    }

    #side-column {
        width: 1fr;
        min-width: 34;
    }

    #main {
        height: 2fr;
        border: round $primary;
        padding: 0 1;
    }

    #activity {
        height: 8;
        border: round $accent;
        padding: 0 1;
    }

    #sources {
        height: 2fr;
        border: round $primary;
    }

    #collections {
        height: 1fr;
        border: round $primary;
    }

    #command {
        height: 3;
        border: round $primary;
        padding: 0 2;
        background: $surface;
    }

    #command:focus {
        border: round $accent;
        background-tint: $foreground 4%;
    }

    #command.-invalid {
        border: round $error 60%;
    }

    #command.-invalid:focus {
        border: round $error;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("tab", "focus_next", "Next pane"),
        Binding("?", "help", "Help"),
    ]

    def __init__(
        self,
        config: Path,
        limit: Optional[int] = None,
        with_parent_window: bool = False,
        collection: Optional[str] = None,
        repo_list: Optional[Path] = None,
        first_run_setup: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.config_source_path = config
        self.limit = limit
        self.with_parent_window = with_parent_window
        self.collection = collection
        self.repo_list = repo_list
        self.first_run_setup = first_run_setup
        self.original_config: Any = {}
        self.working_config: Any = {}
        self.dirty_paths: set[str] = set()
        self.setup_active = first_run_setup
        self.setup_step_index = 0
        self.setup_answers: Dict[str, Any] = {}
        self.loaded_config: Dict[str, Any] = {}
        self.collection_targets: List[CollectionTarget] = []
        self.collection_statuses: List[CollectionCompatibility] = []
        self.last_error: str = ""
        self.active_worker: Any = None
        self.activity_timer: Any = None
        self.operation_running = False
        self.operation_title = ""
        self.operation_step = ""
        self.operation_detail = ""
        self.operation_phases: List[str] = []
        self.operation_started_at = 0.0
        self.operation_frame = 0

    def compose(self) -> ComposeResult:
        """Compose fixed panes for status, output, sources, collections, and input."""
        yield Header(show_clock=False)
        yield Static(id="status")
        with Horizontal(id="body"):
            with Vertical(id="content-column"):
                yield RichLog(id="main", wrap=True, highlight=True, markup=True)
                yield RichLog(id="activity", wrap=True, highlight=True, markup=True)
            with Vertical(id="side-column"):
                yield DataTable(id="sources", cursor_type="row")
                yield DataTable(id="collections", cursor_type="row")
        yield Input(
            placeholder="/ask How do I configure authentication?",
            id="command",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Initialize panes after Textual mounts the widget tree."""
        self.title = PROJECT_TITLE
        self.sub_title = str(self.config)
        self._load_context()
        self._reset_sources_table()
        self._refresh_collections_table()
        self._refresh_status()
        if self.setup_active and not self.config.exists():
            self._render_setup_step()
        else:
            self.setup_active = False
            self._render_help()
            self._set_activity(["Ready. Type `help` for commands."])
        self.activity_timer = self.set_interval(
            0.12,
            self._tick_activity,
            name="activity-spinner",
            pause=True,
        )
        self.query_one("#command", Input).focus()

    def action_clear(self) -> None:
        """Clear output panes without changing session state."""
        self.query_one("#main", RichLog).clear()
        self._reset_sources_table()
        self._set_activity(["Cleared."])

    def action_help(self) -> None:
        """Show command help in the output pane."""
        self._render_help()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle a submitted command from the bottom input."""
        command = event.value.strip()
        event.input.value = ""
        self.handle_command(command)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Show a lightweight slash-command palette while typing commands."""
        if self.operation_running:
            return
        value = event.value.strip()
        if value == "/" or (value.startswith("/") and " " not in value):
            self._render_command_palette(value[1:].lower())

    def handle_command(self, raw_command: str) -> None:
        """Execute a parsed TUI command."""
        if self.setup_active:
            self._handle_setup_input(raw_command)
            return

        command = parse_tui_command(raw_command)
        if command.name == "help":
            self._render_command_palette(command.argument)
        elif command.name == "clear":
            self.action_clear()
        elif command.name == "quit":
            self.exit()
        elif command.name == "ask":
            self._run_ask(command.argument)
        elif command.name == "search":
            self._run_search(command.argument)
        elif command.name == "collections":
            self._show_collections()
        elif command.name == "scope":
            self._change_scope(command.argument)
        elif command.name == "config":
            self._load_config_command(command.argument)
        elif command.name == "wizard":
            self._start_setup_wizard()
        elif command.name == "repo-list":
            self._load_repo_list_command(command.argument)
        elif command.name == "get":
            self._get_config_command(command.argument)
        elif command.name == "set":
            self._set_config_command(command.argument)
        elif command.name == "secret":
            self._secret_config_command(command.argument)
        elif command.name == "changes":
            self._render_config_changes()
        elif command.name == "save-config":
            self._save_config_command(command.argument)
        elif command.name == "save-config-as":
            self._save_config_as_command(command.argument)
        elif command.name == "discard-config-changes":
            self._discard_config_changes()
        elif command.name == "doctor":
            self._run_doctor_command(command.argument)
        elif command.name == "benchmark":
            self._run_benchmark_command(command.argument)
        elif command.name == "improve":
            self._run_improve_command(command.argument)
        elif command.name == "limit":
            self._change_limit(command.argument)
        elif command.name == "parent":
            self._change_parent_window(command.argument)
        elif command.name == "validate":
            self._run_validation()
        elif command.name == "ingest":
            self._run_ingest_command(command.argument)
        else:
            self._render_error(
                "Unknown command",
                (
                    f"`{command.argument}` is not a command. Type `help` "
                    "or use slash commands like `/ask ...`."
                ),
            )

    def _load_context(self) -> None:
        """Load config and collection metadata for status/sidebar rendering."""
        self.last_error = ""
        if not self.working_config:
            try:
                self._load_config_state(self.config)
            except (OSError, ValueError) as exc:
                self.working_config = {}
                self.loaded_config = {}
                self.last_error = str(exc)
                return

        self.loaded_config = self.working_config
        try:
            self.collection_targets = resolve_collection_targets(
                self.loaded_config,
                collection=self.collection,
                repo_list=str(self.repo_list) if self.repo_list else None,
            )
            names = {target.collection_name for target in self.collection_targets}
            self.collection_statuses = [
                status
                for status in self.collection_statuses
                if status.collection_name in names
            ]
        except (FileNotFoundError, ValueError) as exc:
            self.collection_targets = []
            self.collection_statuses = []
            self.last_error = str(exc)

    def _load_config_state(self, config_path: Path) -> None:
        """Load a config file into original and working editor state."""
        loaded = _load_yaml_config(config_path)
        self.config = config_path
        self.config_source_path = config_path
        self.original_config = deepcopy(loaded)
        self.working_config = deepcopy(loaded)
        self.loaded_config = self.working_config
        self.dirty_paths.clear()
        self.collection_statuses.clear()

    def _start_setup_wizard(self) -> None:
        """Start the lightweight TUI setup wizard."""
        if self.config.exists():
            self._render_error(
                "Config already exists",
                (
                    f"{self.config} already exists. Use `/config` to edit it, "
                    "`/save-config-as PATH` for a copy, or run the CLI wizard "
                    "with `--output PATH`."
                ),
            )
            return
        self.setup_active = True
        self.setup_step_index = 0
        self.setup_answers = {}
        self._reset_sources_table()
        self._refresh_collections_table()
        self._refresh_status()
        self._render_setup_step()

    def _setup_repo_slug(self) -> str:
        """Return a config-friendly collection/name default from the repository URL."""
        url = str(self.setup_answers.get("repository_url") or "").strip().rstrip("/")
        slug = url.split("/")[-1] if url else "github-docs"
        if slug.endswith(".git"):
            slug = slug[:-4]
        slug = re.sub(r"[^A-Za-z0-9_]+", "_", slug).strip("_").lower()
        return slug or "github_docs"

    def _setup_steps(self) -> List[SetupStep]:
        """Return setup wizard steps, including provider-specific prompts."""
        provider = str(self.setup_answers.get("embedding_provider") or "mistral_ai")
        model, dimensions, secret_env = SETUP_PROVIDER_DEFAULTS.get(
            provider, SETUP_PROVIDER_DEFAULTS["mistral_ai"]
        )
        steps = [
            SetupStep("repository_url", "GitHub repository URL"),
            SetupStep("branch", "Target branch", "main"),
            SetupStep("name", "Repository display name", self._setup_repo_slug()),
            SetupStep(
                "embedding_provider",
                "Embedding provider",
                "mistral_ai",
                choices=("mistral_ai", "azure_openai", "sentence_transformers"),
            ),
        ]
        if provider == "azure_openai":
            steps.append(
                SetupStep(
                    "azure_endpoint",
                    "Azure OpenAI endpoint",
                    "${AZURE_OPENAI_ENDPOINT}",
                )
            )
        steps.extend(
            [
                SetupStep("provider_model", "Embedding model", str(model)),
                SetupStep(
                    "provider_dimensions",
                    "Embedding dimensions",
                    str(dimensions),
                    value_type="int",
                ),
            ]
        )
        if provider != "sentence_transformers":
            steps.append(
                SetupStep(
                    "provider_secret_env",
                    "Embedding API key environment variable",
                    secret_env,
                    value_type="env",
                )
            )
        steps.extend(
            [
                SetupStep("qdrant_url", "Qdrant URL", "${QDRANT_URL}"),
                SetupStep(
                    "qdrant_api_key_env",
                    "Qdrant API key environment variable",
                    "QDRANT_API_KEY",
                    value_type="env",
                ),
                SetupStep(
                    "collection_name",
                    "Qdrant collection name",
                    self._setup_repo_slug(),
                ),
                SetupStep(
                    "file_mode",
                    "File processing mode",
                    "all_text",
                    choices=("all_text", "markdown_only"),
                ),
                SetupStep(
                    "pdf_mode",
                    "PDF processing mode",
                    "local",
                    choices=("local", "cloud", "hybrid", "disabled"),
                ),
                SetupStep(
                    "confirm",
                    f"Write {self.config}?",
                    "yes",
                    value_type="bool",
                    choices=("yes", "no"),
                ),
            ]
        )
        return steps

    def _coerce_setup_value(self, step: SetupStep, raw_value: str) -> Any:
        """Coerce one setup wizard answer."""
        value = raw_value.strip()
        if not value and step.default:
            value = step.default
        if step.required and not value:
            raise ValueError(f"{step.prompt} cannot be blank")

        if step.choices:
            normalized = value.lower()
            if normalized not in step.choices:
                raise ValueError(f"Choose one of: {', '.join(step.choices)}")
            value = normalized

        if step.value_type == "int":
            number = int(value)
            if number <= 0:
                raise ValueError(f"{step.prompt} must be greater than zero")
            return number
        if step.value_type == "bool":
            normalized = value.lower()
            if normalized in {"yes", "y", "true", "1", "on"}:
                return True
            if normalized in {"no", "n", "false", "0", "off"}:
                return False
            raise ValueError("Answer yes or no")
        if step.value_type == "env":
            return _secret_placeholder(value)[2:-1]
        return value

    def _handle_setup_input(self, raw_value: str) -> None:
        """Advance or control the first-run setup wizard."""
        command = raw_value.strip()
        lowered = command.lower()
        if lowered in {"/quit", "quit", "/q", "q", "/exit", "exit"}:
            self.exit()
            return
        if lowered in {"/help", "help", "?"}:
            self._render_setup_step()
            return
        if lowered in {"/cancel", "cancel"}:
            self.setup_active = False
            self._render_error(
                "Setup cancelled",
                "No config was written. Use `/wizard` to start setup again.",
            )
            return
        if lowered in {"/back", "back"}:
            if self.setup_step_index > 0:
                previous_step = self._setup_steps()[self.setup_step_index - 1]
                self.setup_answers.pop(previous_step.key, None)
                self.setup_step_index -= 1
            self._render_setup_step()
            return

        steps = self._setup_steps()
        if self.setup_step_index >= len(steps):
            self._finish_setup_wizard()
            return
        step = steps[self.setup_step_index]
        try:
            answer = self._coerce_setup_value(step, command)
        except (TypeError, ValueError) as exc:
            self._set_activity(
                [f"[red]{exc}[/red]", "Type `back`, `cancel`, or an answer."]
            )
            return

        if step.key == "confirm":
            if not answer:
                self.setup_active = False
                self._render_error(
                    "Setup cancelled",
                    "No config was written. Use `/wizard` to start setup again.",
                )
                return
            self._finish_setup_wizard()
            return

        self.setup_answers[step.key] = answer
        self.setup_step_index += 1
        self._render_setup_step()

    def _finish_setup_wizard(self) -> None:
        """Write the generated first-run config and load it into the TUI."""
        if self.config.exists():
            self.setup_active = False
            self._render_error("Config already exists", str(self.config))
            return

        provider = str(self.setup_answers.get("embedding_provider") or "mistral_ai")
        config_data = _default_config(
            WizardConfigAnswers(
                repository_url=str(self.setup_answers.get("repository_url") or ""),
                branch=str(self.setup_answers.get("branch") or "main"),
                name=str(self.setup_answers.get("name") or ""),
                embedding_provider=provider,
                provider_model=str(self.setup_answers.get("provider_model") or ""),
                provider_dimensions=int(
                    self.setup_answers.get("provider_dimensions") or 0
                ),
                qdrant_url=str(self.setup_answers.get("qdrant_url") or "${QDRANT_URL}"),
                qdrant_api_key_env=str(
                    self.setup_answers.get("qdrant_api_key_env") or "QDRANT_API_KEY"
                ),
                collection_name=str(self.setup_answers.get("collection_name") or ""),
                file_mode=str(self.setup_answers.get("file_mode") or "all_text"),
                pdf_mode=str(self.setup_answers.get("pdf_mode") or "local"),
                provider_secret_env=(
                    str(self.setup_answers.get("provider_secret_env"))
                    if self.setup_answers.get("provider_secret_env")
                    else None
                ),
                azure_endpoint=str(
                    self.setup_answers.get("azure_endpoint", "${AZURE_OPENAI_ENDPOINT}")
                ),
            )
        )
        report = validate_config_data(config_data)
        if not report.ok:
            self._render_error("Generated config is invalid", "\n".join(report.errors))
            self._set_activity(
                ["Config was not written. Fix setup answers and try again."]
            )
            return

        _write_yaml_config(self.config, config_data)
        self.setup_active = False
        self.first_run_setup = False
        self.setup_answers = {}
        self.setup_step_index = 0
        self._load_config_state(self.config)
        self._load_context()
        self._reset_sources_table()
        self._refresh_collections_table()
        self._refresh_status()
        self._render_config_summary()
        self._set_activity(
            [
                f"[green]Config written:[/green] {self.config}",
                "Next: set environment variables, then run `/ingest config`.",
            ]
        )

    def _render_setup_step(self) -> None:
        """Render the current first-run setup wizard prompt."""
        steps = self._setup_steps()
        current = steps[min(self.setup_step_index, len(steps) - 1)]
        main = self.query_one("#main", RichLog)
        main.clear()
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column()
        table.add_row("Target", str(self.config))
        table.add_row("Step", f"{self.setup_step_index + 1} / {len(steps)}")
        table.add_row("Prompt", current.prompt)
        if current.default:
            table.add_row("Default", current.default)
        if current.choices:
            table.add_row("Choices", ", ".join(current.choices))
        table.add_row("Controls", "Enter accepts default · back · cancel · q")
        main.write(
            Panel(
                table,
                title="First-run setup",
                border_style="cyan",
                box=box.ROUNDED,
                safe_box=True,
            )
        )
        answered = [
            f"[dim]{key}[/dim] = {value}" for key, value in self.setup_answers.items()
        ]
        self._set_activity(answered or ["No config found. Setup will create one."])
        command_input = self.query_one("#command", Input)
        command_input.placeholder = f"{current.prompt}" + (
            f" [{current.default}]" if current.default else ""
        )

    def _scope_label(self) -> str:
        """Return a compact label for the active retrieval scope."""
        if self.repo_list and not self.collection:
            count = len(self.collection_targets)
            return f"all repo-list collections ({count})"
        if self.collection:
            return self.collection
        default_collection = self.loaded_config.get("qdrant", {}).get(
            "collection_name", "config default"
        )
        return str(default_collection)

    def _provider_label(self) -> str:
        """Return embedding and answer provider details for the header."""
        embedding = self.loaded_config.get("embedding_provider", "unknown")
        try:
            answering = _answering_config(self.loaded_config)
            answer = f"{answering.get('provider')} / {answering.get('model')}"
        except ValueError:
            answer = "answers not configured"
        return f"embedding {embedding} · answer {answer}"

    def _refresh_status(self) -> None:
        """Render fixed session status at the top of the app."""
        dirty_label = (
            f" [yellow]unsaved {len(self.dirty_paths)}[/yellow]"
            if self.dirty_paths
            else ""
        )
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(justify="right", no_wrap=True)
        grid.add_row(f"[bold cyan]{PROJECT_TITLE}[/bold cyan]", f"v{_app_version()}")
        grid.add_row(
            f"[dim]{PROJECT_REPO_LABEL}[/dim]",
            f"[bold]scope[/bold] {self._scope_label()}",
        )
        grid.add_row(
            f"[dim]{self.config}[/dim]{dirty_label}",
            f"limit {self.limit or 'config'} · parent "
            f"{'on' if self.with_parent_window else 'off'}",
        )
        grid.add_row("[dim]" + self._provider_label() + "[/dim]", "")
        self.query_one("#status", Static).update(grid)

    def _set_activity(self, lines: List[str]) -> None:
        """Replace the activity pane with the latest progress or status lines."""
        activity = self.query_one("#activity", RichLog)
        activity.clear()
        for line in lines:
            activity.write(line)

    @contextmanager
    def _backend_config_path(self) -> Iterator[str]:
        """Yield a config path for backend code, using a temp file when dirty."""
        if not self.dirty_paths:
            yield str(self.config)
            return

        suffix = self.config.suffix or ".yaml"
        with tempfile.NamedTemporaryFile(
            dir=self.config.parent,
            prefix=f".{self.config.stem}.tui-",
            suffix=suffix,
            delete=False,
        ) as temp_config:
            temp_path = Path(temp_config.name)
        os.chmod(temp_path, 0o600)
        _write_yaml_config(temp_path, self.working_config)
        try:
            yield str(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def _begin_operation(self, title: str, step: str, detail: str) -> bool:
        """Start a visible background operation and disable command input."""
        if self.active_worker is not None and self.active_worker.is_running:
            self._render_error(
                "Still working",
                "Wait for the current operation to finish before starting another one.",
            )
            return False

        self.operation_running = True
        self.operation_title = title
        self.operation_step = step
        self.operation_detail = detail
        self.operation_phases = [step]
        self.operation_started_at = time.monotonic()
        self.operation_frame = 0
        command_input = self.query_one("#command", Input)
        command_input.disabled = True
        command_input.placeholder = "Working..."
        self._render_working_panel(title, detail)
        if self.activity_timer is not None:
            self.activity_timer.resume()
        self._tick_activity()
        return True

    def _finish_operation(self, lines: List[str]) -> None:
        """Stop the spinner and re-enable the command input."""
        self.operation_running = False
        if self.activity_timer is not None:
            self.activity_timer.pause()
        command_input = self.query_one("#command", Input)
        command_input.disabled = False
        command_input.placeholder = "/ask How do I configure authentication?"
        command_input.focus()
        self._set_activity(lines)

    def _finish_operation_with_error(self, title: str, message: str) -> None:
        """Stop the active operation and render an error."""
        self.operation_running = False
        if self.activity_timer is not None:
            self.activity_timer.pause()
        command_input = self.query_one("#command", Input)
        command_input.disabled = False
        command_input.placeholder = "/ask How do I configure authentication?"
        command_input.focus()
        self._render_error(title, message)

    def _render_working_panel(self, title: str, detail: str) -> None:
        """Replace the main pane with an in-progress state immediately."""
        main = self.query_one("#main", RichLog)
        main.clear()
        main.write(
            Panel(
                detail,
                title=f"{title} in progress",
                border_style="yellow",
                box=box.ROUNDED,
                safe_box=True,
            )
        )

    def _tick_activity(self) -> None:
        """Animate the activity pane while a worker is running."""
        if not self.operation_running:
            return
        frame = SPINNER_FRAMES[self.operation_frame % len(SPINNER_FRAMES)]
        self.operation_frame += 1
        elapsed = time.monotonic() - self.operation_started_at
        phases = " -> ".join(dict.fromkeys(self.operation_phases))
        self._set_activity(
            [
                f"[bold cyan]{frame} {self.operation_title}[/bold cyan] "
                f"[dim]{elapsed:.1f}s[/dim]",
                f"Current: [bold]{self.operation_step}[/bold]",
                f"[dim]{phases}[/dim]",
            ]
        )

    def _record_phase(self, label: str, step: str, phases: List[str]) -> None:
        """Record worker progress and repaint the activity pane."""
        self.operation_title = label
        self.operation_step = step
        self.operation_phases = phases
        self._tick_activity()

    def _phase_callback(self, label: str, phases: List[str]):
        """Return a progress callback that updates the activity pane."""

        def update(step: str) -> None:
            phases.append(step)
            self.call_from_thread(
                self._record_phase,
                label,
                step,
                list(dict.fromkeys(phases)),
            )

        return update

    def _render_help(self) -> None:
        """Render TUI command help without appending another menu block."""
        self._render_command_palette("")

    def _matching_command_specs(self, filter_text: str) -> List[SlashCommandSpec]:
        """Return command specs matching a live filter or category name."""
        normalized_filter = filter_text.strip().lower()
        if not normalized_filter:
            return list(SLASH_COMMANDS)

        matched: List[SlashCommandSpec] = []
        for spec in SLASH_COMMANDS:
            category = spec.category.lower()
            command = spec.command.lower().lstrip("/")
            aliases = " ".join(alias.lower().lstrip("/") for alias in spec.aliases)
            searchable = " ".join(
                [
                    category,
                    command,
                    aliases,
                    spec.description.lower(),
                    " ".join(tag.lower() for tag in spec.tags),
                ]
            )
            if (
                command.startswith(normalized_filter)
                or category.startswith(normalized_filter)
                or normalized_filter in searchable
            ):
                matched.append(spec)
        return matched

    def _render_command_palette(self, filter_text: str = "") -> None:
        """Render slash command suggestions in the main pane."""
        normalized_filter = filter_text.strip().lower()
        rows = self._matching_command_specs(normalized_filter)
        category_names = {category.lower(): category for category in COMMAND_CATEGORIES}
        category_focus = category_names.get(normalized_filter)
        table = Table(
            title=category_focus or "Slash Commands",
            box=box.SIMPLE_HEAD,
            header_style="bold cyan",
            expand=True,
        )
        if category_focus is None:
            table.add_column("Group", style="bold", no_wrap=True)
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Action")
        if category_focus is not None:
            table.add_column("Example", style="dim")

        for category in COMMAND_CATEGORIES:
            category_rows = [spec for spec in rows if spec.category == category]
            for index, spec in enumerate(category_rows):
                row = [spec.command, spec.description]
                if category_focus is None:
                    row.insert(0, category if index == 0 else "")
                else:
                    row.append(spec.example)
                table.add_row(*row)
        if not rows:
            if category_focus is None:
                table.add_row("", "[dim]No matches[/dim]", "Keep typing or use /help.")
            else:
                table.add_row("[dim]No matches[/dim]", "Use `/help` to show all.", "")
        main = self.query_one("#main", RichLog)
        main.clear()
        main.write(
            Panel(
                table,
                title=(
                    f"{category_focus} Commands"
                    if category_focus
                    else "Type / to Browse Commands"
                ),
                border_style="cyan",
                box=box.ROUNDED,
                safe_box=True,
            )
        )
        self._set_activity(
            [
                "Slash commands are grouped by workflow.",
                "Use `/help config`, `/help quality`, or keep typing to filter.",
            ]
        )
        if self.last_error:
            self._set_activity(
                [f"[bold red]Startup warning:[/bold red] {self.last_error}"]
            )

    def _reset_sources_table(self) -> None:
        """Reset the ranked sources table."""
        table = self.query_one("#sources", DataTable)
        table.clear(columns=True)
        table.add_columns("#", "Score", "Collection", "Source", "Snippet")

    def _collection_status_for(
        self, collection_name: str
    ) -> Optional[CollectionCompatibility]:
        """Return the most recent compatibility status for a collection."""
        for status in self.collection_statuses:
            if status.collection_name == collection_name:
                return status
        return None

    def _refresh_collections_table(self) -> None:
        """Render the available collection list in the sidebar."""
        table = self.query_one("#collections", DataTable)
        table.clear(columns=True)
        table.add_columns("#", "Collection", "Status", "Repository", "Reason")
        if not self.collection_targets:
            table.add_row("-", self._scope_label(), "configured", "config", "")
            return
        for index, target in enumerate(self.collection_targets, 1):
            status = self._collection_status_for(target.collection_name)
            table.add_row(
                str(index),
                target.collection_name,
                status.status if status else "configured",
                _clip(target.repository_name or target.repository_url or "config", 30),
                _clip(status.reason if status else target.branch or "default", 44),
            )

    def _update_sources(self, response: QueryResponse) -> None:
        """Render search or ask source hits in the source table."""
        self._reset_sources_table()
        table = self.query_one("#sources", DataTable)
        for index, hit in enumerate(response.hits, 1):
            table.add_row(
                str(index),
                f"{hit.score:.4f}",
                hit.collection or response.collection,
                _clip(hit.file_path, 36),
                _clip(hit.preview or hit.content, 120),
            )

    def _render_error(self, title: str, message: str) -> None:
        """Render an error state without crashing the app."""
        message = _redact_text(message, self.working_config)
        self.operation_running = False
        if self.activity_timer is not None:
            self.activity_timer.pause()
        command_input = self.query_one("#command", Input)
        command_input.disabled = False
        main = self.query_one("#main", RichLog)
        main.clear()
        main.write(
            Panel(
                message,
                title=title,
                border_style="red",
                box=box.ROUNDED,
                safe_box=True,
            )
        )
        self._set_activity([f"[bold red]{title}[/bold red]", message])

    def _run_search(self, query: str) -> None:
        """Run semantic retrieval and replace the main pane with results."""
        if not query:
            self._render_error("Search needs text", "Use `/search <query>`.")
            return

        if not self._begin_operation(
            "Search",
            "Encoding query",
            f"Searching for `{_clip(query, 180)}`.",
        ):
            return
        self._reset_sources_table()
        self.active_worker = self.run_worker(
            partial(
                self._search_worker,
                query,
                self.limit,
                self.with_parent_window,
                self.collection,
                str(self.repo_list) if self.repo_list else None,
            ),
            name="search",
            group="operation",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    def _search_worker(
        self,
        query: str,
        limit: Optional[int],
        with_parent_window: bool,
        collection: Optional[str],
        repo_list: Optional[str],
    ) -> None:
        """Run semantic retrieval in a worker thread."""
        phases: List[str] = ["Encoding query"]
        try:
            with self._backend_config_path() as config_path:
                response = execute_query(
                    config_path=config_path,
                    query=query,
                    limit=limit,
                    with_parent_window=with_parent_window,
                    quiet=True,
                    progress=self._phase_callback("Search", phases),
                    collection=collection,
                    repo_list=repo_list,
                )
        except (LookupError, ValueError, SystemExit) as exc:
            self.call_from_thread(
                self._finish_operation_with_error, "Search failed", str(exc)
            )
            return

        completed = list(dict.fromkeys(phases))
        self.call_from_thread(self._render_query_response, response, completed)
        self.call_from_thread(
            self._finish_operation,
            [
                "[bold green]Search complete[/bold green]",
                "Phases: " + " -> ".join(completed),
            ],
        )

    def _run_ask(self, question: str) -> None:
        """Run retrieval-augmented answering and replace the main pane."""
        if not question:
            self._render_error("Ask needs a question", "Use `/ask <question>`.")
            return

        if not self._begin_operation(
            "Answer",
            "Encoding question",
            f"Answering `{_clip(question, 180)}`.",
        ):
            return
        self._reset_sources_table()
        self.active_worker = self.run_worker(
            partial(
                self._ask_worker,
                question,
                self.limit,
                self.with_parent_window,
                self.collection,
                str(self.repo_list) if self.repo_list else None,
            ),
            name="ask",
            group="operation",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    def _ask_worker(
        self,
        question: str,
        limit: Optional[int],
        with_parent_window: bool,
        collection: Optional[str],
        repo_list: Optional[str],
    ) -> None:
        """Run retrieval-augmented answering in a worker thread."""
        phases: List[str] = ["Encoding question"]
        try:
            with self._backend_config_path() as config_path:
                response = generate_answer(
                    config_path=config_path,
                    question=question,
                    limit=limit,
                    with_parent_window=with_parent_window,
                    quiet=True,
                    progress=self._phase_callback("Answer", phases),
                    collection=collection,
                    repo_list=repo_list,
                )
        except (LookupError, ValueError, RuntimeError, SystemExit) as exc:
            self.call_from_thread(
                self._finish_operation_with_error, "Ask failed", str(exc)
            )
            return

        completed = list(dict.fromkeys(phases))
        self.call_from_thread(self._render_answer_response, response, completed)
        self.call_from_thread(
            self._finish_operation,
            [
                "[bold green]Answer complete[/bold green]",
                "Phases: " + " -> ".join(completed),
                (
                    f"retrieve {response.timings.retrieval_seconds:.2f}s · "
                    f"answer {response.timings.answer_seconds:.2f}s"
                ),
            ],
        )

    def _render_query_response(
        self, response: QueryResponse, phases: List[str]
    ) -> None:
        """Render a retrieval response in the main output pane."""
        self.collection_statuses = response.collection_statuses
        self._refresh_collections_table()
        self._update_sources(response)
        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="bold")
        summary.add_column()
        summary.add_row("Query", _clip(response.query, 160))
        summary.add_row("Scope", ", ".join(response.collections) or response.collection)
        summary.add_row(
            "Results",
            f"{len(response.hits)} matches from {response.candidates} candidates",
        )
        if phases:
            summary.add_row("Phases", " -> ".join(phases))
        if response.warnings:
            summary.add_row("Warnings", _clip("; ".join(response.warnings), 180))
        if response.skipped_collections:
            skipped = "; ".join(
                f"{status.collection_name} ({status.status})"
                for status in response.skipped_collections
            )
            summary.add_row("Skipped", _clip(skipped, 180))

        results = Table(
            box=box.SIMPLE_HEAD,
            header_style="bold cyan",
            expand=True,
        )
        results.add_column("#", justify="right", width=3)
        results.add_column("Score", justify="right", width=8)
        results.add_column("Collection", style="cyan")
        results.add_column("Source", style="green")
        results.add_column("Matched snippet")
        for index, hit in enumerate(response.hits, 1):
            results.add_row(
                str(index),
                f"{hit.score:.4f}",
                hit.collection or response.collection,
                _clip(hit.file_path, 36),
                _clip(hit.preview or hit.content, 180),
            )

        main = self.query_one("#main", RichLog)
        main.clear()
        main.write(
            Panel(
                summary,
                title="Vector Search",
                border_style="cyan",
                box=box.ROUNDED,
                safe_box=True,
            )
        )
        main.write(results)

    def _render_answer_response(
        self, response: AnswerResponse, phases: List[str]
    ) -> None:
        """Render a generated answer and source metadata."""
        self.collection_statuses = response.retrieval.collection_statuses
        self._refresh_collections_table()
        self._update_sources(response.retrieval)
        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="bold")
        summary.add_column()
        summary.add_row("Question", _clip(response.question, 160))
        summary.add_row("Model", response.model)
        summary.add_row(
            "Context",
            f"{len(response.retrieval.hits)} sources · {response.context_chars} chars",
        )
        if phases:
            summary.add_row("Phases", " -> ".join(phases))
        if response.retrieval.warnings:
            summary.add_row(
                "Warnings", _clip("; ".join(response.retrieval.warnings), 180)
            )
        if response.retrieval.skipped_collections:
            skipped = "; ".join(
                f"{status.collection_name} ({status.status})"
                for status in response.retrieval.skipped_collections
            )
            summary.add_row("Skipped", _clip(skipped, 180))

        main = self.query_one("#main", RichLog)
        main.clear()
        main.write(
            Panel(
                summary,
                title="AI Answer",
                border_style="cyan",
                box=box.ROUNDED,
                safe_box=True,
            )
        )
        main.write(
            Panel(
                response.answer or "No answer was returned by the chat model.",
                title="Answer",
                border_style="green" if response.retrieval.hits else "yellow",
                box=box.ROUNDED,
                safe_box=True,
            )
        )

    def _show_collections(self) -> None:
        """Render available collections in the main pane and sidebar."""
        self._load_context()
        if self.collection_targets:
            try:
                with self._backend_config_path() as config_path:
                    self.collection_statuses = inspect_collection_targets(
                        config_path, self.collection_targets, quiet=True
                    )
            except (OSError, ValueError, SystemExit) as exc:
                self.last_error = _redact_text(str(exc), self.working_config)
        self._refresh_collections_table()
        table = Table(
            title="Collections",
            box=box.SIMPLE_HEAD,
            header_style="bold cyan",
            expand=True,
        )
        table.add_column("#", justify="right", width=3)
        table.add_column("Collection", style="cyan")
        table.add_column("Status")
        table.add_column("Repository", style="green")
        table.add_column("Reason")
        targets = self.collection_targets or [
            CollectionTarget(
                collection_name=self._scope_label(), repository_name="config"
            )
        ]
        for index, target in enumerate(targets, 1):
            status = self._collection_status_for(target.collection_name)
            table.add_row(
                str(index),
                target.collection_name,
                status.status if status else "configured",
                _clip(target.repository_name or target.repository_url or "config", 40),
                _clip(status.reason if status else target.branch or "default", 80),
            )
        main = self.query_one("#main", RichLog)
        main.clear()
        main.write(table)
        activity = [f"Scope: {self._scope_label()}"]
        if self.last_error:
            activity.append(f"[yellow]{self.last_error}[/yellow]")
        self._set_activity(activity)

    def _path_from_argument(self, value: str) -> Optional[Path]:
        """Return a path from a command argument, supporting shell-style quotes."""
        try:
            parts = shlex.split(value)
        except ValueError as exc:
            self._render_error("Invalid path", str(exc))
            return None
        if not parts:
            return None
        return Path(parts[0]).expanduser()

    def _load_config_command(self, value: str) -> None:
        """Load a different config file without leaving the TUI."""
        if not value.strip():
            self._render_config_summary()
            return

        try:
            args = _split_args(value)
        except ValueError as exc:
            self._render_error("Invalid config command", str(exc))
            return
        force = "--force" in args
        args = [arg for arg in args if arg != "--force"]
        if self.dirty_paths and not force:
            self._render_error(
                "Unsaved config changes",
                (
                    "Save or discard changes before loading another config. "
                    "Use `/save-config --confirm`, `/discard-config-changes`, "
                    "or `/config PATH --force`."
                ),
            )
            return

        config_path = Path(args[0]).expanduser() if args else None
        if config_path is None:
            self._render_error(
                "Config path needed", "Use `/config path/to/config.yaml`."
            )
            return
        if not config_path.exists() or not config_path.is_file():
            self._render_error("Config not found", str(config_path))
            return

        try:
            self._load_config_state(config_path)
            self.collection = None
            self.collection_targets = resolve_collection_targets(
                self.working_config,
                collection=None,
                repo_list=str(self.repo_list) if self.repo_list else None,
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            self._render_error("Config load failed", str(exc))
            return

        self.last_error = ""
        self.sub_title = str(self.config)
        self._reset_sources_table()
        self._refresh_collections_table()
        self._refresh_status()
        self._render_help()
        self._set_activity(
            [
                f"Loaded config: {self.config}",
                f"Scope reset to {self._scope_label()}.",
            ]
        )

    def _load_repo_list_command(self, value: str) -> None:
        """Load or clear a repository list without restarting the TUI."""
        normalized = value.strip().lower()
        if normalized in {"clear", "none", "off"}:
            self.repo_list = None
            self.collection = None
            self._load_context()
            self._refresh_collections_table()
            self._refresh_status()
            self._set_activity(
                ["Repository list cleared.", f"Scope: {self._scope_label()}"]
            )
            return

        repo_list_path = self._path_from_argument(value)
        if repo_list_path is None:
            self._render_error(
                "Repo-list path needed",
                "Use `/repo-list repositories.yaml` or `/repo-list clear`.",
            )
            return
        if not repo_list_path.exists() or not repo_list_path.is_file():
            self._render_error("Repo-list not found", str(repo_list_path))
            return

        self.repo_list = repo_list_path
        self.collection = None
        self._load_context()
        if self.last_error:
            self._render_error("Repo-list load failed", self.last_error)
            return
        self._refresh_collections_table()
        self._refresh_status()
        self._show_collections()
        self._set_activity(
            [
                f"Loaded repo-list: {self.repo_list}",
                f"Scope: {self._scope_label()}.",
            ]
        )

    def _render_config_summary(self) -> None:
        """Render grouped editable config values."""
        table = Table(
            title="Editable Config",
            box=box.SIMPLE_HEAD,
            header_style="bold cyan",
            expand=True,
        )
        table.add_column("Group", style="bold")
        table.add_column("Path", style="cyan")
        table.add_column("Value")
        for group_name, paths in CONFIG_SUMMARY_GROUPS.items():
            for index, path in enumerate(paths):
                try:
                    value = _get_config_path(self.working_config, path)
                except KeyError:
                    value = "<missing>"
                table.add_row(
                    group_name if index == 0 else "",
                    path,
                    _redact_config_value(path, value),
                )
        main = self.query_one("#main", RichLog)
        main.clear()
        main.write(table)
        self._set_activity(
            [
                f"Config: {self.config}",
                "Use `/set`, `/secret`, `/changes`, and `/save-config`.",
            ]
        )

    def _get_config_command(self, value: str) -> None:
        """Render one editable config value."""
        path = value.strip()
        if path not in CONFIG_PATH_TYPES:
            self._render_error("Unknown config path", f"Not editable: {path}")
            return
        try:
            current = _get_config_path(self.working_config, path)
        except KeyError:
            current = "<missing>"
        self.query_one("#main", RichLog).clear()
        self.query_one("#main", RichLog).write(
            Panel(
                _redact_config_value(path, current),
                title=path,
                border_style="cyan",
                box=box.ROUNDED,
                safe_box=True,
            )
        )
        self._set_activity([f"{path} shown."])

    def _set_config_command(self, value: str) -> None:
        """Stage a typed non-secret config value."""
        try:
            args = _split_args(value)
        except ValueError as exc:
            self._render_error("Invalid set command", str(exc))
            return
        if len(args) < 2:
            self._render_error("Set needs a value", "Use `/set path value`.")
            return
        path = args[0]
        raw_value = " ".join(args[1:])
        try:
            parsed_value = _coerce_config_value(path, raw_value)
        except (TypeError, ValueError) as exc:
            self._render_error("Config value rejected", str(exc))
            return
        self._stage_config_value(path, parsed_value)

    def _secret_config_command(self, value: str) -> None:
        """Stage a secret config value as an env placeholder."""
        try:
            args = _split_args(value)
        except ValueError as exc:
            self._render_error("Invalid secret command", str(exc))
            return
        if len(args) != 2:
            self._render_error(
                "Secret needs path and env var", "Use `/secret path ENV_VAR`."
            )
            return
        path, env_name = args
        if path not in SECRET_PATHS:
            self._render_error("Not a secret path", f"Use `/set` for {path}.")
            return
        try:
            placeholder = _secret_placeholder(env_name)
        except ValueError as exc:
            self._render_error("Invalid secret env var", str(exc))
            return
        self._stage_config_value(path, placeholder)

    def _stage_config_value(self, path: str, value: Any) -> None:
        """Stage a working config value and refresh dependent state."""
        try:
            old_value = _get_config_path(self.working_config, path)
        except KeyError:
            old_value = "<missing>"
        _set_config_path(self.working_config, path, value)
        try:
            original_value = _get_config_path(self.original_config, path)
        except KeyError:
            original_value = "<missing>"
        if original_value == value:
            self.dirty_paths.discard(path)
        else:
            self.dirty_paths.add(path)
        self.loaded_config = self.working_config
        self._load_context()
        self._refresh_collections_table()
        self._refresh_status()
        warning = (
            " This may require reingestion." if path in INDEX_AFFECTING_PATHS else ""
        )
        self._set_activity(
            [
                f"Staged {path}.",
                f"{len(self.dirty_paths)} unsaved config change(s).{warning}",
            ]
        )
        main = self.query_one("#main", RichLog)
        main.clear()
        main.write(
            Panel(
                _format_change(path, old_value, value),
                title="Config Change Staged",
                border_style="yellow" if path in INDEX_AFFECTING_PATHS else "cyan",
                box=box.ROUNDED,
                safe_box=True,
            )
        )

    def _config_changes(self) -> List[tuple[str, Any, Any]]:
        """Return dirty path changes in display order."""
        changes = []
        for path in sorted(self.dirty_paths):
            try:
                old_value = _get_config_path(self.original_config, path)
            except KeyError:
                old_value = "<missing>"
            try:
                new_value = _get_config_path(self.working_config, path)
            except KeyError:
                new_value = "<missing>"
            if old_value != new_value:
                changes.append((path, old_value, new_value))
        return changes

    def _render_config_changes(self) -> None:
        """Render pending config changes."""
        main = self.query_one("#main", RichLog)
        main.clear()
        changes = self._config_changes()
        if not changes:
            main.write(
                Panel(
                    "No unsaved config changes.",
                    title="Config Changes",
                    border_style="green",
                    box=box.ROUNDED,
                    safe_box=True,
                )
            )
            self._set_activity(["No unsaved config changes."])
            return

        table = Table(
            title="Unsaved Config Changes",
            box=box.SIMPLE_HEAD,
            header_style="bold cyan",
            expand=True,
        )
        table.add_column("Path", style="cyan")
        table.add_column("Before")
        table.add_column("After")
        table.add_column("Note")
        for path, old_value, new_value in changes:
            table.add_row(
                path,
                _redact_config_value(path, old_value),
                _redact_config_value(path, new_value),
                "reingest" if path in INDEX_AFFECTING_PATHS else "",
            )
        main.write(table)
        self._set_activity([f"{len(changes)} unsaved config change(s)."])

    def _validation_table(self, report: Any) -> Table:
        """Build a validation table for the working config."""
        table = Table(
            title="Configuration Validation",
            box=box.SIMPLE_HEAD,
            header_style="bold cyan",
            expand=True,
        )
        table.add_column("Level", no_wrap=True)
        table.add_column("Message")
        if not report.errors and not report.warnings:
            table.add_row("OK", "Config structure looks good.")
        for error in report.errors:
            table.add_row("ERROR", error)
        for warning in report.warnings:
            table.add_row("WARN", warning)
        return table

    def _save_config_command(self, value: str) -> None:
        """Preview or confirm saving the active config."""
        confirm = "--confirm" in value.split()
        self._save_config_to_path(self.config, confirm=confirm)

    def _save_config_as_command(self, value: str) -> None:
        """Preview or confirm saving the active config to a different path."""
        try:
            args = _split_args(value)
        except ValueError as exc:
            self._render_error("Invalid save-as command", str(exc))
            return
        confirm = "--confirm" in args
        args = [arg for arg in args if arg != "--confirm"]
        if not args:
            self._render_error(
                "Save-as needs a path", "Use `/save-config-as path --confirm`."
            )
            return
        self._save_config_to_path(Path(args[0]).expanduser(), confirm=confirm)

    def _save_config_to_path(self, target: Path, confirm: bool) -> None:
        """Preview or write the working config to disk."""
        report = validate_config_data(self.working_config)
        main = self.query_one("#main", RichLog)
        main.clear()
        if report.errors:
            main.write(self._validation_table(report))
            self._set_activity(["Save blocked because config validation has errors."])
            return

        if not confirm:
            self._render_config_changes()
            main.write(self._validation_table(report))
            command = (
                "/save-config --confirm"
                if target == self.config
                else f"/save-config-as {target} --confirm"
            )
            self._set_activity([f"Review changes, then run `{command}` to write."])
            return

        backup = None
        if target.exists():
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup = target.with_name(f"{target.name}.{timestamp}.bak")
            shutil.copy2(target, backup)
        _write_yaml_config(target, self.working_config)
        self.config = target
        self.config_source_path = target
        self.original_config = deepcopy(self.working_config)
        self.dirty_paths.clear()
        self.sub_title = str(self.config)
        self._refresh_status()
        self._render_config_summary()
        lines = [f"Config saved: {target}"]
        if backup:
            lines.append(f"Backup created: {backup}")
        self._set_activity(lines)

    def _discard_config_changes(self) -> None:
        """Discard pending config edits."""
        self.working_config = deepcopy(self.original_config)
        self.loaded_config = self.working_config
        self.dirty_paths.clear()
        self.collection = None
        self._load_context()
        self._refresh_collections_table()
        self._refresh_status()
        self._render_config_summary()
        self._set_activity(["Unsaved config changes discarded."])

    def _change_scope(self, value: str) -> None:
        """Change the collection scope for future ask/search commands."""
        if not value:
            self._render_error(
                "Scope needs a value", "Use `/scope all` or `/scope NAME`."
            )
            return
        if value.lower() == "all":
            self.collection = None
        else:
            self.collection = value
        self._load_context()
        self._refresh_collections_table()
        self._refresh_status()
        self._set_activity([f"Scope set to {self._scope_label()}."])

    def _change_limit(self, value: str) -> None:
        """Change the result limit."""
        try:
            new_limit = int(value)
        except ValueError:
            self._render_error("Invalid limit", "Use `limit <whole number>`.")
            return
        if new_limit <= 0:
            self._render_error("Invalid limit", "Limit must be greater than zero.")
            return
        self.limit = new_limit
        self._refresh_status()
        self._set_activity([f"Result limit set to {self.limit}."])

    def _change_parent_window(self, value: str) -> None:
        """Toggle expanded parent-window context."""
        normalized = value.lower()
        if normalized not in {"on", "off"}:
            self._render_error(
                "Invalid parent setting", "Use `parent on` or `parent off`."
            )
            return
        self.with_parent_window = normalized == "on"
        self._refresh_status()
        self._set_activity(
            [
                "Parent-window context "
                + ("enabled." if self.with_parent_window else "disabled.")
            ]
        )

    def _run_doctor_command(self, value: str) -> None:
        """Run index health checks in a worker thread."""
        try:
            args = _split_args(value)
        except ValueError as exc:
            self._render_error("Invalid doctor command", str(exc))
            return
        apply_indexes = "--apply-indexes" in args
        yes = "--yes" in args
        if not self._begin_operation(
            "Doctor",
            "Checking collections",
            "Checking Qdrant collection, vector, payload, and index health.",
        ):
            return
        self.active_worker = self.run_worker(
            partial(self._doctor_worker, apply_indexes, yes),
            name="doctor",
            group="operation",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    def _doctor_worker(self, apply_indexes: bool, yes: bool) -> None:
        """Run doctor checks in a worker thread."""
        self.call_from_thread(
            self._record_phase,
            "Doctor",
            "Checking Qdrant",
            ["Checking collections", "Checking Qdrant"],
        )
        try:
            with self._backend_config_path() as config_path:
                report = run_doctor(
                    config_path=config_path,
                    repo_list=str(self.repo_list) if self.repo_list else None,
                    collection=self.collection,
                    apply_indexes=apply_indexes,
                    yes=yes,
                )
        except (LookupError, ValueError, SystemExit) as exc:
            self.call_from_thread(
                self._finish_operation_with_error, "Doctor failed", str(exc)
            )
            return
        self.call_from_thread(self._render_doctor_report, report)
        self.call_from_thread(
            self._finish_operation,
            [
                "[bold green]Doctor complete[/bold green]"
                if report.ok
                else "[bold yellow]Doctor found issues[/bold yellow]",
                f"{len(report.findings)} finding(s).",
            ],
        )

    def _run_benchmark_command(self, value: str) -> None:
        """Run retrieval benchmark cases in a worker thread."""
        try:
            args = _split_args(value)
        except ValueError as exc:
            self._render_error("Invalid benchmark command", str(exc))
            return
        case_args: List[str] = []
        skip_next = False
        for index, arg in enumerate(args):
            if skip_next:
                skip_next = False
                continue
            if arg == "--fail-under":
                skip_next = index + 1 < len(args)
                continue
            if not arg.startswith("--"):
                case_args.append(arg)
        if not case_args:
            try:
                cases = resolve_benchmark_cases(None, self.config)
            except ValueError as exc:
                self._render_error("Benchmark file not found", str(exc))
                return
        else:
            try:
                cases = resolve_benchmark_cases(
                    Path(case_args[0]).expanduser(), self.config
                )
            except ValueError as exc:
                self._render_error("Benchmark file not found", str(exc))
                return
        fail_under = 0.8
        if "--fail-under" in args:
            try:
                fail_under = float(args[args.index("--fail-under") + 1])
            except (IndexError, ValueError):
                self._render_error(
                    "Invalid fail-under", "Use `/benchmark --fail-under 0.8`."
                )
                return
        improve_on_fail = "--improve-on-fail" in args
        if not self._begin_operation(
            "Benchmark",
            "Running cases",
            f"Benchmarking retrieval with `{cases}`.",
        ):
            return
        self.active_worker = self.run_worker(
            partial(self._benchmark_worker, cases, fail_under, improve_on_fail),
            name="benchmark",
            group="operation",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    def _benchmark_worker(
        self, cases: Path, fail_under: float, improve_on_fail: bool
    ) -> None:
        """Run retrieval benchmark in a worker thread."""
        self.call_from_thread(
            self._record_phase,
            "Benchmark",
            "Running retrieval cases",
            ["Running cases", "Running retrieval cases"],
        )
        try:
            with self._backend_config_path() as config_path:
                report = run_benchmark(
                    config_path=config_path,
                    cases_path=str(cases),
                    repo_list=str(self.repo_list) if self.repo_list else None,
                    collection=self.collection,
                    limit=self.limit,
                    fail_under=fail_under,
                )
                improve_report = (
                    run_improve(
                        config_path=config_path,
                        cases_path=str(cases),
                        repo_list=str(self.repo_list) if self.repo_list else None,
                        collection=self.collection,
                    )
                    if improve_on_fail and not report.passed
                    else None
                )
        except (LookupError, ValueError, SystemExit) as exc:
            self.call_from_thread(
                self._finish_operation_with_error, "Benchmark failed", str(exc)
            )
            return
        self.call_from_thread(self._render_benchmark_report, report, improve_report)
        self.call_from_thread(
            self._finish_operation,
            [
                "[bold green]Benchmark passed[/bold green]"
                if report.passed
                else "[bold yellow]Benchmark below threshold[/bold yellow]",
                f"Pass rate: {report.pass_rate:.0%}.",
            ],
        )

    def _run_improve_command(self, value: str) -> None:
        """Run safe improvement analysis or apply."""
        try:
            args = _split_args(value)
        except ValueError as exc:
            self._render_error("Invalid improve command", str(exc))
            return
        apply_changes = "--apply" in args
        case_args = [arg for arg in args if not arg.startswith("--")]
        cases = None
        if case_args:
            try:
                cases = resolve_benchmark_cases(
                    Path(case_args[0]).expanduser(), self.config
                )
            except ValueError as exc:
                self._render_error("Benchmark file not found", str(exc))
                return
        else:
            try:
                cases = resolve_benchmark_cases(None, self.config)
            except ValueError:
                cases = None
        if apply_changes and self.dirty_paths:
            self._render_error(
                "Save config first",
                "`/improve --apply` needs a saved config. Use `/save-config --confirm` first.",
            )
            return
        if not self._begin_operation(
            "Improve",
            "Analyzing quality",
            "Building safe retrieval and index improvement suggestions.",
        ):
            return
        self.active_worker = self.run_worker(
            partial(self._improve_worker, cases, apply_changes),
            name="improve",
            group="operation",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    def _improve_worker(self, cases: Optional[Path], apply_changes: bool) -> None:
        """Run improvement analysis in a worker thread."""
        self.call_from_thread(
            self._record_phase,
            "Improve",
            "Checking safe actions",
            ["Analyzing quality", "Checking safe actions"],
        )
        try:
            config_path = str(self.config) if apply_changes else None
            if config_path is None:
                with self._backend_config_path() as temporary_config:
                    report = run_improve(
                        config_path=temporary_config,
                        cases_path=str(cases) if cases else None,
                        repo_list=str(self.repo_list) if self.repo_list else None,
                        collection=self.collection,
                    )
            else:
                report = run_improve(
                    config_path=config_path,
                    cases_path=str(cases) if cases else None,
                    repo_list=str(self.repo_list) if self.repo_list else None,
                    collection=self.collection,
                    apply=True,
                    yes=True,
                )
        except (LookupError, ValueError, SystemExit) as exc:
            self.call_from_thread(
                self._finish_operation_with_error, "Improve failed", str(exc)
            )
            return
        if report.applied:
            self.call_from_thread(self._load_config_state, self.config)
            self.call_from_thread(self._refresh_status)
        self.call_from_thread(self._render_improve_report, report)
        self.call_from_thread(
            self._finish_operation,
            [
                "[bold green]Improvements applied[/bold green]"
                if report.applied
                else "[bold cyan]Improvement report ready[/bold cyan]",
                f"{len(report.actions)} action(s).",
            ],
        )

    def _render_doctor_report(self, report: Any) -> None:
        """Render doctor output in the main pane."""
        main = self.query_one("#main", RichLog)
        main.clear()
        main.write(doctor_table(report))

    def _render_benchmark_report(self, report: Any, improve_report: Any = None) -> None:
        """Render benchmark output in the main pane."""
        main = self.query_one("#main", RichLog)
        main.clear()
        main.write(benchmark_table(report))
        if improve_report is not None:
            main.write(improve_table(improve_report))

    def _render_improve_report(self, report: Any) -> None:
        """Render improvement output in the main pane."""
        main = self.query_one("#main", RichLog)
        main.clear()
        main.write(improve_table(report))
        if report.backup_path:
            main.write(f"[green]Backup created:[/green] {report.backup_path}")

    def _run_validation(self) -> None:
        """Validate the active configuration."""
        try:
            report = validate_config_data(self.working_config)
        except ValueError as exc:
            self._render_error("Validation failed", str(exc))
            return

        table = self._validation_table(report)
        main = self.query_one("#main", RichLog)
        main.clear()
        main.write(table)
        self._set_activity(["Validation complete."])

    def _run_ingest_command(self, value: str) -> None:
        """Run ingestion after an explicit TUI ingestion choice."""
        if not value:
            self._render_error(
                "Choose ingestion mode",
                "Use `/ingest config`, `/ingest repo-url URL`, or `/ingest repo-list PATH`.",
            )
            return

        repo_url = None
        repo_list = None
        parts = value.split(maxsplit=1)
        mode = parts[0].lower()
        argument = parts[1].strip() if len(parts) > 1 else ""
        if mode == "config":
            pass
        elif mode == "repo-url" and argument:
            repo_url = argument
        elif mode == "repo-list" and argument:
            repo_list = argument
        else:
            self._render_error(
                "Invalid ingestion command",
                "Use `/ingest config`, `/ingest repo-url URL`, or `/ingest repo-list PATH`.",
            )
            return

        if not self._begin_operation(
            "Ingestion",
            "Running ingestion",
            "Processing repository content into Qdrant.",
        ):
            return
        self.active_worker = self.run_worker(
            partial(self._ingest_worker, repo_url, repo_list),
            name="ingest",
            group="operation",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    def _ingest_worker(self, repo_url: Optional[str], repo_list: Optional[str]) -> None:
        """Run ingestion in a worker thread."""
        self.call_from_thread(
            self._record_phase,
            "Ingestion",
            "Running ingestion",
            ["Running ingestion"],
        )
        with self._backend_config_path() as config_path:
            exit_code = run_ingest(
                config_path=config_path,
                repo_url=repo_url,
                repo_list=repo_list,
            )
        if exit_code == 0:
            message = "Ingestion completed successfully."
            style = "green"
        else:
            message = f"Ingestion failed with exit code {exit_code}."
            style = "red"
        self.call_from_thread(self._render_ingest_result, message, style)
        self.call_from_thread(self._finish_operation, [message])

    def _render_ingest_result(self, message: str, style: str) -> None:
        """Render the final ingestion result."""
        main = self.query_one("#main", RichLog)
        main.clear()
        main.write(
            Panel(
                message,
                title="Repository Ingestion",
                border_style=style,
                box=box.ROUNDED,
                safe_box=True,
            )
        )


def run_tui(
    config: Path,
    limit: Optional[int] = None,
    with_parent_window: bool = False,
    collection: Optional[str] = None,
    repo_list: Optional[Path] = None,
    first_run_setup: bool = False,
) -> None:
    """Run the Textual terminal app."""
    GithubQdrantSyncApp(
        config=config,
        limit=limit,
        with_parent_window=with_parent_window,
        collection=collection,
        repo_list=repo_list,
        first_run_setup=first_run_setup,
    ).run()
