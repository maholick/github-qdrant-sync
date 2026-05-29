#!/usr/bin/env python3
"""
GitHub Repository to Qdrant Vector Database Processor

This script clones a GitHub repository, extracts text-based files (configurable),
combines them into a single document, and inserts them into a Qdrant collection
using various embedding providers (Azure OpenAI, Mistral AI, or Sentence Transformers).
"""

import argparse
import fnmatch
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Union
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import AzureOpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models as qdrant_models
import tiktoken

# Import PDF processor
from pdf_processor import PDFProcessor


@dataclass(frozen=True)
class IngestProgressEvent:
    """Structured progress event emitted during repository ingestion."""

    stage: str
    message: str
    current: Optional[int] = None
    total: Optional[int] = None
    percent: Optional[float] = None
    level: str = "info"
    repo: Optional[str] = None
    collection: Optional[str] = None


IngestProgressCallback = Callable[[IngestProgressEvent], None]


class IngestCancelled(Exception):
    """Raised by progress callbacks to cooperatively stop ingestion."""


TURBO_QUANT_BITS = {
    "bits1": qdrant_models.TurboQuantBitSize.BITS1,
    "bits1_5": qdrant_models.TurboQuantBitSize.BITS1_5,
    "bits2": qdrant_models.TurboQuantBitSize.BITS2,
    "bits4": qdrant_models.TurboQuantBitSize.BITS4,
}
TEXT_DETECTION_SAMPLE_BYTES = 8192


def build_qdrant_quantization_config(
    qdrant_config: Dict[str, Any],
) -> Optional[qdrant_models.TurboQuantization]:
    """Build optional Qdrant TurboQuant config from qdrant.quantization."""
    quant_cfg = qdrant_config.get("quantization", {})
    if not isinstance(quant_cfg, dict) or not quant_cfg.get("enabled", False):
        return None

    method = str(quant_cfg.get("method", "turbo")).strip().lower()
    if method != "turbo":
        raise ValueError("qdrant.quantization.method currently supports only 'turbo'")

    bits = str(quant_cfg.get("bits", "bits4")).strip().lower()
    if bits not in TURBO_QUANT_BITS:
        allowed = ", ".join(sorted(TURBO_QUANT_BITS))
        raise ValueError(f"qdrant.quantization.bits must be one of: {allowed}")

    return qdrant_models.TurboQuantization(
        turbo=qdrant_models.TurboQuantQuantizationConfig(
            always_ram=bool(quant_cfg.get("always_ram", True)),
            bits=TURBO_QUANT_BITS[bits],
        )
    )


class EmbeddingInterface(Protocol):
    """Protocol defining the interface for embedding clients."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        ...

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        ...


class EmbeddingCache:
    """Cache for embedding generation to avoid redundant API calls."""

    def __init__(self, max_size=500):
        """Initialize cache with maximum size."""
        self.cache = {}  # content_hash -> embedding
        self.access_order = []  # Track LRU
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get_or_generate(self, text: str, generate_fn):
        """Get cached embedding or generate new one."""
        text_hash = hashlib.md5(text.encode()).hexdigest()

        if text_hash in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.access_order.remove(text_hash)
            self.access_order.append(text_hash)
            return self.cache[text_hash]

        self.misses += 1
        embedding = generate_fn(text)

        # Add to cache with LRU eviction
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_hash = self.access_order.pop(0)
            del self.cache[lru_hash]

        self.cache[text_hash] = embedding
        self.access_order.append(text_hash)
        return embedding

    def get(self, text: str):
        """Get cached embedding if it exists, otherwise return None."""
        text_hash = hashlib.md5(text.encode()).hexdigest()

        if text_hash in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.access_order.remove(text_hash)
            self.access_order.append(text_hash)
            return self.cache[text_hash]

        self.misses += 1
        return None

    def set(self, text: str, embedding):
        """Add embedding to cache."""
        text_hash = hashlib.md5(text.encode()).hexdigest()

        # Add to cache with LRU eviction
        if len(self.cache) >= self.max_size and text_hash not in self.cache:
            # Remove least recently used
            lru_hash = self.access_order.pop(0)
            del self.cache[lru_hash]

        self.cache[text_hash] = embedding
        if text_hash in self.access_order:
            self.access_order.remove(text_hash)
        self.access_order.append(text_hash)

    def get_stats(self):
        """Return cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "size": len(self.cache),
        }


def detect_source_type(file_path: str) -> str:
    """Detect source type from file extension."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return "pdf"
    elif ext in [".md", ".markdown", ".mdx"]:
        return "markdown"
    elif ext in [
        ".py",
        ".js",
        ".ts",
        ".java",
        ".go",
        ".rs",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
    ]:
        return "code"
    elif ext in [".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf"]:
        return "config"
    elif ext in [".txt", ".text"]:
        return "text"
    elif ext in [".html", ".htm", ".xml"]:
        return "markup"
    elif ext in [".css", ".scss", ".sass", ".less"]:
        return "stylesheet"
    elif ext in [".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd"]:
        return "script"
    elif ext in [".sql"]:
        return "database"
    else:
        return "document"


def calculate_quality_score(chunk: Document) -> float:
    """
    Calculate quality score (0-1) based on content characteristics.
    Higher scores indicate more valuable content for retrieval.
    """
    content = chunk.page_content
    score_components = []

    # 1. Information density (30% weight)
    # Ratio of non-whitespace to total characters
    density = len(content.strip()) / max(len(content), 1)
    score_components.append(density * 0.3)

    # 2. Optimal length (30% weight)
    # Best: 500-2000 chars, penalty for too short or too long
    length = len(content)
    if length < 100:
        length_score = length / 500  # Linear penalty for very short
    elif length <= 2000:
        length_score = 1.0  # Optimal range
    else:
        # Gradual penalty for being too long
        length_score = max(0.5, 1.0 - (length - 2000) / 5000)
    score_components.append(length_score * 0.3)

    # 3. Content type bonus (20% weight)
    content_lower = content.lower()
    if (
        "```" in content
        or "def " in content
        or "function " in content
        or "class " in content
    ):
        content_type_score = 1.0  # Code blocks
    elif "##" in content or "**" in content:
        content_type_score = 0.9  # Formatted markdown
    elif "." in content and len(content.split(".")) > 2:
        content_type_score = 0.8  # Prose with sentences
    else:
        content_type_score = 0.6  # Plain text
    score_components.append(content_type_score * 0.2)

    # 4. Keyword richness (20% weight)
    # Technical terms and documentation keywords
    tech_keywords = [
        "api",
        "function",
        "class",
        "method",
        "parameter",
        "return",
        "example",
        "usage",
        "config",
        "install",
        "import",
        "export",
        "interface",
        "implementation",
    ]
    keyword_count = sum(1 for kw in tech_keywords if kw in content_lower)
    keyword_score = min(1.0, keyword_count / 3)  # Cap at 3 keywords
    score_components.append(keyword_score * 0.2)

    return round(sum(score_components), 2)


def get_metadata_structure(config: Dict[str, Any]) -> str:
    """
    Return the configured payload metadata layout.

    `payload.metadata_structure` is the canonical location. The retrieval section
    fallback preserves compatibility with older example configs that placed this
    option under `retrieval`.
    """
    payload_value = config.get("payload", {}).get("metadata_structure")
    if payload_value in {"nested", "flat"}:
        return payload_value

    retrieval_value = config.get("retrieval", {}).get("metadata_structure")
    if retrieval_value in {"nested", "flat"}:
        return retrieval_value

    return "nested"


def payload_field_path(name: str, metadata_structure: str) -> str:
    """Map a logical metadata field to the configured Qdrant payload path."""
    return f"metadata.{name}" if metadata_structure == "nested" else name


def marker_exclusion_filter(metadata_structure: str) -> qdrant_models.Filter:
    """Build a filter that excludes internal incremental-sync marker points."""
    return qdrant_models.Filter(
        must_not=[
            qdrant_models.FieldCondition(
                key=payload_field_path("record_type", metadata_structure),
                match=qdrant_models.MatchValue(value="file_marker"),
            )
        ]
    )


def is_excluded_path(path: str, root: str, exclude_patterns: List[str]) -> bool:
    """
    Return whether a file or directory path matches configured exclude patterns.

    Supports basename globs such as `*.pyc`, path globs such as `docs/generated/*`,
    and directory segment names such as `node_modules`.
    """
    try:
        relative_path = os.path.relpath(path, root)
    except ValueError:
        relative_path = path

    normalized = relative_path.replace("\\", "/").strip("/")
    basename = os.path.basename(normalized)
    parts = [part for part in normalized.split("/") if part and part != "."]

    for raw_pattern in exclude_patterns:
        pattern = str(raw_pattern).replace("\\", "/").strip("/")
        if not pattern:
            continue

        if any(char in pattern for char in "*?[]"):
            if fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(
                basename, pattern
            ):
                return True
            if "/" not in pattern and any(
                fnmatch.fnmatch(part, pattern) for part in parts
            ):
                return True
            continue

        if (
            pattern in parts
            or normalized == pattern
            or normalized.startswith(f"{pattern}/")
        ):
            return True

    return False


def is_likely_text_file(
    path: str, sample_bytes: int = TEXT_DETECTION_SAMPLE_BYTES
) -> bool:
    """Return whether a file looks like readable text based on a small byte sample."""
    try:
        with open(path, "rb") as file_handle:
            sample = file_handle.read(sample_bytes)
    except OSError:
        return False

    if not sample:
        return True
    if b"\0" in sample:
        return False

    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        pass

    allowed_controls = {7, 8, 9, 10, 12, 13, 27}
    control_bytes = sum(
        1 for byte in sample if byte < 32 and byte not in allowed_controls
    )
    return (control_bytes / len(sample)) < 0.05


def create_payload(
    chunk: Document,
    config: Dict[str, Any],
    chunk_index: int,
    repo_name: str,
    file_path: str,
) -> Dict[str, Any]:
    """
    Create optimized payload with configurable content fields and metadata structure.

    Supports both nested (LangChain default) and flat (v0.3.2) metadata structures.
    """

    # Get content field configuration
    payload_config = config.get("payload", {})
    content_fields = payload_config.get("content_fields", ["content", "page_content"])
    preview_length = payload_config.get("preview_length", 200)
    minimal_mode = payload_config.get("minimal_mode", False)
    metadata_structure = get_metadata_structure(config)
    metadata_allowlist = payload_config.get("metadata_allowlist")
    metadata_denylist = payload_config.get("metadata_denylist", [])

    # Apply minimal mode if enabled
    if minimal_mode and len(content_fields) > 2:
        content_fields = content_fields[:2]

    # Create preview snippet
    preview = chunk.page_content[:preview_length]
    if len(chunk.page_content) > preview_length:
        # Try to cut at word boundary
        last_space = preview.rfind(" ")
        if last_space > preview_length * 0.8:  # Only cut at word if not losing too much
            preview = preview[:last_space] + "..."
        else:
            preview = preview + "..."

    # Build payload with configurable content fields (always at root level)
    payload = {}

    # Add content to all configured fields
    for field_name in content_fields:
        payload[field_name] = chunk.page_content

    # Get embedding provider and model information
    embedding_provider = config.get("embedding_provider", "unknown")
    embedding_model = "unknown"
    if embedding_provider == "azure_openai":
        embedding_model = config.get("azure_openai", {}).get("model", "unknown")
    elif embedding_provider == "mistral_ai":
        embedding_model = config.get("mistral_ai", {}).get("model", "unknown")
    elif embedding_provider == "sentence_transformers":
        embedding_model = config.get("sentence_transformers", {}).get(
            "model", "unknown"
        )

    # Build metadata dictionary
    metadata_dict = {
        # Identifiers
        "doc_id": f"{repo_name}_{os.path.basename(file_path)}_{chunk_index}",
        "chunk_id": chunk_index,
        # Source information
        "source": chunk.metadata.get("source", file_path),
        "source_type": detect_source_type(file_path),
        "repository": chunk.metadata.get("repository", repo_name),
        "name": chunk.metadata.get("name", ""),
        "url": chunk.metadata.get("url", ""),
        "branch": chunk.metadata.get("branch", "main"),
        # Content metrics
        "preview": preview,
        "chunk_size": len(chunk.page_content),
        "token_count": len(chunk.page_content.split()),
        "quality_score": calculate_quality_score(chunk),
        # Processing metadata
        "processed_at": datetime.now().isoformat(),
        # Chunk hash (SHA-256). Keep short display hash for readability.
        "content_hash": hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest(),
        "content_hash_short": hashlib.sha256(
            chunk.page_content.encode("utf-8")
        ).hexdigest()[:12],
        "extraction_method": chunk.metadata.get("extraction_method", "default"),
        # Embedding information
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
    }

    # Add PDF-specific metadata if applicable
    if chunk.metadata.get("page"):
        metadata_dict["page_number"] = chunk.metadata.get("page")
        metadata_dict["total_pages"] = chunk.metadata.get("total_pages")

    # Add any additional metadata that's not already included
    for key, value in chunk.metadata.items():
        if key in metadata_dict:
            continue
        if key in [
            "page_content",
            "content",
            "text",
            "document",
        ]:
            continue

        # Optional schema control
        if isinstance(metadata_allowlist, list) and key not in metadata_allowlist:
            continue
        if isinstance(metadata_denylist, list) and key in metadata_denylist:
            continue

        metadata_dict[key] = value

    # Apply metadata structure: nested (LangChain) or flat
    if metadata_structure == "nested":
        # Nested structure: metadata under "metadata" key (LangChain default)
        payload["metadata"] = metadata_dict
    else:
        # Flat structure: all fields at root level (v0.3.2 behavior)
        payload.update(metadata_dict)

    return payload


@dataclass
class RepositoryConfig:
    """Configuration for a single repository to process."""

    url: str
    branch: Optional[str] = None
    collection_name: Optional[str] = None
    name: Optional[str] = None


@dataclass
class ProcessingResult:
    """Result of processing a single repository."""

    repo_url: str
    collection_name: str
    status: str  # 'success' or 'failed'
    error: Optional[str] = None
    chunks_created: int = 0
    files_processed: int = 0
    processing_time: float = 0.0


@dataclass
class UploadStats:
    """Exact chunk upload statistics for one ingestion operation."""

    original_chunks: int = 0
    unique_chunks: int = 0
    uploaded_chunks: int = 0
    deduplicated_chunks: int = 0


# Mistral AI imports (optional)
try:
    from mistralai import Mistral

    MISTRAL_AVAILABLE = True
except ImportError:
    Mistral = None
    MISTRAL_AVAILABLE = False

# Sentence Transformers imports (optional)
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class ConfigLoader:
    """
    Configuration loader supporting YAML (and JSON) formats with environment variable substitution.
    Primary format is YAML for better readability and environment variable support.
    """

    @staticmethod
    def load_env_for_config(config_path: str, quiet: bool = False) -> None:
        """Load .env next to the config file, falling back to the cwd."""
        config_env = Path(config_path).expanduser().resolve().parent / ".env"
        if config_env.exists():
            load_dotenv(config_env)
            if not quiet:
                print(f"📋 Loaded environment variables from: {config_env}")
        elif os.path.exists(".env"):
            load_dotenv(".env")
            if not quiet:
                print("📋 Loaded environment variables from .env file")

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file with environment variable support.
        JSON format is also supported for backward compatibility.

        Args:
            config_path: Path to configuration file (preferably .yaml)

        Returns:
            Configuration dictionary with environment variables resolved
        """
        ConfigLoader.load_env_for_config(config_path)

        # Determine file format
        file_extension = os.path.splitext(config_path)[1].lower()

        # Load configuration
        with open(config_path, "r") as f:
            if file_extension in [".yaml", ".yml"]:
                config = yaml.safe_load(f)
                print(f"📋 Configuration loaded from: {config_path} (YAML)")
            elif file_extension == ".json":
                config = json.load(f)
                print(f"📋 Configuration loaded from: {config_path} (JSON)")
            else:
                raise ValueError(f"Unsupported configuration format: {file_extension}")

        # Resolve environment variables
        config = ConfigLoader._resolve_env_vars(config)

        return config

    @staticmethod
    def _resolve_env_vars(obj: Any) -> Any:
        """
        Recursively resolve environment variables in configuration.

        Supports formats:
        - ${VAR_NAME} - Basic substitution
        - ${VAR_NAME:-default} - With default value

        Args:
            obj: Configuration object (dict, list, or string)

        Returns:
            Configuration with environment variables resolved
        """
        if isinstance(obj, dict):
            return {k: ConfigLoader._resolve_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ConfigLoader._resolve_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Pattern to match ${VAR} or ${VAR:-default}
            pattern = r"\$\{([^}]+)\}"

            def replacer(match):
                var_expr = match.group(1)
                # Check for default value syntax
                if ":-" in var_expr:
                    var_name, default_value = var_expr.split(":-", 1)
                    return os.getenv(var_name, default_value)
                else:
                    value = os.getenv(var_expr)
                    if value is None:
                        # Keep original if env var not found (for backward compatibility)
                        return match.group(0)
                    return value

            return re.sub(pattern, replacer, obj)
        else:
            return obj


def _resolve_placeholder(value: Any, env_name: str) -> Any:
    """Resolve unresolved ${VAR} placeholders left in optional config values."""
    if isinstance(value, str) and value.startswith("${"):
        return os.environ.get(env_name)
    return value


def create_qdrant_client_from_config(
    qdrant_config: Dict[str, Any],
    *,
    test_connection: bool = False,
    log_status: bool = False,
    client_class: Any = QdrantClient,
) -> QdrantClient:
    """
    Create a Qdrant client from this project's config conventions.

    Used by ingestion and retrieval so connection parsing does not drift.
    """
    connection_method = qdrant_config.get("connection_method", "auto")
    raw_url = qdrant_config.get("url") or os.environ.get("QDRANT_URL")
    url = _resolve_placeholder(raw_url, "QDRANT_URL")
    if isinstance(raw_url, str) and raw_url.startswith("${") and not url:
        raise ValueError(
            "Qdrant URL is not configured. Set QDRANT_URL or replace qdrant.url "
            "with a concrete URL."
        )
    api_key = _resolve_placeholder(
        qdrant_config.get("api_key") or os.environ.get("QDRANT_API_KEY"),
        "QDRANT_API_KEY",
    )
    timeout = qdrant_config.get("timeout", 30)

    hostname = qdrant_config.get("host", "")
    port = qdrant_config.get("port")
    use_https = False
    default_port = 6333

    if isinstance(url, str) and url:
        if url.startswith(("https://", "http://")):
            parsed = urlparse(url)
            hostname = hostname or parsed.hostname or ""
            use_https = parsed.scheme == "https"
            if parsed.port:
                default_port = parsed.port
        elif ":" in url and not hostname:
            hostname, port_s = url.split(":", 1)
            default_port = int(port_s)
        else:
            hostname = hostname or url

    hostname = hostname or "localhost"
    port = int(port or default_port)

    common_kwargs = {
        "api_key": api_key,
        "timeout": timeout,
        "cloud_inference": bool(qdrant_config.get("cloud_inference", False)),
    }
    if qdrant_config.get("local_inference_batch_size") is not None:
        common_kwargs["local_inference_batch_size"] = qdrant_config[
            "local_inference_batch_size"
        ]

    attempts = []
    if connection_method == "auto":
        if log_status:
            print("🔍 Auto-detecting Qdrant connection method...")
        if use_https:
            attempts.append(
                (
                    "reverse_proxy",
                    lambda: client_class(
                        host=hostname,
                        port=443,
                        https=True,
                        prefer_grpc=False,
                        **common_kwargs,
                    ),
                )
            )
        attempts.append(
            (
                "direct",
                lambda: client_class(
                    host=hostname,
                    port=port,
                    https=use_https,
                    **common_kwargs,
                ),
            )
        )
        if url:
            attempts.append(
                (
                    "url",
                    lambda: client_class(
                        url=url,
                        prefer_grpc=False,
                        **common_kwargs,
                    ),
                )
            )
    elif connection_method == "reverse_proxy":
        attempts.append(
            (
                "reverse_proxy",
                lambda: client_class(
                    host=hostname,
                    port=443,
                    https=True,
                    prefer_grpc=False,
                    **common_kwargs,
                ),
            )
        )
    elif connection_method == "direct":
        attempts.append(
            (
                "direct",
                lambda: client_class(
                    host=hostname,
                    port=port,
                    https=use_https,
                    **common_kwargs,
                ),
            )
        )
    elif connection_method == "url":
        if not url:
            raise ValueError("qdrant.url is required when connection_method is 'url'")
        attempts.append(
            (
                "url",
                lambda: client_class(url=url, prefer_grpc=False, **common_kwargs),
            )
        )
    else:
        raise ValueError(f"Unknown connection_method: {connection_method}")

    last_error = None
    for method_name, client_factory in attempts:
        try:
            client = client_factory()
            if not test_connection:
                return client
            client.get_collections()
            if log_status:
                print(f"  ✓ Connected using {method_name}")
                if connection_method == "auto":
                    print(
                        f'💡 Add "connection_method": "{method_name}" to config for faster startup'
                    )
            return client
        except Exception as e:
            last_error = e
            if log_status and connection_method != "auto":
                print(f"  ✗ Failed with {method_name}: {str(e)[:60]}")

    raise ConnectionError(
        f"Failed to connect to Qdrant.\n"
        f"Connection method: {connection_method}\n"
        f"URL: {url}\n"
        f"Last error: {last_error}\n"
        f"Try setting 'connection_method' to 'reverse_proxy', 'direct', or 'url'"
    )


class MistralEmbeddingClient:
    """
    Mistral AI embedding client with batch processing support.

    This class provides a unified interface for generating embeddings using Mistral AI's
    embedding models (mistral-embed, codestral-embed). It handles API authentication,
    request formatting, and supports configurable output dimensions for codestral-embed.
    """

    def __init__(
        self, api_key: str, model: str = "codestral-embed", dimensions: int = 1536
    ):
        """
        Initialize Mistral AI embedding client.

        Args:
            api_key: Mistral AI API key for authentication
            model: Embedding model name (mistral-embed, codestral-embed)
            dimensions: Output vector dimensions for codestral-embed (ignored for mistral-embed)
        """
        if not MISTRAL_AVAILABLE or Mistral is None:
            raise ImportError(
                "Mistral AI library not available. Install with: pip install mistralai"
            )

        self.client = Mistral(api_key=api_key)
        self.model = model
        self.dimensions = dimensions if model == "codestral-embed" else None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        try:
            # Prepare parameters
            params = {"model": self.model, "inputs": texts}
            if self.dimensions and self.model == "codestral-embed":
                params["output_dimension"] = self.dimensions

            response = self.client.embeddings.create(**params)
            return [
                embedding.embedding
                for embedding in response.data
                if embedding.embedding is not None
            ]

        except Exception as e:
            raise Exception(f"Mistral AI embedding error: {e}") from e

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        return self.embed_documents([text])[0]


class SentenceTransformerClient:
    """
    Sentence Transformers embedding client with batch processing support.

    This class provides a unified interface for generating embeddings using
    Sentence Transformers models like all-MiniLM-L6-v2 (384d) and
    multilingual-e5-large (1024d). It handles model loading and provides
    consistent embed_documents() and embed_query() methods.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimensions: Optional[int] = None,
    ):
        """
        Initialize Sentence Transformers embedding client.

        Args:
            model_name: Model name (e.g., 'sentence-transformers/all-MiniLM-L6-v2',
                       'intfloat/multilingual-e5-large')
            dimensions: Optional expected dimensions (for validation). Auto-detected from model if not specified.
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE or SentenceTransformer is None:
            raise ImportError(
                "Sentence Transformers library not available. Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        # Get embedding dimension from model
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.dimensions = dimensions or self.embedding_dim

        # Validate if dimensions were specified
        if dimensions and dimensions != self.embedding_dim:
            logging.warning(
                f"Specified dimensions ({dimensions}) doesn't match model's native dimensions ({self.embedding_dim}). Using model's native dimensions."
            )
            self.dimensions = self.embedding_dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        try:
            # Handle multilingual-e5 models that benefit from prefixing
            if "multilingual-e5" in self.model_name.lower():
                # Add "passage:" prefix for better performance with e5 models
                prefixed_texts = [f"passage: {text}" for text in texts]
                embeddings = self.model.encode(prefixed_texts, convert_to_numpy=False)
            else:
                embeddings = self.model.encode(texts, convert_to_numpy=False)

            # Convert numpy arrays/tensors to lists of floats
            result = []
            if hasattr(embeddings, "tolist"):
                return embeddings.tolist()

            # Handle list of embeddings
            for emb in embeddings:
                if hasattr(emb, "tolist"):
                    result.append(emb.tolist())
                elif hasattr(emb, "__iter__"):
                    result.append([float(x) for x in emb])
                else:
                    result.append(emb)
            return result

        except Exception as e:
            raise Exception(f"Sentence Transformers embedding error: {e}") from e

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        try:
            # Handle multilingual-e5 models that benefit from prefixing
            if "multilingual-e5" in self.model_name.lower():
                # Add "query:" prefix for better performance with e5 models
                prefixed_text = f"query: {text}"
                embedding = self.model.encode([prefixed_text], convert_to_numpy=False)[
                    0
                ]
            else:
                embedding = self.model.encode([text], convert_to_numpy=False)[0]

            # Convert numpy array/tensor to list of floats
            if hasattr(embedding, "tolist"):
                return embedding.tolist()
            elif hasattr(embedding, "__iter__"):
                return [float(x) for x in embedding]
            else:
                # Fallback: convert single value to list
                return [float(embedding)]

        except Exception as e:
            raise Exception(f"Sentence Transformers embedding error: {e}") from e


class GitHubToQdrantProcessor:
    """
    Main processor class for converting GitHub repositories to Qdrant vector collections.

    This class orchestrates the entire pipeline: clones GitHub repositories, extracts
    text files (markdown or all text types), combines them into structured documents, generates embeddings using
    Azure OpenAI, Mistral AI, or Sentence Transformers, performs deduplication, and uploads to Qdrant.

    Key features:
    - Supports both Azure OpenAI and Mistral AI embedding providers
    - Advanced deduplication using content hashing and semantic similarity
    - Configurable text chunking with markdown-aware splitting
    - Rate limiting and retry logic for API calls
    - Folder-based document organization
    """

    def __init__(
        self,
        config_path: str,
        progress: Optional[IngestProgressCallback] = None,
    ):
        """Initialize the processor with configuration."""
        self.progress = progress
        self._emit_progress("initialize", "Loading configuration")
        print("🚀 GitHub to Qdrant Vector Database Processor")
        print("=" * 60)

        self.config = self._load_config(config_path)
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        self.embeddings: Union[
            MistralEmbeddingClient, SentenceTransformerClient, AzureOpenAIEmbeddings
        ]

        print(f"🎯 Target collection: {self.config['qdrant']['collection_name']}")
        self._emit_progress(
            "initialize",
            f"Target collection: {self.config['qdrant']['collection_name']}",
            collection=self.config["qdrant"]["collection_name"],
        )

        # Initialize embedding cache
        self.embedding_cache = EmbeddingCache(max_size=500)
        print("💾 Embedding cache initialized (max size: 500)")

        # Display embedding provider info
        provider = self.config.get("embedding_provider", "azure_openai")
        if provider == "mistral_ai":
            model_name = self.config["mistral_ai"]["model"]
            print(f"🤖 Using embedding provider: Mistral AI ({model_name})")
            provider_label = "Mistral AI"
        elif provider == "sentence_transformers":
            model_name = self.config["sentence_transformers"]["model"]
            print(f"🤖 Using embedding provider: Sentence Transformers ({model_name})")
            provider_label = "Sentence Transformers"
        else:
            model_name = self.config["azure_openai"]["model"]
            print(f"🤖 Using embedding provider: Azure OpenAI ({model_name})")
            provider_label = "Azure OpenAI"
        self._emit_progress(
            "initialize",
            f"Using {provider_label} embeddings ({model_name})",
            collection=self.config["qdrant"]["collection_name"],
        )

        print(f"📏 Embedding dimension: {self.config['qdrant']['vector_size']}")

        # Show branch info if specified
        branch = self.config["github"].get("branch")
        if branch:
            print(f"🌿 Target branch: {branch}")
        else:
            print("🌿 Target branch: default (main/master)")

        # Initialize clients
        print("\n🔗 Initializing connections...")
        self._emit_progress("connect", "Initializing embedding and Qdrant clients")
        self.embeddings = self._initialize_embeddings()
        self.qdrant_client = self._initialize_qdrant()
        self._test_connections()

        # Initialize text splitter based on strategy
        chunking_strategy = self.config["processing"].get(
            "chunking_strategy", "recursive"
        )

        if chunking_strategy == "semantic":
            # Use semantic chunking with embeddings
            self.text_splitter = SemanticChunker(
                embeddings=self.embeddings,  # type: ignore
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=95,  # 95th percentile for semantic similarity
            )
            print("📝 Semantic text splitter configured with percentile threshold")
            self._emit_progress("chunk", "Semantic text splitter configured")
        elif chunking_strategy in ("token_recursive", "token"):
            # Token-aware recursive splitter (uses tiktoken length function)
            encoding_name = self.config.get("processing", {}).get(
                "tiktoken_encoding", "cl100k_base"
            )
            try:
                encoding = tiktoken.get_encoding(encoding_name)
            except Exception:
                # Fallback to cl100k_base if unknown encoding provided
                encoding = tiktoken.get_encoding("cl100k_base")

            chunk_size_tokens = self.config["processing"].get(
                "chunk_size_tokens", self.config["processing"]["chunk_size"]
            )
            chunk_overlap_tokens = self.config["processing"].get(
                "chunk_overlap_tokens", self.config["processing"]["chunk_overlap"]
            )

            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size_tokens,
                chunk_overlap=chunk_overlap_tokens,
                separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
                length_function=lambda s: len(encoding.encode(s)),
            )
            print(
                f"📝 Token-aware text splitter configured: {chunk_size_tokens} tokens/chunk with {chunk_overlap_tokens} overlap (encoding: {encoding_name})"
            )
            self._emit_progress("chunk", "Token-aware text splitter configured")
        else:
            # Default to recursive character text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config["processing"]["chunk_size"],
                chunk_overlap=self.config["processing"]["chunk_overlap"],
                # Markdown-aware separators
                separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
                length_function=len,
            )
            chunk_size = self.config["processing"]["chunk_size"]
            chunk_overlap = self.config["processing"]["chunk_overlap"]
            print(
                f"📝 Text splitter configured: {chunk_size} chars/chunk with {chunk_overlap} overlap"
            )
            self._emit_progress("chunk", "Recursive text splitter configured")

    def _emit_progress(
        self,
        stage: str,
        message: str,
        current: Optional[int] = None,
        total: Optional[int] = None,
        percent: Optional[float] = None,
        level: str = "info",
        repo: Optional[str] = None,
        collection: Optional[str] = None,
    ) -> None:
        """Emit a structured progress event if a callback is configured."""
        progress = getattr(self, "progress", None)
        if not callable(progress):
            return
        event = IngestProgressEvent(
            stage=stage,
            message=message,
            current=current,
            total=total,
            percent=percent,
            level=level,
            repo=repo,
            collection=collection,
        )
        try:
            progress(event)  # pylint: disable=not-callable
        except IngestCancelled:
            raise
        except Exception as exc:  # pragma: no cover - defensive callback isolation
            logging.getLogger(__name__).debug(
                "Ingest progress callback failed: %s", exc
            )

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file with environment variable support."""
        try:
            return ConfigLoader.load_config(config_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            ) from None
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Invalid configuration file format: {e}") from e

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config["logging"]["level"]),
            format=self.config["logging"]["format"],
        )
        # Suppress verbose HTTP logs
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)

    def _initialize_embeddings(
        self,
    ) -> Union[
        MistralEmbeddingClient, SentenceTransformerClient, AzureOpenAIEmbeddings
    ]:
        """
        Initialize embeddings client based on provider selection.

        Creates either a Mistral AI or Azure OpenAI embeddings client based on the
        'embedding_provider' configuration. This abstraction allows seamless switching
        between providers with the same interface.

        Returns:
            Embedding client instance with embed_documents() and embed_query() methods
        """
        provider = self.config.get("embedding_provider", "azure_openai")

        if provider == "mistral_ai":
            mistral_config = self.config["mistral_ai"]
            return MistralEmbeddingClient(
                api_key=mistral_config["api_key"],
                model=mistral_config["model"],
                dimensions=mistral_config.get("dimensions", 1536),
            )
        elif provider == "sentence_transformers":
            st_config = self.config["sentence_transformers"]
            return SentenceTransformerClient(
                model_name=st_config["model"],
                dimensions=st_config.get("dimensions"),
            )
        else:
            # Default to Azure OpenAI
            azure_config = self.config["azure_openai"]
            embeddings_params = {
                "azure_endpoint": azure_config["endpoint"],
                "api_key": azure_config["api_key"],
                "azure_deployment": azure_config["model"],
                "api_version": azure_config["api_version"],
            }

            # Add dimensions parameter if specified (for dimension reduction)
            if "dimensions" in azure_config:
                embeddings_params["dimensions"] = azure_config["dimensions"]

            return AzureOpenAIEmbeddings(**embeddings_params)

    def _initialize_qdrant(self) -> QdrantClient:
        """
        Initialize Qdrant client with auto-detection and flexible configuration.

        Supports multiple connection methods with auto-detection:
        1. Auto mode: Tries multiple connection methods automatically
        2. Reverse proxy: HTTPS connection through reverse proxy (port 443)
        3. Direct connection: Standard Qdrant port (6333 or custom)
        4. URL mode: Direct URL-based connection

        Config options:
        - connection_method: "auto" (default), "reverse_proxy", "direct", "url"
        - url: Full URL for Qdrant (e.g., "https://qdrant.example.com")
        - host: Hostname for direct connection
        - port: Port for direct connection (default: 6333)

        Returns:
            Configured QdrantClient instance
        """
        return create_qdrant_client_from_config(
            self.config["qdrant"], test_connection=True, log_status=True
        )

    def _test_connections(self) -> None:
        """Test connections to embedding provider and Qdrant."""
        # Test embedding provider connection
        provider = self.config.get("embedding_provider", "azure_openai")
        if provider == "mistral_ai":
            provider_name = "Mistral AI"
        elif provider == "sentence_transformers":
            provider_name = "Sentence Transformers"
        else:
            provider_name = "Azure OpenAI"

        try:
            test_response = self.embeddings.embed_query("test connection")
            print(
                f"✅ Connected to {provider_name}. Embedding dimension: {len(test_response)}"
            )
            self._emit_progress(
                "connect",
                f"Connected to {provider_name}; embedding dimension {len(test_response)}",
            )
        except Exception as e:
            print(f"❌ Failed to connect to {provider_name}: {e}")
            self._emit_progress(
                "connect",
                f"Failed to connect to {provider_name}: {e}",
                level="error",
            )
            raise

        # Test Qdrant connection
        try:
            collections = self.qdrant_client.get_collections()
            print(
                f"✅ Connected to Qdrant. Found {len(collections.collections)} existing collections"
            )
            self._emit_progress(
                "connect",
                f"Connected to Qdrant; found {len(collections.collections)} collections",
            )
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant: {e}")
            self._emit_progress(
                "connect",
                f"Failed to connect to Qdrant: {e}",
                level="error",
            )
            raise

    def _extract_repo_name(self, repo_url: str) -> str:
        """Extract repository name from URL."""
        parsed_url = urlparse(repo_url)
        repo_path = parsed_url.path.strip("/")
        if repo_path.endswith(".git"):
            repo_path = repo_path[:-4]
        return repo_path.split("/")[-1]

    def _clone_repository(self, repo_url: str, temp_dir: str) -> str:
        """
        Clone GitHub repository to temporary directory with authentication support.

        Handles both public and private repositories by injecting GitHub tokens into
        the URL when provided. Supports shallow cloning for performance and specific
        branch targeting to reduce clone size and processing time.

        Args:
            repo_url: GitHub repository URL
            temp_dir: Temporary directory for cloning

        Returns:
            Path to cloned repository
        """
        print(f"\n📦 Cloning repository: {repo_url}")
        self._emit_progress(
            "clone",
            f"Cloning repository {repo_url}",
            repo=repo_url,
            collection=self.config["qdrant"].get("collection_name"),
        )

        clone_path = os.path.join(temp_dir, "repo")

        # Handle authentication for private repositories
        auth_repo_url = repo_url

        # Check if using SSH URL (git@github.com:...)
        if repo_url.startswith("git@github.com:"):
            print("🔑 Using SSH authentication (no token needed)")
            auth_repo_url = repo_url
        else:
            # HTTPS URL - check for token
            token = self.config["github"].get("token")
            # Check if token exists and is not an unresolved placeholder
            if token and not token.startswith("${"):
                # Insert token into URL for private repo access
                from urllib.parse import urlparse

                parsed = urlparse(repo_url)
                if parsed.hostname == "github.com":
                    auth_repo_url = f"https://{token}@github.com{parsed.path}"
                print("🔐 Using GitHub token for HTTPS authentication")
            elif token and token.startswith("${"):
                print("⚠️  No GitHub token configured - using public access")
                print("    For private repos via HTTPS, set GITHUB_TOKEN in .env")
                print("    Or use SSH URL format: git@github.com:owner/repo.git")

        cmd = ["git", "clone"]
        if self.config["github"]["clone_depth"]:
            cmd.extend(["--depth", str(self.config["github"]["clone_depth"])])
            print(
                f"📈 Clone depth: {self.config['github']['clone_depth']} (shallow clone)"
            )

        # Add branch specification if provided
        branch = self.config["github"].get("branch")
        if branch:
            cmd.extend(["--branch", branch])
            print(f"🌿 Target branch: {branch}")

        cmd.extend([auth_repo_url, clone_path])

        try:
            print("⏳ Cloning in progress...")
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✅ Repository cloned successfully")
            self._emit_progress(
                "clone",
                "Repository cloned successfully",
                repo=repo_url,
                collection=self.config["qdrant"].get("collection_name"),
                level="success",
            )
            return clone_path
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to clone repository: {e.stderr}")
            self._emit_progress(
                "clone",
                f"Failed to clone repository: {e.stderr}",
                repo=repo_url,
                collection=self.config["qdrant"].get("collection_name"),
                level="error",
            )
            raise

    def _find_text_files(self, directory: str) -> List[str]:
        """
        Recursively find text files based on configuration mode.

        Searches through directory structure while respecting exclude patterns
        to skip unwanted directories (e.g., node_modules, .git) and files.
        Supports two modes:
        - markdown_only: Only processes markdown files
        - all_text: Processes all text-based files (code, config, docs, etc.)

        Args:
            directory: Root directory to search

        Returns:
            List of paths to discovered text files
        """
        file_mode = self.config["processing"].get("file_mode", "markdown_only")

        if file_mode == "all_text":
            print("\n🔍 Searching for all text-based files...")
            self._emit_progress("scan", "Searching for all text-based files")
            extensions = self.config["processing"].get("text_extensions", [])
            detect_text_content = self.config["processing"].get(
                "detect_text_content", True
            )
            # Also check for files without extensions that are commonly text files
            no_ext_names = [f for f in extensions if not f.startswith(".")]
        else:
            print("\n🔍 Searching for markdown files...")
            self._emit_progress("scan", "Searching for markdown files")
            extensions = self.config["processing"]["markdown_extensions"]
            detect_text_content = False
            no_ext_names = []

        text_files = []
        content_detected_count = 0
        exclude_patterns = self.config["processing"]["exclude_patterns"]

        # Only show first 10 extensions for readability
        ext_display = [e for e in extensions if e.startswith(".")][:10]
        if len([e for e in extensions if e.startswith(".")]) > 10:
            print(f"📝 Looking for extensions: {', '.join(ext_display)}... and more")
        else:
            print(f"📝 Looking for extensions: {', '.join(ext_display)}")
        print(f"🚫 Excluding patterns: {', '.join(exclude_patterns)}")

        for root, dirs, files in os.walk(directory):
            # Remove excluded directories from search
            dirs[:] = [
                d
                for d in dirs
                if not is_excluded_path(
                    os.path.join(root, d), directory, exclude_patterns
                )
            ]

            for file in files:
                file_path = os.path.join(root, file)
                if is_excluded_path(file_path, directory, exclude_patterns):
                    continue
                # Check if file has one of the specified extensions or matches no-extension names
                matches_configured_text_file = any(
                    file.lower().endswith(ext)
                    for ext in extensions
                    if ext.startswith(".")
                ) or (file in no_ext_names)
                if matches_configured_text_file:
                    text_files.append(file_path)
                    continue
                if detect_text_content and is_likely_text_file(file_path):
                    content_detected_count += 1
                    text_files.append(file_path)

        file_type = "text" if file_mode == "all_text" else "markdown"
        if content_detected_count:
            print(
                f"✅ Found {len(text_files)} eligible {file_type} files "
                f"({content_detected_count} detected by content)"
            )
            progress_message = (
                f"Found {len(text_files)} eligible {file_type} files after excludes; "
                f"{content_detected_count} detected by content"
            )
        else:
            print(f"✅ Found {len(text_files)} eligible {file_type} files")
            progress_message = (
                f"Found {len(text_files)} eligible {file_type} files after excludes"
            )
        self._emit_progress(
            "scan",
            progress_message,
            current=len(text_files),
            total=len(text_files),
            level="success" if text_files else "warning",
        )
        if len(text_files) > 0:
            print(f"📊 File size range: {self._get_file_size_stats(text_files)}")
        return text_files

    def _get_file_size_stats(self, file_paths: List[str]) -> str:
        """Get file size statistics for display."""
        sizes = []
        for file_path in file_paths:
            try:
                size = os.path.getsize(file_path)
                sizes.append(size)
            except OSError:
                continue

        if not sizes:
            return "No readable files"

        min_size = min(sizes) / 1024  # KB
        max_size = max(sizes) / 1024  # KB
        total_size = sum(sizes) / 1024 / 1024  # MB

        return f"{min_size:.1f}KB - {max_size:.1f}KB (Total: {total_size:.1f}MB)"

    def _combine_text_files(self, text_files: List[str], repo_name: str) -> str:
        """
        Combine text files into structured documents organized by folder hierarchy.

        Creates multiple output files:
        1. Individual folder-based combined files (e.g., 'api.md', 'guides.md')
        2. Root-level files combined into '{repo_name}_root.md'
        3. Master combined file containing all content with folder sections

        This organization preserves document structure while creating a comprehensive
        searchable corpus. Files are grouped by top-level directories to maintain
        logical content boundaries.

        Args:
            text_files: List of discovered text file paths
            repo_name: Repository name for output file naming

        Returns:
            Complete combined text content string
        """
        file_mode = self.config["processing"].get("file_mode", "markdown_only")
        file_type = "text" if file_mode == "all_text" else "markdown"
        print(f"\n📄 Combining {len(text_files)} {file_type} files by folder...")

        # Create output directory
        output_dir = os.path.join(self.config["output"]["base_directory"], repo_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Output directory: {output_dir}")

        # Group files by top-level folder
        repo_root = os.path.dirname(text_files[0]) if text_files else ""
        # Find the actual repository root by going up until we find .git or reach reasonable depth
        temp_root = repo_root
        for _ in range(10):  # Max 10 levels up
            if (
                os.path.exists(os.path.join(temp_root, ".git"))
                or os.path.basename(temp_root) == "repo"
            ):
                repo_root = temp_root
                break
            temp_root = os.path.dirname(temp_root)

        folder_groups = {}
        root_files = []

        print("📂 Grouping files by top-level folders...")
        for text_file in text_files:
            relative_path = os.path.relpath(text_file, repo_root)
            path_parts = relative_path.split(os.sep)

            if len(path_parts) == 1:
                # File is in root directory
                root_files.append(text_file)
            else:
                # File is in a subdirectory
                top_folder = path_parts[0]
                if top_folder not in folder_groups:
                    folder_groups[top_folder] = []
                folder_groups[top_folder].append(text_file)

        print(f"📊 Found {len(folder_groups)} folders and {len(root_files)} root files")
        for folder, files in folder_groups.items():
            print(f"   📁 {folder}/: {len(files)} files")
        if root_files:
            print(f"   📄 Root level: {len(root_files)} files")

        # Create combined files for each folder
        all_combined_content = []
        all_combined_content.append(
            f"# Combined {file_type.capitalize()} Documentation for {repo_name}\n\n"
        )
        all_combined_content.append(
            f"This document contains all {file_type} files from the repository, organized by folder.\n\n"
        )
        all_combined_content.append(
            f"📊 **Statistics**: {len(text_files)} files from {len(folder_groups)} folders\n\n"
        )
        # Note: Timestamp removed from content to ensure deterministic IDs
        # Timestamp is still available in metadata for tracking
        all_combined_content.append("---\n\n")

        successful_reads = 0
        total_chars = 0

        # Initialize PDF processor once if needed
        pdf_processor = None
        if self.config.get("pdf_processing", {}).get("enabled", False):
            pdf_processor = PDFProcessor(self.config, self.logger)
            print("   📑 PDF processing enabled")

        # Process root files first
        if root_files:
            print(f"\n📝 Processing {len(root_files)} root-level files...")
            root_content = self._combine_files_in_group(
                root_files, "root", repo_root, pdf_processor
            )
            root_file_path = os.path.join(output_dir, f"{repo_name}_root.md")
            with open(root_file_path, "w", encoding="utf-8") as f:
                f.write(root_content)

            all_combined_content.append("# Root Level Files\n\n")
            all_combined_content.append(root_content)
            all_combined_content.append("\n\n---\n\n")
            successful_reads += len(root_files)
            total_chars += len(root_content)
            print(f"   ✅ Created: {os.path.basename(root_file_path)}")

        # Process each folder in sorted order for consistency
        for folder_name, files in sorted(folder_groups.items()):
            print(f"\n📝 Processing folder '{folder_name}' with {len(files)} files...")
            folder_content = self._combine_files_in_group(
                files, folder_name, repo_root, pdf_processor
            )

            # Save folder-specific combined file
            folder_file_path = os.path.join(output_dir, f"{folder_name}.md")
            with open(folder_file_path, "w", encoding="utf-8") as f:
                f.write(folder_content)

            # Add to overall combined content
            all_combined_content.append(f"# Folder: {folder_name}\n\n")
            all_combined_content.append(folder_content)
            all_combined_content.append("\n\n---\n\n")

            successful_reads += len(files)
            total_chars += len(folder_content)
            print(f"   ✅ Created: {os.path.basename(folder_file_path)}")

        # Write overall combined markdown file
        combined_file_path = os.path.join(
            output_dir, self.config["output"]["combined_filename"]
        )
        with open(combined_file_path, "w", encoding="utf-8") as f:
            f.write("".join(all_combined_content))

        file_size_mb = os.path.getsize(combined_file_path) / (1024 * 1024)
        created_files = (
            len(folder_groups) + (1 if root_files else 0) + 1
        )  # +1 for combined file

        print(f"\n✅ {file_type.capitalize()} combination completed!")
        print("📊 Summary:")
        print(f"   Files processed: {successful_reads}/{len(text_files)}")
        print(f"   Created files: {created_files} (folder files + combined)")
        print(
            f"   Combined file: {os.path.basename(combined_file_path)} ({file_size_mb:.2f}MB)"
        )
        print(f"   Total characters: {total_chars:,}")

        return "".join(all_combined_content)

    def _combine_files_in_group(
        self, files: List[str], group_name: str, repo_root: str, pdf_processor=None
    ) -> str:
        """Combine files within a specific group/folder, including PDF processing."""
        content_parts = []
        content_parts.append(f"## Files in {group_name}\n\n")

        for text_file in sorted(files):
            try:
                relative_path = os.path.relpath(text_file, repo_root)

                # Check if file is a PDF
                if text_file.lower().endswith(".pdf") and pdf_processor:
                    print(f"   📑 Processing PDF: {os.path.basename(text_file)}")
                    pdf_docs = pdf_processor.process_pdf(text_file)

                    if pdf_docs:
                        content_parts.append(f"### File: {relative_path} [PDF]\n\n")
                        # Combine all pages from PDF
                        for doc in pdf_docs:
                            page_num = doc.metadata.get("page", "")
                            if page_num:
                                content_parts.append(f"#### Page {page_num}\n\n")
                            content_parts.append(doc.page_content)
                            content_parts.append("\n\n")
                    else:
                        print(f"   ⚠️  No content extracted from PDF: {text_file}")
                        continue
                else:
                    # Regular text file processing
                    with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
                        file_content = f.read()

                    content_parts.append(f"### File: {relative_path}\n\n")
                    content_parts.append(file_content)
                    content_parts.append("\n\n")

            except Exception as e:
                print(f"⚠️  Warning: Could not read {text_file}: {e}")
                continue

        return "".join(content_parts)

    def _generate_chunk_id(
        self,
        content: str,
        chunk_index: int,
        repo_name: str,
        file_path: Optional[str] = None,  # noqa: ARG002
    ) -> str:
        """
        Generate deterministic UUID for document chunk.

        Creates consistent, reproducible IDs for chunks based on content hash,
        repository name, and optionally the source file path. This ensures that
        re-processing the same repository produces identical chunk IDs for identical
        content, enabling efficient updates and avoiding duplicate entries in Qdrant.

        The ID is based on:
        - Repository name (constant across runs)
        - File path (if provided, ensures file-specific uniqueness)
        - Content hash (ensures content uniqueness)

        This approach ensures the same content always gets the same ID, regardless
        of processing order or when files are added/removed from the repository.

        Args:
            content: Chunk text content
            chunk_index: Sequential chunk number (kept for compatibility but not used)
            repo_name: Repository name for uniqueness
            file_path: Optional source file path for additional uniqueness

        Returns:
            Deterministic UUID string
        """
        # Create deterministic UUID based on content hash
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Create a deterministic UUID from the hash
        namespace = uuid.UUID("12345678-1234-5678-1234-123456789abc")

        # Use file path if provided, otherwise just repo and content
        if file_path:
            # Normalize the file path to ensure consistency
            normalized_path = file_path.replace("\\", "/").strip("/")
            unique_string = f"{repo_name}_{normalized_path}_{content_hash}"
        else:
            # Fallback for backward compatibility
            unique_string = f"{repo_name}_{content_hash}"

        return str(uuid.uuid5(namespace, unique_string))

    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        # Calculate cosine similarity
        dot_product = np.dot(vec1_np, vec2_np)
        norms = np.linalg.norm(vec1_np) * np.linalg.norm(vec2_np)
        return dot_product / norms if norms != 0 else 0

    def _calculate_batch_similarities(
        self, query_embedding: np.ndarray, target_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarities using optimized vectorized operations.

        This is the performance-critical function that enables 5-15x faster deduplication
        compared to traditional approaches. Uses NumPy's vectorized operations to compute
        similarities between one embedding and multiple target embeddings simultaneously,
        rather than individual comparisons in a loop.

        Args:
            query_embedding: Single embedding vector to compare
            target_embeddings: Batch of embeddings to compare against

        Returns:
            Array of cosine similarity scores
        """
        # Normalize embeddings for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        target_norms = target_embeddings / np.linalg.norm(
            target_embeddings, axis=1, keepdims=True
        )

        # Compute similarities using matrix multiplication
        similarities = np.dot(target_norms, query_norm)
        return similarities

    def _calculate_content_hash(self, content: str) -> str:
        """Calculate stable content hash of content for fast duplicate pre-filtering."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _remove_duplicates(
        self,
        chunks: List[Document],
        embeddings: List[List[float]],
        similarity_threshold: float = 0.95,
    ) -> tuple[List[Document], List[List[float]]]:
        """
        High-performance deduplication using two-stage filtering approach.

        Stage 1: Content Hash Pre-filtering
        - Calculates MD5 hashes for instant exact duplicate detection
        - Groups chunks by hash and removes all but first occurrence
        - Eliminates O(n²) comparisons for exact duplicates

        Stage 2: Semantic Similarity Deduplication
        - Uses vectorized cosine similarity calculations
        - Processes chunks in batches to manage memory usage
        - Compares each chunk against previously accepted unique chunks
        - Removes chunks exceeding similarity threshold

        This two-stage approach provides 5-15x performance improvement over
        traditional methods while maintaining high accuracy.

        Args:
            chunks: Document chunks to deduplicate
            embeddings: Corresponding embedding vectors
            similarity_threshold: Cosine similarity threshold for duplicates (0.95 = 95%)

        Returns:
            Tuple of (unique_chunks, unique_embeddings)
        """
        if not chunks or not embeddings:
            return chunks, embeddings

        print(
            f"🔍 Checking for duplicates with similarity threshold: {similarity_threshold}"
        )
        print(f"📊 Processing {len(chunks)} chunks for deduplication...")
        self._emit_progress(
            "dedupe",
            f"Checking {len(chunks)} chunks for duplicates",
            current=0,
            total=len(chunks),
        )

        # Convert to numpy array for faster operations
        embeddings_np = np.array(embeddings)

        # Pre-compute content hashes for fast duplicate detection
        content_hashes = [
            self._calculate_content_hash(chunk.page_content) for chunk in chunks
        ]
        hash_to_indices = {}

        # Group chunks by content hash for exact duplicates
        for i, content_hash in enumerate(content_hashes):
            if content_hash not in hash_to_indices:
                hash_to_indices[content_hash] = []
            hash_to_indices[content_hash].append(i)

        # Find exact duplicates by hash
        exact_duplicates = set()
        for indices in hash_to_indices.values():
            if len(indices) > 1:
                # Keep first occurrence, mark others as duplicates
                exact_duplicates.update(indices[1:])

        print(f"  📋 Found {len(exact_duplicates)} exact duplicates by content hash")

        # Process remaining chunks for similarity-based deduplication
        unique_indices = []
        removed_count = len(exact_duplicates)
        processed_count = 0

        # Batch size for similarity processing (to manage memory)
        batch_size = 100

        for i in range(len(chunks)):
            if i in exact_duplicates:
                continue

            processed_count += 1
            if processed_count % 100 == 0 or processed_count == len(chunks) - len(
                exact_duplicates
            ):
                progress = (
                    processed_count / (len(chunks) - len(exact_duplicates))
                ) * 100
                print(
                    f"  📈 Similarity check progress: {progress:.1f}% ({processed_count}/{len(chunks) - len(exact_duplicates)})"
                )
                self._emit_progress(
                    "dedupe",
                    "Running semantic similarity checks",
                    current=processed_count,
                    total=len(chunks) - len(exact_duplicates),
                    percent=progress,
                )

            is_duplicate = False

            # Check against previously accepted unique chunks in batches
            if unique_indices:
                # Process in batches to avoid memory issues
                for batch_start in range(0, len(unique_indices), batch_size):
                    batch_end = min(batch_start + batch_size, len(unique_indices))
                    batch_indices = unique_indices[batch_start:batch_end]

                    # Get embeddings for this batch
                    batch_embeddings = embeddings_np[batch_indices]

                    # Calculate similarities for entire batch at once
                    similarities = self._calculate_batch_similarities(
                        embeddings_np[i], batch_embeddings
                    )

                    # Check if any similarity exceeds threshold
                    max_similarity_idx = np.argmax(similarities)
                    max_similarity = similarities[max_similarity_idx]

                    if max_similarity >= similarity_threshold:
                        duplicate_idx = batch_indices[max_similarity_idx]
                        print(
                            f"  🚫 Removing duplicate chunk (similarity: {max_similarity:.3f})"
                        )
                        print(
                            f"      Chunk {duplicate_idx + 1} vs {i + 1}: {len(chunks[duplicate_idx].page_content)} vs {len(chunks[i].page_content)} chars"
                        )
                        is_duplicate = True
                        removed_count += 1
                        break

            if not is_duplicate:
                unique_indices.append(i)

        # Filter to unique chunks
        unique_chunks = [chunks[i] for i in unique_indices]
        unique_embeddings = [embeddings[i] for i in unique_indices]

        similarity_removed = removed_count - len(exact_duplicates)
        print(f"✅ Deduplication complete: {len(chunks)} → {len(unique_chunks)} chunks")
        print(
            f"   📊 Removed {len(exact_duplicates)} exact duplicates + {similarity_removed} similarity duplicates"
        )
        self._emit_progress(
            "dedupe",
            f"Deduplication complete: {len(unique_chunks)} unique chunks; {removed_count} removed",
            current=len(chunks),
            total=len(chunks),
            percent=100.0,
            level="success",
        )

        return unique_chunks, unique_embeddings

    def _generate_embeddings_with_retry(
        self, texts: List[str], max_retries: int = 3
    ) -> List[List[float]]:
        """
        Generate embeddings with intelligent retry logic for API rate limiting.

        Implements provider-specific retry strategies:
        - Detects rate limit errors (429, quota exceeded, etc.)
        - Applies exponential backoff with provider-optimized base delays
        - Attempts to extract retry-after values from error messages
        - Uses different wait times for Azure OpenAI (60s) vs Mistral AI (30s)

        This robust error handling ensures processing continues even with
        aggressive rate limits or temporary API issues.

        Args:
            texts: List of text chunks to embed
            max_retries: Maximum number of retry attempts

        Returns:
            List of embedding vectors
        """
        provider = self.config.get("embedding_provider", "azure_openai")

        for attempt in range(max_retries):
            try:
                return self.embeddings.embed_documents(texts)

            except Exception as e:
                error_str = str(e)

                # Check if it's a rate limit error (429) or similar
                is_rate_limit = (
                    "429" in error_str
                    or "rate limit" in error_str.lower()
                    or "quota" in error_str.lower()
                    or "too many requests" in error_str.lower()
                )

                if is_rate_limit:
                    if attempt < max_retries - 1:
                        # Different wait strategies for different providers
                        if provider == "mistral_ai":
                            wait_time = (
                                30  # Mistral AI typically has shorter wait times
                            )
                        else:
                            wait_time = 60  # Azure OpenAI default

                        # Try to extract wait time from error message
                        if "retry after" in error_str.lower():
                            try:
                                import re

                                match = re.search(
                                    r"retry after (\d+)", error_str.lower()
                                )
                                if match:
                                    wait_time = int(match.group(1))
                            except (ValueError, AttributeError):
                                pass

                        # Add exponential backoff
                        backoff_time = wait_time + (2**attempt * 5)

                        provider_name = (
                            "Mistral AI" if provider == "mistral_ai" else "Azure OpenAI"
                        )
                        print(
                            f"⏳ {provider_name} rate limit hit. Waiting {backoff_time} seconds before retry (attempt {attempt + 1}/{max_retries})..."
                        )
                        time.sleep(backoff_time)
                        continue
                    else:
                        print("❌ Max retries exceeded for rate limiting")
                        raise

                # For non-rate limit errors, raise immediately
                print(f"❌ Error generating embeddings: {e}")
                raise

        raise Exception("Failed to generate embeddings after all retries")

    def _build_quantization_config(self):
        """Build optional Qdrant quantization config from `qdrant.quantization`."""
        quant_cfg = self.config.get("qdrant", {}).get("quantization", {})
        if not quant_cfg.get("enabled", False):
            return None

        method = str(quant_cfg.get("method", "turbo")).lower()
        if method != "turbo":
            raise ValueError(
                "Only qdrant.quantization.method='turbo' is supported in v0.5"
            )

        bit_map = {
            "bits1": qdrant_models.TurboQuantBitSize.BITS1,
            "bits1_5": qdrant_models.TurboQuantBitSize.BITS1_5,
            "bits2": qdrant_models.TurboQuantBitSize.BITS2,
            "bits4": qdrant_models.TurboQuantBitSize.BITS4,
        }
        bits = str(quant_cfg.get("bits", "bits4"))
        if bits not in bit_map:
            raise ValueError(
                "qdrant.quantization.bits must be one of: bits4, bits2, bits1_5, bits1"
            )

        return qdrant_models.TurboQuantization(
            turbo=qdrant_models.TurboQuantQuantizationConfig(
                always_ram=bool(quant_cfg.get("always_ram", True)),
                bits=bit_map[bits],
            )
        )

    def _sparse_vector_config(self):
        """Build optional sparse vector config for Qdrant BM25 hybrid retrieval."""
        sparse_cfg = self.config.get("qdrant", {}).get("sparse_vector", {})
        if not sparse_cfg.get("enabled", False):
            return None

        sparse_name = sparse_cfg.get("name", "sparse")
        return {
            sparse_name: qdrant_models.SparseVectorParams(
                modifier=qdrant_models.Modifier.IDF
            )
        }

    def _is_sparse_vector_enabled(self) -> bool:
        return bool(
            self.config.get("qdrant", {}).get("sparse_vector", {}).get("enabled", False)
        )

    def _validate_sparse_vector_setup(self) -> None:
        """Hybrid sparse vectors require named dense vectors for clear query routing."""
        if self._is_sparse_vector_enabled() and not self.config["qdrant"].get(
            "vector_name"
        ):
            raise ValueError(
                "qdrant.sparse_vector.enabled requires qdrant.vector_name to name "
                "the dense vector (for example: vector_name: dense)."
            )
        if self._is_sparse_vector_enabled() and not self.config["qdrant"].get(
            "cloud_inference", False
        ):
            self.logger.warning(
                "Sparse BM25 vectors use Qdrant Document inference. Set "
                "qdrant.cloud_inference=true or install qdrant-client[fastembed] "
                "for local inference."
            )

    def _validate_embeddings(
        self, chunks: List[Document], embeddings: List[List[float]]
    ) -> None:
        """Validate embedding count and vector dimensions before upload."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Embedding count mismatch: {len(chunks)} chunks but "
                f"{len(embeddings)} embeddings"
            )

        expected_size = int(self.config["qdrant"]["vector_size"])
        for index, embedding in enumerate(embeddings):
            if len(embedding) != expected_size:
                raise ValueError(
                    f"Embedding dimension mismatch at chunk {index}: "
                    f"expected {expected_size}, got {len(embedding)}"
                )

    def _embed_chunks(self, chunks: List[Document]) -> List[List[float]]:
        """Generate embeddings for chunks with cache and retry handling."""
        print("🧠 Generating embeddings for all chunks (with rate limit protection)...")
        all_texts = [chunk.page_content for chunk in chunks]
        self._emit_progress(
            "embed",
            f"Generating embeddings for {len(all_texts)} chunks",
            current=0,
            total=len(all_texts),
        )

        embedding_batch_size = self.config["processing"].get("embedding_batch_size", 20)
        batch_delay = self.config["processing"].get("batch_delay_seconds", 1)
        max_retries = self.config["processing"].get("max_retries", 3)
        simulate_partial = self.config.get("processing", {}).get(
            "simulate_partial_upload", False
        )

        all_embeddings = []
        for i in range(0, len(all_texts), embedding_batch_size):
            batch_texts = all_texts[i : i + embedding_batch_size]
            batch_num = (i // embedding_batch_size) + 1
            total_embedding_batches = (
                len(all_texts) + embedding_batch_size - 1
            ) // embedding_batch_size

            print(
                f"  🧠 Processing embedding batch {batch_num}/{total_embedding_batches} "
                f"({len(batch_texts)} chunks)"
            )
            self._emit_progress(
                "embed",
                f"Embedding batch {batch_num}/{total_embedding_batches}",
                current=batch_num,
                total=total_embedding_batches,
            )

            batch_embeddings = []
            texts_to_generate = []
            text_indices = []
            for idx, text in enumerate(batch_texts):
                cached_embedding = self.embedding_cache.get(text)
                if cached_embedding is not None:
                    batch_embeddings.append(cached_embedding)
                else:
                    texts_to_generate.append(text)
                    text_indices.append(idx)

            if texts_to_generate:
                new_embeddings = self._generate_embeddings_with_retry(
                    texts_to_generate, max_retries
                )
                for text, embedding in zip(texts_to_generate, new_embeddings):
                    self.embedding_cache.set(text, embedding)
                for idx, embedding in zip(text_indices, new_embeddings):
                    batch_embeddings.insert(idx, embedding)

            all_embeddings.extend(batch_embeddings)

            if simulate_partial and batch_num == 1:
                raise RuntimeError(
                    "Simulated partial upload (processing.simulate_partial_upload=true)"
                )

            if i + embedding_batch_size < len(all_texts) and texts_to_generate:
                time.sleep(batch_delay)

        self._validate_embeddings(chunks, all_embeddings)
        self._emit_progress(
            "embed",
            f"Generated {len(all_embeddings)} embeddings",
            current=len(all_embeddings),
            total=len(all_texts),
            percent=100.0,
            level="success",
        )
        return all_embeddings

    def _build_point_vector(self, chunk: Document, embedding: List[float]):
        """Build dense-only or dense+sparse vector payload for a Qdrant point."""
        vector_name = self.config["qdrant"].get("vector_name")
        sparse_cfg = self.config.get("qdrant", {}).get("sparse_vector", {})

        if sparse_cfg.get("enabled", False):
            if not vector_name:
                raise ValueError(
                    "Sparse vector ingestion requires qdrant.vector_name for "
                    "the dense vector."
                )
            sparse_name = sparse_cfg.get("name", "sparse")
            sparse_model = sparse_cfg.get("model", "qdrant/bm25")
            return {
                vector_name: embedding,
                sparse_name: qdrant_models.Document(
                    text=chunk.page_content, model=sparse_model
                ),
            }

        if vector_name:
            return {vector_name: embedding}
        return embedding

    def _build_point_id(
        self, chunk: Document, chunk_index: int, repo_name: str, file_aware_ids: bool
    ) -> str:
        """Generate the appropriate deterministic ID for a chunk."""
        file_path = chunk.metadata.get("file_path") or chunk.metadata.get(
            "source", "unknown"
        )
        if file_aware_ids:
            return self._generate_file_aware_chunk_id(
                chunk.page_content, chunk_index, repo_name, file_path
            )
        return self._generate_chunk_id(
            chunk.page_content, chunk_index, repo_name, file_path
        )

    def _upload_chunks(
        self, chunks: List[Document], repo_name: str, *, file_aware_ids: bool
    ) -> UploadStats:
        """Shared embed, deduplicate, and upload pipeline for all ingestion modes."""
        if not chunks:
            return UploadStats()

        print("\n🧠 Processing and uploading chunks to Qdrant...")
        print(f"📝 Processing {len(chunks)} chunks")
        self._emit_progress(
            "chunk",
            f"Preparing {len(chunks)} chunks for upload",
            current=0,
            total=len(chunks),
            collection=self.config["qdrant"].get("collection_name"),
        )

        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks) if chunks else 0
        print(f"📊 Average chunk size: {avg_chunk_size:.0f} characters")

        all_embeddings = self._embed_chunks(chunks)

        if self.config["processing"].get("deduplication_enabled", True):
            print("🔍 Running deduplication analysis...")
            self._emit_progress("dedupe", "Running deduplication analysis")
            similarity_threshold = self.config["processing"].get(
                "similarity_threshold", 0.95
            )
            unique_chunks, unique_embeddings = self._remove_duplicates(
                chunks, all_embeddings, similarity_threshold=similarity_threshold
            )
        else:
            print("ℹ️  Deduplication disabled - using all chunks")
            self._emit_progress(
                "dedupe",
                "Deduplication disabled; using all chunks",
                level="warning",
            )
            unique_chunks, unique_embeddings = chunks, all_embeddings

        if not unique_chunks:
            print("❌ No unique chunks remaining after deduplication!")
            self._emit_progress(
                "dedupe",
                "No unique chunks remaining after deduplication",
                level="error",
            )
            return UploadStats(original_chunks=len(chunks))

        upload_batch_size = int(self.config["qdrant"].get("upload_batch_size", 64))
        collection_name = self.config["qdrant"]["collection_name"]
        total_batches = (
            len(unique_chunks) + upload_batch_size - 1
        ) // upload_batch_size

        print(
            f"🚀 Starting batch upload: {total_batches} batches of "
            f"{upload_batch_size} chunks each"
        )
        self._emit_progress(
            "upload",
            f"Uploading {len(unique_chunks)} unique chunks in {total_batches} batches",
            current=0,
            total=total_batches,
            collection=collection_name,
        )

        successful_uploads = 0
        for i in range(0, len(unique_chunks), upload_batch_size):
            batch_num = i // upload_batch_size + 1
            batch_chunks = unique_chunks[i : i + upload_batch_size]
            batch_embeddings = unique_embeddings[i : i + upload_batch_size]
            points = []

            for j, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                chunk_index = int(chunk.metadata.get("chunk_index", i + j))
                file_path = chunk.metadata.get("file_path") or chunk.metadata.get(
                    "source", "unknown"
                )
                point_id = self._build_point_id(
                    chunk, chunk_index, repo_name, file_aware_ids
                )
                payload = create_payload(
                    chunk=chunk,
                    config=self.config,
                    chunk_index=chunk_index,
                    repo_name=repo_name,
                    file_path=file_path,
                )
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=self._build_point_vector(chunk, embedding),
                        payload=payload,
                    )
                )

            self.qdrant_client.upsert(collection_name=collection_name, points=points)
            successful_uploads += len(batch_chunks)
            print(
                f"  ✅ Uploaded batch {batch_num}/{total_batches} "
                f"({len(batch_chunks)} chunks)"
            )
            self._emit_progress(
                "upload",
                f"Uploaded batch {batch_num}/{total_batches}",
                current=batch_num,
                total=total_batches,
                collection=collection_name,
            )

            if batch_num % 10 == 0 or batch_num == total_batches:
                progress = (successful_uploads / len(unique_chunks)) * 100
                print(
                    f"  📊 Progress: {progress:.0f}% "
                    f"({successful_uploads}/{len(unique_chunks)} unique chunks)"
                )
                self._emit_progress(
                    "upload",
                    f"Uploaded {successful_uploads}/{len(unique_chunks)} unique chunks",
                    current=successful_uploads,
                    total=len(unique_chunks),
                    percent=progress,
                    collection=collection_name,
                )

        duplicate_count = len(chunks) - len(unique_chunks)
        print(
            f"\n✅ Upload completed: {successful_uploads} chunks uploaded to "
            f"collection '{collection_name}'"
        )
        self._emit_progress(
            "upload",
            f"Upload completed: {successful_uploads} chunks in '{collection_name}'",
            current=successful_uploads,
            total=len(unique_chunks),
            percent=100.0,
            level="success",
            collection=collection_name,
        )
        if duplicate_count > 0:
            print(
                f"   📊 Deduplication stats: {duplicate_count} duplicates removed "
                f"from {len(chunks)} original chunks"
            )

        cache_stats = self.embedding_cache.get_stats()
        if cache_stats["hits"] > 0 or cache_stats["misses"] > 0:
            print(
                f"   💾 Cache stats: {cache_stats['hits']} hits, "
                f"{cache_stats['misses']} misses ({cache_stats['hit_rate']} hit rate)"
            )

        return UploadStats(
            original_chunks=len(chunks),
            unique_chunks=len(unique_chunks),
            uploaded_chunks=successful_uploads,
            deduplicated_chunks=duplicate_count,
        )

    def _setup_qdrant_collection(self) -> None:
        """
        Setup or configure Qdrant collection with proper vector parameters.

        Handles collection lifecycle management:
        - Checks for existing collections
        - Optionally recreates collections for fresh starts
        - Creates new collections with specified vector dimensions and distance metrics
        - Configures distance metrics (Cosine, Euclidean, Dot Product)

        The vector size must match the embedding model's output dimension
        (e.g., 3072 for text-embedding-3-large, 1536 for text-embedding-3-small).
        """
        print("\n🏗️  Setting up Qdrant collection...")

        qdrant_config = self.config["qdrant"]
        collection_name = qdrant_config["collection_name"]
        self._emit_progress(
            "collection",
            f"Setting up Qdrant collection '{collection_name}'",
            collection=collection_name,
        )
        self._validate_sparse_vector_setup()
        quantization_config = self._build_quantization_config()
        sparse_vectors_config = self._sparse_vector_config()

        # Check if collection exists
        collections = self.qdrant_client.get_collections()
        collection_exists = any(
            col.name == collection_name for col in collections.collections
        )

        if collection_exists and qdrant_config["recreate_collection"]:
            print(f"🔄 Recreating existing collection: {collection_name}")
            self._emit_progress(
                "collection",
                f"Recreating existing collection '{collection_name}'",
                collection=collection_name,
                level="warning",
            )
            self.qdrant_client.delete_collection(collection_name)
            collection_exists = False

        if not collection_exists:
            print(f"📚 Creating new collection: {collection_name}")
            self._emit_progress(
                "collection",
                f"Creating new collection '{collection_name}'",
                collection=collection_name,
            )
            print(f"   Vector size: {qdrant_config['vector_size']}")
            print(f"   Distance metric: {qdrant_config['distance']}")
            if quantization_config:
                quant_cfg = qdrant_config.get("quantization", {})
                print(f"   Quantization: TurboQuant ({quant_cfg.get('bits', 'bits4')})")
            if sparse_vectors_config:
                print(
                    "   Sparse vector: "
                    f"{qdrant_config['sparse_vector'].get('name', 'sparse')}"
                )

            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclidean": Distance.EUCLID,
                "Dot": Distance.DOT,
            }

            # Check if we should create named vectors for MCP compatibility
            vector_name = qdrant_config.get("vector_name")
            if vector_name:
                print(f"   Creating named vector: {vector_name}")
                # Create collection with named vectors
                vectors_config = {
                    vector_name: VectorParams(
                        size=qdrant_config["vector_size"],
                        distance=distance_map.get(
                            qdrant_config["distance"], Distance.COSINE
                        ),
                    )
                }
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=vectors_config,
                    sparse_vectors_config=sparse_vectors_config,
                    quantization_config=quantization_config,
                )
            else:
                print("   Creating default (unnamed) vectors")
                # Create collection with default vectors
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=qdrant_config["vector_size"],
                        distance=distance_map.get(
                            qdrant_config["distance"], Distance.COSINE
                        ),
                    ),
                    sparse_vectors_config=sparse_vectors_config,
                    quantization_config=quantization_config,
                )
            print(f"✅ Collection '{collection_name}' created successfully")
            self._emit_progress(
                "collection",
                f"Collection '{collection_name}' created",
                collection=collection_name,
                level="success",
            )
            # Optional: Create payload indexes for faster filtered queries
            self._ensure_qdrant_payload_indexes(collection_name=collection_name)
        else:
            print(f"📚 Using existing collection: {collection_name}")
            self._emit_progress(
                "collection",
                f"Using existing collection '{collection_name}'",
                collection=collection_name,
            )
            if sparse_vectors_config:
                try:
                    self.qdrant_client.update_collection(
                        collection_name=collection_name,
                        sparse_vectors_config=sparse_vectors_config,
                    )
                    print("   ✅ Ensured sparse vector configuration")
                except Exception as e:
                    self.logger.warning(
                        "Failed to ensure sparse vector config for '%s': %s",
                        collection_name,
                        e,
                    )
            if quantization_config and qdrant_config.get("quantization", {}).get(
                "apply_to_existing_collections", False
            ):
                self.qdrant_client.update_collection(
                    collection_name=collection_name,
                    quantization_config=quantization_config,
                )
                print("   ✅ Applied quantization config to existing collection")
            # Optional: Create payload indexes for faster filtered queries
            if (
                self.config.get("qdrant", {})
                .get("payload_indexes", {})
                .get("apply_to_existing_collections", True)
            ):
                self._ensure_qdrant_payload_indexes(collection_name=collection_name)

    def _ensure_qdrant_payload_indexes(self, collection_name: str) -> None:
        """
        Ensure configured Qdrant payload indexes exist (idempotent).

        Supports both payload layouts depending on `payload.metadata_structure`:
        - nested: fields are under `metadata.*` (e.g. `metadata.repository`)
        - flat: fields are at root (e.g. `repository`)
        """
        qdrant_cfg = self.config.get("qdrant", {})
        idx_cfg = qdrant_cfg.get("payload_indexes", {})
        if not idx_cfg.get("enabled", False):
            return

        metadata_structure = get_metadata_structure(self.config)

        fields = idx_cfg.get("fields", [])
        if not isinstance(fields, list) or not fields:
            return

        type_map: Dict[str, qdrant_models.PayloadSchemaType] = {
            "keyword": qdrant_models.PayloadSchemaType.KEYWORD,
            "integer": qdrant_models.PayloadSchemaType.INTEGER,
            "float": qdrant_models.PayloadSchemaType.FLOAT,
            "bool": qdrant_models.PayloadSchemaType.BOOL,
            "datetime": qdrant_models.PayloadSchemaType.DATETIME,
            "text": qdrant_models.PayloadSchemaType.TEXT,
            "uuid": qdrant_models.PayloadSchemaType.UUID,
        }

        created = 0
        # Best-effort: detect existing indexes from collection payload schema to make this
        # truly idempotent even if the API does not error on duplicates.
        existing_schema: Dict[str, Any] = {}
        try:
            info = self.qdrant_client.get_collection(collection_name=collection_name)
            if getattr(info, "payload_schema", None):
                existing_schema = dict(info.payload_schema)
        except Exception as e:
            # Non-fatal; we'll fall back to attempting create calls below.
            self.logger.debug(
                "Could not read payload schema for '%s': %s", collection_name, e
            )

        for spec in fields:
            if not isinstance(spec, dict):
                continue
            name = spec.get("name")
            type_name = str(spec.get("type", "keyword")).lower()
            if not name:
                continue

            field_path = payload_field_path(str(name), metadata_structure)

            schema = type_map.get(type_name)
            if schema is None:
                self.logger.warning(
                    "Unknown payload index type '%s' for field '%s' (skipping)",
                    type_name,
                    field_path,
                )
                continue

            # Check for type mismatch if index already exists
            if field_path in existing_schema:
                existing_type = existing_schema[field_path]
                if existing_type != schema:
                    self.logger.warning(
                        "Index type mismatch for '%s': existing=%s, requested=%s. "
                        "Recreate collection to change index types.",
                        field_path,
                        existing_type,
                        schema,
                    )
                continue

            try:
                self.qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_path,
                    field_schema=schema,
                    wait=True,
                )
                created += 1
                print(f"   ⚡ Created payload index: {field_path} ({type_name})")
                self._emit_progress(
                    "index",
                    f"Created payload index: {field_path} ({type_name})",
                    collection=collection_name,
                )
            except Exception as e:
                # Idempotency: Qdrant returns an error if index already exists.
                msg = str(e).lower()
                if "already exists" in msg or "alreadyexists" in msg:
                    continue
                self.logger.warning(
                    "Failed to create payload index for '%s': %s", field_path, e
                )

        if created:
            print(f"✅ Payload indexing complete ({created} index(es) created)")
            self._emit_progress(
                "index",
                f"Payload indexing complete ({created} index(es) created)",
                collection=collection_name,
                level="success",
            )

    def _process_files_individually(
        self, text_files: List[str], repo_name: str, repo_root: str
    ) -> int:
        """
        Process each file individually for better context and search quality.

        This method processes files one by one, maintaining file-level metadata
        and creating chunks that preserve document boundaries. This approach
        provides better search relevance compared to combining all documents.

        Args:
            text_files: List of file paths to process
            repo_name: Repository name for metadata
            repo_root: Repository root directory

        Returns:
            Total number of chunks created
        """
        print(f"\n📊 Processing {len(text_files)} files individually...")
        self._emit_progress(
            "files",
            f"Processing {len(text_files)} files individually",
            current=0,
            total=len(text_files),
        )

        all_chunks = []
        all_file_paths = []

        # Initialize PDF processor if needed
        pdf_processor = None
        if self.config.get("pdf_processing", {}).get("enabled", False):
            pdf_processor = PDFProcessor(self.config, self.logger)
            print("   📑 PDF processing enabled")
            self._emit_progress("pdf", "PDF processing enabled")

        # Track which files we actually (re)ingested so we can write "file marker" points
        # after a successful upload. This is more reliable than counting chunks in Qdrant,
        # because deduplication can legitimately reduce the number of stored chunks.
        markers_to_upsert: list[dict[str, Any]] = []

        # Track file_ids for orphaned marker cleanup
        processed_file_ids: set[str] = set()

        # Process each file
        for i, file_path in enumerate(text_files, 1):
            relative_path = os.path.relpath(file_path, repo_root)

            print(f"   📄 [{i}/{len(text_files)}] Processing: {relative_path}")
            self._emit_progress(
                "files",
                f"Processing file: {relative_path}",
                current=i,
                total=len(text_files),
            )

            try:
                repo_url = self.config["github"].get("repository_url", "")
                branch_name = self.config["github"].get("branch", "main")
                repo_id = hashlib.sha256(
                    f"{repo_url}@{branch_name}".encode("utf-8")
                ).hexdigest()
                file_id = hashlib.sha256(
                    f"{repo_id}:{relative_path}".encode("utf-8")
                ).hexdigest()

                # Track this file for orphaned marker cleanup
                processed_file_ids.add(file_id)

                # Read file content
                if file_path.lower().endswith(".pdf") and pdf_processor:
                    pdf_docs = pdf_processor.process_pdf(file_path)
                    if pdf_docs:
                        file_content = "\n\n".join(
                            [doc.page_content for doc in pdf_docs]
                        )
                    else:
                        print(
                            f"      ⚠️  No content extracted from PDF: {relative_path}"
                        )
                        self._emit_progress(
                            "pdf",
                            f"No content extracted from PDF: {relative_path}",
                            current=i,
                            total=len(text_files),
                            level="warning",
                        )
                        continue
                else:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        file_content = f.read()

                if not file_content.strip():
                    print(f"      ⚠️  Empty file: {relative_path}")
                    self._emit_progress(
                        "files",
                        f"Skipping empty file: {relative_path}",
                        current=i,
                        total=len(text_files),
                        level="warning",
                    )
                    continue

                # File hash + upload id (SHA-256) used for incremental sync
                track_changes = self.config["processing"].get(
                    "track_file_changes", False
                )
                file_hash = (
                    self._calculate_file_hash(file_path) if track_changes else ""
                )
                file_upload_id = (
                    hashlib.sha256(f"{file_id}:{file_hash}".encode("utf-8")).hexdigest()
                    if track_changes
                    else ""
                )

                # Create document with file-specific metadata
                doc_metadata = {
                    "source": self.config["github"].get("repository_url", ""),
                    "file_path": relative_path,
                    "repository": repo_name,
                    "repo_id": repo_id,
                    "file_id": file_id,
                    "name": self.config["github"].get("name", ""),
                    "url": "",
                    "branch": self.config["github"].get("branch", "main"),
                    "document_type": self._get_document_type(file_path),
                    "file_size": os.path.getsize(file_path),
                }
                if track_changes:
                    doc_metadata["file_hash"] = file_hash
                    doc_metadata["file_upload_id"] = file_upload_id

                document = Document(
                    page_content=file_content,
                    metadata=doc_metadata,
                )

                # Split document into chunks
                file_chunks = self.text_splitter.split_documents([document])
                expected_total_chunks = len(file_chunks)
                chunking_signature = self._chunking_signature()

                # Incremental sync decision after splitting (lets us validate completeness)
                if track_changes:
                    if self._is_file_unchanged(
                        relative_path=relative_path,
                        repo_name=repo_name,
                        repo_id=repo_id,
                        file_id=file_id,
                        file_upload_id=file_upload_id,
                        file_hash=file_hash,
                        expected_total_chunks=expected_total_chunks,
                        chunking_signature=chunking_signature,
                    ):
                        print(
                            f"   ⏭️  Skipping unchanged file (complete): {relative_path} "
                            f"(chunks={expected_total_chunks})"
                        )
                        self._emit_progress(
                            "files",
                            f"Skipping unchanged file: {relative_path}",
                            current=i,
                            total=len(text_files),
                            level="info",
                        )
                        continue

                    # Remove old points for this file (repo-scoped) before re-uploading
                    self._delete_points_for_file(
                        relative_path=relative_path,
                        repo_id=repo_id,
                        file_id=file_id,
                    )

                # Add chunk-specific metadata
                for j, chunk in enumerate(file_chunks):
                    chunk.metadata["chunk_index"] = j
                    chunk.metadata["total_chunks"] = len(file_chunks)
                    chunk.metadata["file_path"] = relative_path
                    # Parent/child metadata for retrieval-time context expansion
                    chunk.metadata["parent_source"] = relative_path
                    chunk.metadata["chunk_index_within_file"] = j
                    chunk.metadata["parent_id"] = hashlib.sha256(
                        f"{repo_name}:{relative_path}".encode("utf-8")
                    ).hexdigest()
                    # Multi-repo-safe incremental IDs
                    chunk.metadata["repo_id"] = repo_id
                    chunk.metadata["file_id"] = file_id
                    if track_changes:
                        chunk.metadata["file_hash"] = file_hash
                        chunk.metadata["file_upload_id"] = file_upload_id

                all_chunks.extend(file_chunks)
                all_file_paths.extend([relative_path] * len(file_chunks))

                print(f"      ✅ Created {len(file_chunks)} chunks")
                self._emit_progress(
                    "chunk",
                    f"Created {len(file_chunks)} chunks from {relative_path}",
                    current=i,
                    total=len(text_files),
                    level="success",
                )

                if track_changes:
                    markers_to_upsert.append(
                        {
                            "repo_name": repo_name,
                            "branch": branch_name,
                            "relative_path": relative_path,
                            "repo_id": repo_id,
                            "file_id": file_id,
                            "file_upload_id": file_upload_id,
                            "file_hash": file_hash,
                            "expected_total_chunks": expected_total_chunks,
                            "chunking_signature": chunking_signature,
                        }
                    )

            except Exception as e:
                print(f"      ❌ Error processing {relative_path}: {e}")
                self._emit_progress(
                    "files",
                    f"Error processing {relative_path}: {e}",
                    current=i,
                    total=len(text_files),
                    level="error",
                )
                continue

        if not all_chunks:
            print("⚠️  No chunks created from any files")
            self._emit_progress(
                "chunk",
                "No chunks created from any files",
                level="warning",
            )
            return 0

        print(
            f"\n📊 Total chunks created: {len(all_chunks)} from {len(set(all_file_paths))} files"
        )
        self._emit_progress(
            "chunk",
            f"Total chunks created: {len(all_chunks)} from {len(set(all_file_paths))} files",
            current=len(all_chunks),
            total=len(all_chunks),
            percent=100.0,
            level="success",
        )

        # Process and upload chunks with file-aware metadata
        upload_stats = self._upload_chunks_with_file_metadata(all_chunks, repo_name)

        # Mark processed files as "complete" for incremental sync. We only write markers after
        # a successful upload of all chunks (so interrupted runs won't create false "complete" states).
        if track_changes and markers_to_upsert:
            self._upsert_file_markers(markers_to_upsert)

        # Cleanup orphaned markers (files that no longer exist in the repo)
        if track_changes and processed_file_ids:
            # Get repo_id from first processed file (all files in same repo have same repo_id)
            repo_url = self.config["github"].get("repository_url", "")
            branch_name = self.config["github"].get("branch", "main")
            repo_id = hashlib.sha256(
                f"{repo_url}@{branch_name}".encode("utf-8")
            ).hexdigest()
            self._cleanup_orphaned_markers(repo_id, processed_file_ids)

        return upload_stats.uploaded_chunks

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate hash of file content for change detection."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _payload_field(self, name: str) -> str:
        """Map a logical metadata field name to the configured payload layout."""
        return payload_field_path(name, get_metadata_structure(self.config))

    def _chunking_signature(self) -> str:
        """Create a stable signature for chunking configuration (affects chunk boundaries/count)."""
        processing = self.config.get("processing", {})
        # Only include settings that change how chunks are created.
        signature_obj = {
            "chunking_strategy": processing.get("chunking_strategy", "recursive"),
            "chunk_size": processing.get("chunk_size"),
            "chunk_overlap": processing.get("chunk_overlap"),
            "chunk_size_tokens": processing.get("chunk_size_tokens"),
            "chunk_overlap_tokens": processing.get("chunk_overlap_tokens"),
            "tiktoken_encoding": processing.get("tiktoken_encoding"),
            "separators": [
                "\n## ",
                "\n### ",
                "\n#### ",
                "\n\n",
                "\n",
                " ",
                "",
            ],
        }
        return hashlib.sha256(
            json.dumps(signature_obj, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _is_file_unchanged(
        self,
        relative_path: str,
        repo_name: str,
        repo_id: str,
        file_id: str,
        file_upload_id: str,
        file_hash: str,
        expected_total_chunks: int,
        chunking_signature: str,
    ) -> bool:
        """Check if a file is unchanged AND fully uploaded (auto-repairs partial uploads)."""
        collection_name = self.config["qdrant"]["collection_name"]

        repo_id_field = self._payload_field("repo_id")
        file_id_field = self._payload_field("file_id")
        upload_id_field = self._payload_field("file_upload_id")
        marker_type_field = self._payload_field("record_type")
        marker_chunking_sig_field = self._payload_field("chunking_signature")
        marker_file_hash_field = self._payload_field("file_hash")
        legacy_repo_field = self._payload_field("repository")
        legacy_path_field = self._payload_field("file_path")
        legacy_hash_field = self._payload_field("file_hash")

        # Preferred path: rely on a per-file marker written only after a successful upload.
        # This avoids false "partial upload" detection when deduplication reduces stored chunk count.
        marker_flt = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key=repo_id_field, match=qdrant_models.MatchValue(value=repo_id)
                ),
                qdrant_models.FieldCondition(
                    key=file_id_field, match=qdrant_models.MatchValue(value=file_id)
                ),
                qdrant_models.FieldCondition(
                    key=upload_id_field,
                    match=qdrant_models.MatchValue(value=file_upload_id),
                ),
                qdrant_models.FieldCondition(
                    key=marker_type_field,
                    match=qdrant_models.MatchValue(value="file_marker"),
                ),
                qdrant_models.FieldCondition(
                    key=marker_chunking_sig_field,
                    match=qdrant_models.MatchValue(value=chunking_signature),
                ),
                qdrant_models.FieldCondition(
                    key=marker_file_hash_field,
                    match=qdrant_models.MatchValue(value=file_hash),
                ),
            ]
        )

        flt = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key=repo_id_field, match=qdrant_models.MatchValue(value=repo_id)
                ),
                qdrant_models.FieldCondition(
                    key=file_id_field, match=qdrant_models.MatchValue(value=file_id)
                ),
                qdrant_models.FieldCondition(
                    key=upload_id_field,
                    match=qdrant_models.MatchValue(value=file_upload_id),
                ),
            ]
        )

        try:
            # Marker check (fast and reliable)
            marker_points, _ = self.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=marker_flt,
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            if marker_points:
                return True

            # Count up to expected_total_chunks; if fewer exist, treat as partial -> re-upload
            points, _ = self.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=flt,
                limit=max(expected_total_chunks, 1),
                with_payload=False,
                with_vectors=False,
            )
            found = len(points)
            if found >= expected_total_chunks:
                return True

            # Backward-compatible fallback: collections ingested before repo_id/file_id/file_upload_id existed.
            legacy_flt = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key=legacy_repo_field,
                        match=qdrant_models.MatchValue(value=repo_name),
                    ),
                    qdrant_models.FieldCondition(
                        key=legacy_path_field,
                        match=qdrant_models.MatchValue(value=relative_path),
                    ),
                    qdrant_models.FieldCondition(
                        key=legacy_hash_field,
                        match=qdrant_models.MatchValue(value=file_hash),
                    ),
                ]
            )
            legacy_points, _ = self.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=legacy_flt,
                limit=max(expected_total_chunks, 1),
                with_payload=False,
                with_vectors=False,
            )
            return len(legacy_points) >= expected_total_chunks
        except Exception as e:
            # Non-fatal: fall back to reprocessing if we can't confirm unchanged.
            self.logger.debug("File-change check failed for %s: %s", relative_path, e)
            return False

    def _upsert_file_markers(self, markers: list[dict[str, Any]]) -> None:
        """Upsert per-file marker points used for robust incremental sync."""
        if not markers:
            return

        if not self.qdrant_client:
            return

        try:
            collection_name = self.config["qdrant"]["collection_name"]
            vector_size = int(self.config["qdrant"]["vector_size"])
            zero_vector = [0.0] * vector_size

            vector_name = self.config["qdrant"].get("vector_name")
            metadata_structure = get_metadata_structure(self.config)

            # Qdrant point IDs must be UUID or unsigned int.
            namespace = uuid.UUID("12345678-1234-5678-1234-123456789abc")

            points: list[PointStruct] = []
            for m in markers:
                marker_uuid = uuid.uuid5(
                    namespace,
                    f"file_marker:{m['file_upload_id']}:{m['chunking_signature']}",
                )

                marker_metadata = {
                    "record_type": "file_marker",
                    "repository": m["repo_name"],
                    "branch": m.get("branch", ""),
                    "file_path": m["relative_path"],
                    "repo_id": m["repo_id"],
                    "file_id": m["file_id"],
                    "file_upload_id": m["file_upload_id"],
                    "file_hash": m["file_hash"],
                    "expected_total_chunks": int(m.get("expected_total_chunks", 0)),
                    "chunking_signature": m["chunking_signature"],
                    "marked_at": datetime.now(timezone.utc).isoformat(),
                }

                # Match the same payload schema as chunks (nested vs flat)
                payload: dict[str, Any] = {
                    # keep empty content fields so downstream tooling that expects them won't break
                    "content": "",
                    "page_content": "",
                }
                if metadata_structure == "nested":
                    payload["metadata"] = marker_metadata
                else:
                    payload.update(marker_metadata)

                if vector_name:
                    points.append(
                        PointStruct(
                            id=str(marker_uuid),
                            vector={vector_name: zero_vector},
                            payload=payload,
                        )
                    )
                else:
                    points.append(
                        PointStruct(
                            id=str(marker_uuid), vector=zero_vector, payload=payload
                        )
                    )

            # Upsert markers in a single request (cheap)
            self.qdrant_client.upsert(collection_name=collection_name, points=points)
        except Exception as e:
            # Non-fatal but important: markers enable incremental sync
            self.logger.warning("Failed to upsert file markers: %s", e)
            print(
                "⚠️  Warning: Could not save incremental sync markers. "
                "Next run will reprocess all files."
            )

    def _cleanup_orphaned_markers(
        self, repo_id: str, current_file_ids: set[str]
    ) -> None:
        """
        Remove markers for files that no longer exist in the repository.
        This prevents accumulation of stale markers over time.

        Args:
            repo_id: Repository identifier (SHA-256 of repo_url@branch)
            current_file_ids: Set of file_id values for files processed in current run
        """
        if not self.qdrant_client:
            return

        # Only run if explicitly enabled in config
        cleanup_enabled = self.config.get("processing", {}).get(
            "cleanup_orphaned_markers", False
        )
        if not cleanup_enabled:
            return

        try:
            collection_name = self.config["qdrant"]["collection_name"]
            metadata_structure = get_metadata_structure(self.config)

            repo_id_field = self._payload_field("repo_id")
            marker_type_field = self._payload_field("record_type")

            # Query for all markers belonging to this repo
            marker_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key=repo_id_field,
                        match=qdrant_models.MatchValue(value=repo_id),
                    ),
                    qdrant_models.FieldCondition(
                        key=marker_type_field,
                        match=qdrant_models.MatchValue(value="file_marker"),
                    ),
                ]
            )

            # Scroll through all markers (use scroll to handle large collections)
            orphaned_ids = []
            offset = None
            while True:
                points, offset = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    scroll_filter=marker_filter,
                    limit=100,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset,
                )

                for point in points:
                    payload = point.payload or {}
                    meta = (
                        payload.get("metadata", {})
                        if metadata_structure == "nested"
                        else payload
                    )
                    file_id = meta.get("file_id")
                    if file_id and file_id not in current_file_ids:
                        orphaned_ids.append(point.id)

                if offset is None:
                    break

            if orphaned_ids:
                self.logger.info(f"Cleaning up {len(orphaned_ids)} orphaned markers")
                self.qdrant_client.delete(
                    collection_name=collection_name,
                    points_selector=orphaned_ids,
                    wait=True,
                )
                print(f"   🧹 Cleaned up {len(orphaned_ids)} orphaned file markers")
            else:
                self.logger.debug("No orphaned markers found")

        except Exception as e:
            # Non-fatal: cleanup is an optimization
            self.logger.warning("Failed to cleanup orphaned markers: %s", e)

    def _delete_points_for_file(
        self, relative_path: str, repo_id: str, file_id: str
    ) -> None:
        """Delete all points for a file, scoped by repo_id+file_id (multi-repo safe)."""
        collection_name = self.config["qdrant"]["collection_name"]

        repo_id_field = self._payload_field("repo_id")
        file_id_field = self._payload_field("file_id")
        file_path_field = self._payload_field("file_path")

        scoped = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key=repo_id_field, match=qdrant_models.MatchValue(value=repo_id)
                ),
                qdrant_models.FieldCondition(
                    key=file_id_field, match=qdrant_models.MatchValue(value=file_id)
                ),
            ]
        )

        legacy = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key=file_path_field,
                    match=qdrant_models.MatchValue(value=relative_path),
                )
            ]
        )

        try:
            # Preferred: repo-scoped delete
            self.qdrant_client.delete(
                collection_name=collection_name,
                points_selector=scoped,
                wait=True,
            )
        except Exception as e:
            self.logger.warning(
                "Failed deleting scoped points for %s (repo_id=%s): %s",
                relative_path,
                repo_id[:12],
                e,
            )

        # Best-effort cleanup of legacy points that may not have repo_id/file_id
        legacy_cleanup = self.config.get("processing", {}).get(
            "legacy_cleanup_delete_by_file_path", False
        )
        if legacy_cleanup:
            try:
                self.qdrant_client.delete(
                    collection_name=collection_name,
                    points_selector=legacy,
                    wait=True,
                )
            except Exception:
                pass

    def _get_document_type(self, file_path: str) -> str:
        """Determine document type from file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        type_map = {
            ".md": "markdown",
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "cpp",
            ".c": "c",
            ".html": "html",
            ".css": "css",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".pdf": "pdf",
            ".txt": "text",
        }
        return type_map.get(ext, "text")

    def _upload_chunks_with_file_metadata(
        self, chunks: List[Document], repo_name: str
    ) -> UploadStats:
        """
        Upload chunks to Qdrant with file-aware metadata and improved ID generation.

        This method handles embedding generation and upload for chunks that have
        been processed from individual files, maintaining file-level context
        and metadata for better search quality.

        Args:
            chunks: List of document chunks with file metadata
            repo_name: Repository name for ID generation
        """
        return self._upload_chunks(chunks, repo_name, file_aware_ids=True)

    def _generate_file_aware_chunk_id(
        self, content: str, chunk_index: int, repo_name: str, file_path: str
    ) -> str:
        """
        Generate deterministic UUID for document chunk with file awareness.

        This improved version ensures unique IDs for chunks from different files
        even if they have similar content, by incorporating the full file path
        and chunk position within that specific file.

        Args:
            content: Chunk text content
            chunk_index: Chunk index within the file
            repo_name: Repository name
            file_path: Relative file path within repository

        Returns:
            Deterministic UUID string
        """
        # Create deterministic UUID based on content hash and file context
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Create a deterministic UUID from the hash
        namespace = uuid.UUID("12345678-1234-5678-1234-123456789abc")

        # Normalize the file path to ensure consistency
        normalized_path = file_path.replace("\\", "/").strip("/")

        # Include chunk index in the unique string for file-specific positioning
        unique_string = f"{repo_name}_{normalized_path}_{chunk_index}_{content_hash}"

        return str(uuid.uuid5(namespace, unique_string))

    def _process_and_upload_documents(
        self, combined_content: str, repo_name: str
    ) -> UploadStats:
        """
        Process combined content into chunks and upload to Qdrant with comprehensive pipeline.

        Processing Pipeline:
        1. Document Creation: Wraps content with metadata (repo, branch, timestamp)
        2. Text Chunking: Uses RecursiveCharacterTextSplitter with markdown-aware separators
        3. Batch Embedding Generation: Processes chunks in configurable batch sizes
        4. Rate Limit Protection: Implements delays and retry logic
        5. Deduplication: Applies two-stage duplicate removal (optional)
        6. Batch Upload: Uploads to Qdrant in optimized batches

        Each chunk receives comprehensive metadata including:
        - Source repository and branch information
        - Chunk index and content hash for tracking
        - Content preview for debugging
        - Processing timestamps

        Args:
            combined_content: Complete markdown content to process
            repo_name: Repository name for metadata and tracking
        """
        print("\n🧠 Processing and uploading documents to Qdrant...")

        # Create document
        branch = self.config["github"].get("branch", "default")
        document = Document(
            page_content=combined_content,
            metadata={
                "source": self.config["github"].get("repository_url", ""),
                "repository": repo_name,
                "name": self.config["github"].get("name", ""),
                "url": "",
                "branch": branch,
                "document_type": "combined_text",
                "processed_at": datetime.now().isoformat(),
            },
        )

        # Split document into chunks
        print("✂️  Splitting document into chunks...")
        chunks = self.text_splitter.split_documents([document])
        print(f"📝 Split document into {len(chunks)} chunks")

        return self._upload_chunks(chunks, repo_name, file_aware_ids=False)

    def _process_and_upload_documents_with_stats(
        self, combined_content: str, repo_name: str
    ) -> int:
        """
        Process and upload documents, returning the number of chunks created.

        Args:
            combined_content: Combined markdown content to process
            repo_name: Repository name for metadata and tracking

        Returns:
            Number of chunks successfully uploaded
        """
        stats = self._process_and_upload_documents(combined_content, repo_name)
        return stats.uploaded_chunks

    def process_repository_with_override(
        self,
        repo_url: str,
        branch: Optional[str] = None,
        collection_name: Optional[str] = None,
        name: Optional[str] = None,
    ) -> ProcessingResult:
        """
        Process a repository with optional overrides for branch, collection, and name.

        Args:
            repo_url: GitHub repository URL
            branch: Optional branch override
            collection_name: Optional collection name override
            name: Optional human-readable repository name

        Returns:
            ProcessingResult with status and statistics
        """
        start_time = datetime.now()
        result = ProcessingResult(
            repo_url=repo_url,
            collection_name=collection_name or self.config["qdrant"]["collection_name"],
            status="failed",
        )

        # Temporarily override config values if provided
        original_branch = self.config["github"].get("branch")
        original_collection = self.config["qdrant"]["collection_name"]
        original_name = self.config["github"].get("name")
        original_url = self.config["github"].get("repository_url")

        try:
            # Always set the repository_url for metadata
            self.config["github"]["repository_url"] = repo_url

            if branch:
                self.config["github"]["branch"] = branch
            if collection_name:
                self.config["qdrant"]["collection_name"] = collection_name
            if name:
                self.config["github"]["name"] = name

            # Process the repository
            files_processed, chunks_created = self._process_repository_internal(
                repo_url
            )

            # Update result
            result.status = "success"
            result.files_processed = files_processed
            result.chunks_created = chunks_created
            result.processing_time = (datetime.now() - start_time).total_seconds()

        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            result.processing_time = (datetime.now() - start_time).total_seconds()
            raise

        finally:
            # Restore original config values
            if original_branch:
                self.config["github"]["branch"] = original_branch
            elif "branch" in self.config["github"]:
                del self.config["github"]["branch"]

            if original_name:
                self.config["github"]["name"] = original_name
            elif "name" in self.config["github"]:
                del self.config["github"]["name"]

            if original_url:
                self.config["github"]["repository_url"] = original_url
            elif "repository_url" in self.config["github"]:
                del self.config["github"]["repository_url"]

            self.config["qdrant"]["collection_name"] = original_collection

        return result

    def _process_repository_internal(self, repo_url: str) -> tuple[int, int]:
        """
        Internal method to process a repository and return statistics.

        Returns:
            Tuple of (files_processed, chunks_created)
        """
        repo_name = self._extract_repo_name(repo_url)
        print(f"\n🎯 Processing repository: {repo_name}")
        self._emit_progress(
            "repository",
            f"Processing repository {repo_name}",
            repo=repo_url,
            collection=self.config["qdrant"].get("collection_name"),
        )

        files_processed = 0
        chunks_created = 0

        # Create temporary directory for cloning
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Clone repository
                clone_path = self._clone_repository(repo_url, temp_dir)

                # Find text files based on configured mode
                text_files = self._find_text_files(clone_path)
                files_processed = len(text_files)

                if not text_files:
                    file_mode = self.config["processing"].get(
                        "file_mode", "markdown_only"
                    )
                    file_type = "text" if file_mode == "all_text" else "markdown"
                    print(f"⚠️  No {file_type} files found in repository")
                    self._emit_progress(
                        "scan",
                        f"No {file_type} files found in repository",
                        level="warning",
                        repo=repo_url,
                        collection=self.config["qdrant"].get("collection_name"),
                    )
                    return (0, 0)

                # Setup Qdrant collection
                self._setup_qdrant_collection()

                # Check if we should process files individually or combine them
                combine_docs = self.config["processing"].get("combine_documents", True)
                print(f"\n🔍 Debug: combine_documents = {combine_docs}")

                if combine_docs is False:  # Explicitly check for False
                    # Process files individually for better context and search quality
                    print("\n📄 Processing files individually for better context...")
                    self._emit_progress(
                        "files",
                        "Processing files individually for better context",
                    )
                    chunks_created = self._process_files_individually(
                        text_files, repo_name, clone_path
                    )
                else:
                    # Legacy mode: Combine text files into folder-based files + overall combined file
                    print("\n📄 Combining documents (legacy mode)...")
                    self._emit_progress("files", "Combining documents in legacy mode")
                    combined_content = self._combine_text_files(text_files, repo_name)

                    # Process and upload ONLY the final combined document
                    print(
                        "\n🎯 Creating vector embeddings for the combined document only..."
                    )
                    self._emit_progress(
                        "embed",
                        "Creating vector embeddings for combined document",
                    )
                    chunks_created = self._process_and_upload_documents_with_stats(
                        combined_content, repo_name
                    )

                return (files_processed, chunks_created)

            finally:
                if (
                    self.config["github"]["cleanup_after_processing"]
                    and not interrupted
                ):
                    print("🧹 Cleaning up temporary files")
                    self._emit_progress("cleanup", "Cleaning up temporary files")

        return (files_processed, chunks_created)

    def process_repository(self, repo_url: Optional[str] = None) -> None:
        """
        Main orchestration method for complete repository processing pipeline.

        Execution Flow:
        1. Repository cloning to temporary directory
        2. Text file discovery with filtering
        3. Processing based on combine_documents setting
        4. Qdrant collection setup
        5. Document processing and vector upload
        6. Cleanup and reporting

        Uses temporary directory management to ensure clean cleanup even if
        processing fails. Provides comprehensive progress reporting and final
        statistics including processing time and upload counts.

        Args:
            repo_url: Optional repository URL override (uses config if not provided)
        """
        start_time = datetime.now()

        if not repo_url:
            repo_url = self.config["github"]["repository_url"]

        # Type guard to ensure repo_url is not None
        if repo_url is None:
            raise ValueError(
                "Repository URL is required either as parameter or in config"
            )

        # Call the internal method which has the proper logic
        repo_name = self._extract_repo_name(repo_url)
        files_processed, chunks_created = self._process_repository_internal(repo_url)

        # Calculate and display final statistics
        end_time = datetime.now()
        duration = end_time - start_time

        print("\n🎉 Repository processing completed successfully!")
        print("=" * 60)
        print("📊 **Final Summary**")
        print(f"   Repository: {repo_name}")
        branch = self.config["github"].get("branch")
        if branch:
            print(f"   Branch: {branch}")
        print(f"   Files processed: {files_processed}")
        print(f"   Chunks created: {chunks_created}")
        print(f"   Total processing time: {duration.total_seconds():.1f} seconds")
        print(f"   Collection: {self.config['qdrant']['collection_name']}")
        print(f"   Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._emit_progress(
            "complete",
            (
                f"Repository completed: {files_processed} files, "
                f"{chunks_created} chunks in {duration.total_seconds():.1f}s"
            ),
            current=files_processed,
            total=files_processed,
            percent=100.0,
            level="success",
            repo=repo_url,
            collection=self.config["qdrant"].get("collection_name"),
        )


def load_repository_list(repo_list_path: str) -> List[RepositoryConfig]:
    """
    Load and validate repository list from YAML file.

    Args:
        repo_list_path: Path to YAML file containing repository list

    Returns:
        List of RepositoryConfig objects

    Raises:
        ValueError: If file format is invalid or required fields are missing
    """
    print(f"📋 Loading repository list from: {repo_list_path}")

    try:
        with open(repo_list_path, "r") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Repository list file not found: {repo_list_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format in repository list: {e}")

    if not data or "repositories" not in data:
        raise ValueError(
            "Repository list file must contain a 'repositories' key with a list of repositories"
        )

    repositories = data["repositories"]
    if not isinstance(repositories, list):
        raise ValueError("'repositories' must be a list")

    configs = []
    for i, repo in enumerate(repositories, 1):
        if not isinstance(repo, dict):
            raise ValueError(f"Repository {i}: Each repository must be a dictionary")

        if "url" not in repo:
            raise ValueError(f"Repository {i}: 'url' field is required")

        if "collection_name" not in repo:
            raise ValueError(f"Repository {i}: 'collection_name' field is required")

        config = RepositoryConfig(
            url=repo["url"],
            branch=repo.get("branch"),
            collection_name=repo["collection_name"],
            name=repo.get("name"),
        )
        configs.append(config)

    print(f"✅ Loaded {len(configs)} repositories from list")
    return configs


def process_repository_list(
    processor: GitHubToQdrantProcessor, repo_list_path: str
) -> List[ProcessingResult]:
    """
    Process multiple repositories sequentially from a list file.

    Args:
        processor: GitHubToQdrantProcessor instance
        repo_list_path: Path to repository list YAML file

    Returns:
        List of ProcessingResult objects
    """
    repositories = load_repository_list(repo_list_path)
    results = []
    processor._emit_progress(
        "repo-list",
        f"Loaded {len(repositories)} repositories from {repo_list_path}",
        current=0,
        total=len(repositories),
    )

    print("\n" + "=" * 60)
    print("STARTING MULTI-REPOSITORY PROCESSING")
    print(f"Total repositories to process: {len(repositories)}")
    print("=" * 60)

    overall_start_time = datetime.now()

    for i, repo_config in enumerate(repositories, 1):
        print("\n" + "=" * 60)
        print(f"Processing repository {i}/{len(repositories)}")
        print(f"Repository: {repo_config.url}")
        if repo_config.name:
            print(f"Name: {repo_config.name}")
        print(f"Branch: {repo_config.branch or 'default'}")
        print(f"Collection: {repo_config.collection_name}")
        print("=" * 60)
        processor._emit_progress(
            "repo-list",
            f"Processing repository {i}/{len(repositories)}: {repo_config.url}",
            current=i,
            total=len(repositories),
            repo=repo_config.url,
            collection=repo_config.collection_name,
        )

        try:
            result = processor.process_repository_with_override(
                repo_url=repo_config.url,
                branch=repo_config.branch,
                collection_name=repo_config.collection_name,
                name=repo_config.name,
            )
            results.append(result)
            print(f"✅ Successfully processed: {repo_config.url}")
            processor._emit_progress(
                "repo-list",
                f"Successfully processed {repo_config.url}",
                current=i,
                total=len(repositories),
                level="success",
                repo=repo_config.url,
                collection=repo_config.collection_name,
            )

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Failed to process {repo_config.url}: {error_msg}")
            processor._emit_progress(
                "repo-list",
                f"Failed to process {repo_config.url}: {error_msg}",
                current=i,
                total=len(repositories),
                level="error",
                repo=repo_config.url,
                collection=repo_config.collection_name,
            )

            # Create failed result
            result = ProcessingResult(
                repo_url=repo_config.url,
                collection_name=repo_config.collection_name or "default",
                status="failed",
                error=error_msg,
            )
            results.append(result)

            # Continue with next repository
            continue

    overall_duration = datetime.now() - overall_start_time

    # Print summary report
    print_summary_report(results, overall_duration, processor)
    processor._emit_progress(
        "complete",
        (
            f"Repository list completed: "
            f"{sum(1 for result in results if result.status == 'success')}/"
            f"{len(results)} succeeded in {overall_duration.total_seconds():.1f}s"
        ),
        current=len(results),
        total=len(results),
        percent=100.0,
        level=(
            "success"
            if all(result.status == "success" for result in results)
            else "warning"
        ),
    )

    return results


def print_summary_report(
    results: List[ProcessingResult],
    overall_duration,
    processor: GitHubToQdrantProcessor,
):
    """
    Print a comprehensive summary report of multi-repository processing.

    Args:
        results: List of ProcessingResult objects
        overall_duration: Total processing time
        processor: GitHubToQdrantProcessor instance for cache stats
    """
    successful = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status == "failed"]

    print("\n" + "=" * 60)
    print("MULTI-REPOSITORY PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total repositories: {len(results)}")
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    print()

    print("Details:")
    print("-" * 60)

    for result in results:
        repo_name = result.repo_url.split("/")[-1].replace(".git", "")

        if result.status == "success":
            print(f"✅ {repo_name} → {result.collection_name}")
            print(
                f"   Files: {result.files_processed}, Chunks: {result.chunks_created}"
            )
            print(f"   Time: {result.processing_time:.1f}s")
        else:
            print(f"❌ {repo_name} → Failed")
            error_preview = result.error[:60] if result.error else "Unknown error"
            print(f"   Error: {error_preview}")

    print("-" * 60)

    # Calculate totals
    total_files = sum(r.files_processed for r in successful)
    total_chunks = sum(r.chunks_created for r in successful)

    print("\nTotals:")
    print(f"   Files processed: {total_files:,}")
    print(f"   Chunks created: {total_chunks:,}")
    print(f"   Processing time: {overall_duration.total_seconds():.1f}s")

    # Display overall cache statistics
    cache_stats = processor.embedding_cache.get_stats()
    if cache_stats["hits"] > 0 or cache_stats["misses"] > 0:
        print()
        print("Embedding Cache Performance:")
        print(f"   💾 Total hits: {cache_stats['hits']:,}")
        print(f"   💾 Total misses: {cache_stats['misses']:,}")
        print(f"   💾 Hit rate: {cache_stats['hit_rate']}")
        print(
            f"   💾 Cache size: {cache_stats['size']}/{processor.embedding_cache.max_size}"
        )

    if overall_duration.total_seconds() > 60:
        minutes = int(overall_duration.total_seconds() // 60)
        seconds = int(overall_duration.total_seconds() % 60)
        print(f"   ({minutes}m {seconds}s)")

    print("=" * 60)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


# Global flag to track interruption
interrupted = False


def signal_handler(_signum, _frame):
    """Handle interrupt signals gracefully."""
    global interrupted
    interrupted = True
    # Suppress any further output from libraries
    sys.stderr = open(os.devnull, "w")
    sys.stdout.write("\n\n⚠️  Process interrupted by user (Ctrl+C)\n")
    sys.stdout.write("🧹 Cleaning up and exiting...\n")
    sys.stdout.flush()
    sys.exit(130)  # Standard Unix exit code for SIGINT


def run_ingest(
    config_path: str,
    repo_url: Optional[str] = None,
    repo_list: Optional[str] = None,
    progress: Optional[IngestProgressCallback] = None,
) -> int:
    """Run ingestion for one repository or a repository list."""
    # Set up signal handler for clean interruption
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, signal_handler)

    # Load environment variables from .env file
    load_dotenv()

    try:
        processor = GitHubToQdrantProcessor(config_path, progress=progress)

        # Check if repository list is provided
        if repo_list:
            # Process multiple repositories from list
            process_repository_list(processor, repo_list)
        else:
            # Process single repository (current behavior)
            processor.process_repository(repo_url)

    except IngestCancelled as e:
        if progress is not None:
            try:
                progress(
                    IngestProgressEvent(
                        stage="stopped",
                        message=str(e) or "Ingestion stopped",
                        level="warning",
                    )
                )
            except IngestCancelled:
                pass
        return 130
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully without showing traceback
        print("\n\n⚠️  Process interrupted by user (Ctrl+C)")
        print("🧹 Cleaning up and exiting...")
        if progress is not None:
            try:
                progress(
                    IngestProgressEvent(
                        stage="stopped",
                        message="Process interrupted by user",
                        level="warning",
                    )
                )
            except Exception:
                pass
        return 130  # Standard Unix exit code for SIGINT
    except Exception as e:
        logging.error("Script failed: %s", e)
        if progress is not None:
            try:
                progress(
                    IngestProgressEvent(
                        stage="failed",
                        message=f"Ingestion failed: {e}",
                        level="error",
                    )
                )
            except Exception:
                pass
        return 1

    return 0


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Process GitHub repository text files into Qdrant vector database"
    )
    parser.add_argument(
        "config",
        help="Path to configuration file (YAML format recommended, JSON supported)",
    )
    parser.add_argument(
        "--repo-url", help="GitHub repository URL (overrides config file)", default=None
    )
    parser.add_argument(
        "--repo-list",
        help="Path to YAML file containing list of repositories to process",
        default=None,
    )

    args = parser.parse_args()

    return run_ingest(
        config_path=args.config,
        repo_url=args.repo_url,
        repo_list=args.repo_list,
    )


if __name__ == "__main__":
    sys.exit(main())
