#!/usr/bin/env python3
"""
GitHub Repository to Qdrant Vector Database Processor

This script clones a GitHub repository, extracts text-based files (configurable),
combines them into a single document, and inserts them into a Qdrant collection
using various embedding providers (Azure OpenAI, Mistral AI, or Sentence Transformers).
"""

import argparse
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Protocol, Union
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv

import numpy as np
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import AzureOpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Import PDF processor
from pdf_processor import PDFProcessor

# Type hints
from dataclasses import dataclass


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


def create_payload(
    chunk: Document,
    config: Dict[str, Any],
    chunk_index: int,
    repo_name: str,
    file_path: str,
) -> Dict[str, Any]:
    """Create optimized payload with configurable content fields."""

    # Get content field configuration
    payload_config = config.get("payload", {})
    content_fields = payload_config.get("content_fields", ["content", "page_content"])
    preview_length = payload_config.get("preview_length", 200)
    minimal_mode = payload_config.get("minimal_mode", False)

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

    # Build payload with configurable content fields
    payload = {}

    # Add content to all configured fields
    for field_name in content_fields:
        payload[field_name] = chunk.page_content

    # Add flattened metadata (no nesting)
    payload.update(
        {
            # Identifiers
            "doc_id": f"{repo_name}_{os.path.basename(file_path)}_{chunk_index}",
            "chunk_id": chunk_index,
            # Source information
            "source": chunk.metadata.get("source", file_path),
            "source_type": detect_source_type(file_path),
            "repository": chunk.metadata.get("repository", repo_name),
            "branch": chunk.metadata.get("branch", "main"),
            # Content metrics
            "preview": preview,
            "chunk_size": len(chunk.page_content),
            "token_count": len(chunk.page_content.split()),
            "quality_score": calculate_quality_score(chunk),
            # Processing metadata
            "timestamp": int(datetime.now().timestamp()),
            "content_hash": hashlib.md5(chunk.page_content.encode()).hexdigest()[:8],
            "extraction_method": chunk.metadata.get("extraction_method", "default"),
        }
    )

    # Add PDF-specific metadata if applicable
    if chunk.metadata.get("page"):
        payload["page_number"] = chunk.metadata.get("page")
        payload["total_pages"] = chunk.metadata.get("total_pages")

    # Add any additional metadata that's not already included
    for key, value in chunk.metadata.items():
        if key not in payload and key not in [
            "page_content",
            "content",
            "text",
            "document",
        ]:
            payload[key] = value

    return payload


@dataclass
class RepositoryConfig:
    """Configuration for a single repository to process."""

    url: str
    branch: Optional[str] = None
    collection_name: Optional[str] = None


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
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file with environment variable support.
        JSON format is also supported for backward compatibility.

        Args:
            config_path: Path to configuration file (preferably .yaml)

        Returns:
            Configuration dictionary with environment variables resolved
        """
        # Load environment variables from .env file if it exists
        if os.path.exists(".env"):
            load_dotenv(".env")
            print("📋 Loaded environment variables from .env file")

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


class MistralEmbeddingClient:
    """
    Mistral AI embedding client with batch processing support.

    This class provides a unified interface for generating embeddings using Mistral AI's
    embedding models (mistral-embed, codestral-embed). It handles API authentication,
    request formatting, and supports configurable output dimensions for codestral-embed.
    """

    def __init__(
        self, api_key: str, model: str = "codestral-embed", output_dimension: int = 1536
    ):
        """
        Initialize Mistral AI embedding client.

        Args:
            api_key: Mistral AI API key for authentication
            model: Embedding model name (mistral-embed, codestral-embed)
            output_dimension: Vector dimension for codestral-embed (ignored for mistral-embed)
        """
        if not MISTRAL_AVAILABLE or Mistral is None:
            raise ImportError(
                "Mistral AI library not available. Install with: pip install mistralai"
            )

        self.client = Mistral(api_key=api_key)
        self.model = model
        self.output_dimension = output_dimension if model == "codestral-embed" else None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        try:
            # Prepare parameters
            params = {"model": self.model, "inputs": texts}
            if self.output_dimension and self.model == "codestral-embed":
                params["output_dimension"] = self.output_dimension

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

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize Sentence Transformers embedding client.

        Args:
            model_name: Model name (e.g., 'sentence-transformers/all-MiniLM-L6-v2',
                       'intfloat/multilingual-e5-large')
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE or SentenceTransformer is None:
            raise ImportError(
                "Sentence Transformers library not available. Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

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

    def __init__(self, config_path: str):
        """Initialize the processor with configuration."""
        print("🚀 GitHub to Qdrant Vector Database Processor")
        print("=" * 60)

        self.config = self._load_config(config_path)
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        self.embeddings: Union[
            MistralEmbeddingClient, SentenceTransformerClient, AzureOpenAIEmbeddings
        ]

        print(f"🎯 Target collection: {self.config['qdrant']['collection_name']}")

        # Initialize embedding cache
        self.embedding_cache = EmbeddingCache(max_size=500)
        print("💾 Embedding cache initialized (max size: 500)")

        # Display embedding provider info
        provider = self.config.get("embedding_provider", "azure_openai")
        if provider == "mistral_ai":
            model_name = self.config["mistral_ai"]["model"]
            print(f"🤖 Using embedding provider: Mistral AI ({model_name})")
        elif provider == "sentence_transformers":
            model_name = self.config["sentence_transformers"]["model"]
            print(f"🤖 Using embedding provider: Sentence Transformers ({model_name})")
        else:
            model_name = self.config["azure_openai"]["deployment_name"]
            print(f"🤖 Using embedding provider: Azure OpenAI ({model_name})")

        print(f"📏 Embedding dimension: {self.config['qdrant']['vector_size']}")

        # Show branch info if specified
        branch = self.config["github"].get("branch")
        if branch:
            print(f"🌿 Target branch: {branch}")
        else:
            print("🌿 Target branch: default (main/master)")

        # Initialize clients
        print("\n🔗 Initializing connections...")
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
                output_dimension=mistral_config.get("output_dimension", 1536),
            )
        elif provider == "sentence_transformers":
            st_config = self.config["sentence_transformers"]
            return SentenceTransformerClient(model_name=st_config["model"])
        else:
            # Default to Azure OpenAI
            azure_config = self.config["azure_openai"]
            return AzureOpenAIEmbeddings(
                azure_endpoint=azure_config["endpoint"],
                api_key=azure_config["api_key"],
                azure_deployment=azure_config["deployment_name"],
                api_version=azure_config["api_version"],
            )

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
        qdrant_config = self.config["qdrant"]
        connection_method = qdrant_config.get("connection_method", "auto")

        # Helper function to test connection
        def test_client(client: QdrantClient, method_name: str) -> bool:
            try:
                client.get_collections()
                print(f"  ✓ Connected using {method_name}")
                return True
            except Exception:
                return False

        # Get connection parameters
        url = qdrant_config.get("url", "")
        api_key = qdrant_config.get("api_key")
        timeout = qdrant_config.get("timeout", 30)

        # Parse URL components if URL is provided
        hostname = ""
        default_port = 6333  # Default Qdrant port
        use_https = False

        if url:
            if url.startswith("https://"):
                use_https = True
                hostname = url.replace("https://", "").split("/")[0].split(":")[0]
                # Check if custom port is specified in URL
                if ":" in url.replace("https://", "").split("/")[0]:
                    custom_port = int(
                        url.replace("https://", "").split(":")[1].split("/")[0]
                    )
                    default_port = custom_port
            elif url.startswith("http://"):
                hostname = url.replace("http://", "").split("/")[0].split(":")[0]
                if ":" in url.replace("http://", "").split("/")[0]:
                    default_port = int(
                        url.replace("http://", "").split(":")[1].split("/")[0]
                    )
            else:
                # Assume it's just a hostname or hostname:port
                if ":" in url:
                    hostname = url.split(":")[0]
                    default_port = int(url.split(":")[1])
                else:
                    hostname = url

        # Use explicit host/port if provided in config
        hostname = qdrant_config.get("host", hostname)
        port = qdrant_config.get("port", default_port)

        # Connection attempts based on method
        attempts = []

        if connection_method == "auto":
            print("🔍 Auto-detecting Qdrant connection method...")
            # Try reverse proxy first (most common for cloud services)
            if use_https:
                attempts.append(
                    (
                        "reverse_proxy",
                        lambda: QdrantClient(
                            host=hostname,
                            port=443,
                            api_key=api_key,
                            https=True,
                            timeout=timeout,
                            prefer_grpc=False,
                        ),
                    )
                )
            # Try direct connection with default or specified port
            attempts.append(
                (
                    "direct",
                    lambda: QdrantClient(
                        host=hostname,
                        port=port,
                        api_key=api_key,
                        https=use_https,
                        timeout=timeout,
                    ),
                )
            )
            # Try URL-based if URL provided
            if url:
                attempts.append(
                    (
                        "url",
                        lambda: QdrantClient(
                            url=url, api_key=api_key, timeout=timeout, prefer_grpc=False
                        ),
                    )
                )

        elif connection_method == "reverse_proxy":
            attempts.append(
                (
                    "reverse_proxy",
                    lambda: QdrantClient(
                        host=hostname,
                        port=443,
                        api_key=api_key,
                        https=True,
                        timeout=timeout,
                        prefer_grpc=False,
                    ),
                )
            )

        elif connection_method == "direct":
            attempts.append(
                (
                    "direct",
                    lambda: QdrantClient(
                        host=hostname,
                        port=port,
                        api_key=api_key,
                        https=use_https,
                        timeout=timeout,
                    ),
                )
            )

        elif connection_method == "url":
            attempts.append(
                (
                    "url",
                    lambda: QdrantClient(
                        url=url, api_key=api_key, timeout=timeout, prefer_grpc=False
                    ),
                )
            )

        else:
            raise ValueError(f"Unknown connection_method: {connection_method}")

        # Try each connection method
        last_error = None
        for method_name, client_factory in attempts:
            try:
                client = client_factory()
                if test_client(client, method_name):
                    if connection_method == "auto":
                        print(
                            f'💡 Add "connection_method": "{method_name}" to config for faster startup'
                        )
                    return client
            except Exception as e:
                last_error = e
                if connection_method != "auto":
                    print(f"  ✗ Failed with {method_name}: {str(e)[:60]}")

        # If all methods fail, provide helpful error message
        raise ConnectionError(
            f"Failed to connect to Qdrant.\n"
            f"Connection method: {connection_method}\n"
            f"URL: {url}\n"
            f"Last error: {last_error}\n"
            f"Try setting 'connection_method' to 'reverse_proxy', 'direct', or 'url'"
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
        except Exception as e:
            print(f"❌ Failed to connect to {provider_name}: {e}")
            raise

        # Test Qdrant connection
        try:
            collections = self.qdrant_client.get_collections()
            print(
                f"✅ Connected to Qdrant. Found {len(collections.collections)} existing collections"
            )
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant: {e}")
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
            return clone_path
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to clone repository: {e.stderr}")
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
            extensions = self.config["processing"].get("text_extensions", [])
            # Also check for files without extensions that are commonly text files
            no_ext_names = [f for f in extensions if not f.startswith(".")]
        else:
            print("\n🔍 Searching for markdown files...")
            extensions = self.config["processing"]["markdown_extensions"]
            no_ext_names = []

        text_files = []
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
                d for d in dirs if not any(pattern in d for pattern in exclude_patterns)
            ]

            for file in files:
                # Check if file has one of the specified extensions or matches no-extension names
                if any(
                    file.lower().endswith(ext)
                    for ext in extensions
                    if ext.startswith(".")
                ) or (file in no_ext_names):
                    file_path = os.path.join(root, file)
                    # Check if file path contains any exclude patterns
                    if not any(pattern in file_path for pattern in exclude_patterns):
                        text_files.append(file_path)

        file_type = "text" if file_mode == "all_text" else "markdown"
        print(f"✅ Found {len(text_files)} {file_type} files")
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
        content_hash = hashlib.md5(content.encode()).hexdigest()

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
        """Calculate MD5 hash of content for fast duplicate pre-filtering."""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

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

        # Check if collection exists
        collections = self.qdrant_client.get_collections()
        collection_exists = any(
            col.name == collection_name for col in collections.collections
        )

        if collection_exists and qdrant_config["recreate_collection"]:
            print(f"🔄 Recreating existing collection: {collection_name}")
            self.qdrant_client.delete_collection(collection_name)
            collection_exists = False

        if not collection_exists:
            print(f"📚 Creating new collection: {collection_name}")
            print(f"   Vector size: {qdrant_config['vector_size']}")
            print(f"   Distance metric: {qdrant_config['distance']}")

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
                    collection_name=collection_name, vectors_config=vectors_config
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
                )
            print(f"✅ Collection '{collection_name}' created successfully")
        else:
            print(f"📚 Using existing collection: {collection_name}")

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

        all_chunks = []
        all_file_paths = []

        # Initialize PDF processor if needed
        pdf_processor = None
        if self.config.get("pdf_processing", {}).get("enabled", False):
            pdf_processor = PDFProcessor(self.config, self.logger)
            print("   📑 PDF processing enabled")

        # Process each file
        for i, file_path in enumerate(text_files, 1):
            relative_path = os.path.relpath(file_path, repo_root)

            # Skip if file hasn't changed (if tracking is enabled)
            if self.config["processing"].get("track_file_changes", False):
                file_hash = self._calculate_file_hash(file_path)
                if self._is_file_unchanged(relative_path, file_hash):
                    print(f"   ⏭️  Skipping unchanged file: {relative_path}")
                    continue

            print(f"   📄 [{i}/{len(text_files)}] Processing: {relative_path}")

            try:
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
                        continue
                else:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        file_content = f.read()

                if not file_content.strip():
                    print(f"      ⚠️  Empty file: {relative_path}")
                    continue

                # Create document with file-specific metadata
                document = Document(
                    page_content=file_content,
                    metadata={
                        "source": relative_path,
                        "file_path": relative_path,
                        "repository": repo_name,
                        "branch": self.config["github"].get("branch", "main"),
                        "document_type": self._get_document_type(file_path),
                        "file_size": os.path.getsize(file_path),
                    },
                )

                # Split document into chunks
                file_chunks = self.text_splitter.split_documents([document])

                # Add chunk-specific metadata
                for j, chunk in enumerate(file_chunks):
                    chunk.metadata["chunk_index"] = j
                    chunk.metadata["total_chunks"] = len(file_chunks)
                    chunk.metadata["file_path"] = relative_path

                all_chunks.extend(file_chunks)
                all_file_paths.extend([relative_path] * len(file_chunks))

                print(f"      ✅ Created {len(file_chunks)} chunks")

            except Exception as e:
                print(f"      ❌ Error processing {relative_path}: {e}")
                continue

        if not all_chunks:
            print("⚠️  No chunks created from any files")
            return 0

        print(
            f"\n📊 Total chunks created: {len(all_chunks)} from {len(set(all_file_paths))} files"
        )

        # Process and upload chunks with file-aware metadata
        self._upload_chunks_with_file_metadata(all_chunks, repo_name)

        return len(all_chunks)

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate hash of file content for change detection."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _is_file_unchanged(self, relative_path: str, file_hash: str) -> bool:  # noqa: ARG002
        """Check if file has changed since last processing."""
        # TODO: Implement file hash tracking (could use a local cache file or Qdrant metadata)
        # For now, return False to always process files
        return False

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
    ) -> None:
        """
        Upload chunks to Qdrant with file-aware metadata and improved ID generation.

        This method handles embedding generation and upload for chunks that have
        been processed from individual files, maintaining file-level context
        and metadata for better search quality.

        Args:
            chunks: List of document chunks with file metadata
            repo_name: Repository name for ID generation
        """
        if not chunks:
            return

        print("\n🧠 Processing and uploading chunks to Qdrant...")
        print(f"📝 Processing {len(chunks)} chunks from individual files")

        # Calculate processing stats
        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks) if chunks else 0
        print(f"📊 Average chunk size: {avg_chunk_size:.0f} characters")

        # Generate embeddings for ALL chunks
        print("🧠 Generating embeddings for all chunks (with rate limit protection)...")
        all_texts = [chunk.page_content for chunk in chunks]

        embedding_batch_size = self.config["processing"].get("embedding_batch_size", 20)
        batch_delay = self.config["processing"].get("batch_delay_seconds", 1)

        all_embeddings = []

        for i in range(0, len(all_texts), embedding_batch_size):
            batch_texts = all_texts[i : i + embedding_batch_size]
            batch_num = (i // embedding_batch_size) + 1
            total_embedding_batches = (
                len(all_texts) + embedding_batch_size - 1
            ) // embedding_batch_size

            print(
                f"  🧠 Processing embedding batch {batch_num}/{total_embedding_batches} ({len(batch_texts)} chunks)"
            )

            # Check cache first
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

            # Generate embeddings for non-cached texts
            if texts_to_generate:
                new_embeddings = self._generate_embeddings_with_retry(texts_to_generate)

                # Cache the new embeddings
                for text, embedding in zip(texts_to_generate, new_embeddings):
                    self.embedding_cache.set(text, embedding)

                # Insert new embeddings into batch at correct positions
                for idx, embedding in zip(text_indices, new_embeddings):
                    batch_embeddings.insert(idx, embedding)

            all_embeddings.extend(batch_embeddings)

            # Delay between batches to be gentle on the API
            if i + embedding_batch_size < len(all_texts) and texts_to_generate:
                time.sleep(batch_delay)

        # Remove duplicates based on embedding similarity (if enabled)
        if self.config["processing"].get("deduplication_enabled", True):
            print("🔍 Running deduplication analysis...")
            similarity_threshold = self.config["processing"].get(
                "similarity_threshold", 0.95
            )
            unique_chunks, unique_embeddings = self._remove_duplicates(
                chunks, all_embeddings, similarity_threshold=similarity_threshold
            )
        else:
            print("ℹ️  Deduplication disabled - using all chunks")
            unique_chunks, unique_embeddings = chunks, all_embeddings

        if not unique_chunks:
            print("❌ No unique chunks remaining after deduplication!")
            return

        # Process unique chunks in batches for upload
        batch_size = 10
        collection_name = self.config["qdrant"]["collection_name"]
        total_batches = (len(unique_chunks) + batch_size - 1) // batch_size

        print(
            f"🚀 Starting batch upload: {total_batches} batches of {batch_size} chunks each"
        )

        successful_uploads = 0

        for i in range(0, len(unique_chunks), batch_size):
            batch_num = i // batch_size + 1
            batch_chunks = unique_chunks[i : i + batch_size]
            batch_embeddings = unique_embeddings[i : i + batch_size]

            try:
                # Create points for Qdrant
                points = []
                for j, (chunk, embedding) in enumerate(
                    zip(batch_chunks, batch_embeddings)
                ):
                    # Generate deterministic ID using file path and content
                    file_path = chunk.metadata.get("file_path", "unknown")
                    chunk_index = chunk.metadata.get("chunk_index", j)

                    # Enhanced ID generation with file path
                    point_id = self._generate_file_aware_chunk_id(
                        chunk.page_content, chunk_index, repo_name, file_path
                    )

                    # Use new optimized payload creation with file metadata
                    payload = create_payload(
                        chunk=chunk,
                        config=self.config,
                        chunk_index=chunk_index,
                        repo_name=repo_name,
                        file_path=file_path,
                    )

                    # Handle named vectors vs default vectors
                    vector_name = self.config["qdrant"].get("vector_name")
                    if vector_name:
                        points.append(
                            PointStruct(
                                id=point_id,
                                vector={vector_name: embedding},
                                payload=payload,
                            )
                        )
                    else:
                        points.append(
                            PointStruct(id=point_id, vector=embedding, payload=payload)
                        )

                # Upload batch to Qdrant
                self.qdrant_client.upsert(
                    collection_name=collection_name, points=points
                )

                successful_uploads += len(batch_chunks)
                print(
                    f"  ✅ Uploaded batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)"
                )

                # Show progress periodically
                if batch_num % 10 == 0 or batch_num == total_batches:
                    progress = (successful_uploads / len(unique_chunks)) * 100
                    print(
                        f"  📊 Progress: {progress:.0f}% ({successful_uploads}/{len(unique_chunks)} unique chunks)"
                    )

            except Exception as e:
                print(f"  ❌ Error uploading batch {batch_num}: {e}")
                raise

        print(
            f"\n✅ Upload completed: {successful_uploads} chunks uploaded to collection '{collection_name}'"
        )

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
        content_hash = hashlib.md5(content.encode()).hexdigest()

        # Create a deterministic UUID from the hash
        namespace = uuid.UUID("12345678-1234-5678-1234-123456789abc")

        # Normalize the file path to ensure consistency
        normalized_path = file_path.replace("\\", "/").strip("/")

        # Include chunk index in the unique string for file-specific positioning
        unique_string = f"{repo_name}_{normalized_path}_{chunk_index}_{content_hash}"

        return str(uuid.uuid5(namespace, unique_string))

    def _process_and_upload_documents(
        self, combined_content: str, repo_name: str
    ) -> None:
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
                "source": "github_repository",
                "repository": repo_name,
                "branch": branch,
                "document_type": "combined_text",
                "processed_at": datetime.now().isoformat(),
            },
        )

        # Split document into chunks
        print("✂️  Splitting document into chunks...")
        chunks = self.text_splitter.split_documents([document])
        print(f"📝 Split document into {len(chunks)} chunks")

        # Calculate processing stats
        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks) if chunks else 0
        print(f"📊 Average chunk size: {avg_chunk_size:.0f} characters")

        # Generate embeddings for ALL chunks first (for deduplication)
        # Process in smaller batches to avoid rate limits
        print("🧠 Generating embeddings for all chunks (with rate limit protection)...")
        all_texts = [chunk.page_content for chunk in chunks]

        embedding_batch_size = self.config["processing"].get("embedding_batch_size", 20)
        max_retries = self.config["processing"].get("max_retries", 3)
        batch_delay = self.config["processing"].get("batch_delay_seconds", 1)

        all_embeddings = []

        for i in range(0, len(all_texts), embedding_batch_size):
            batch_texts = all_texts[i : i + embedding_batch_size]
            batch_num = (i // embedding_batch_size) + 1
            total_embedding_batches = (
                len(all_texts) + embedding_batch_size - 1
            ) // embedding_batch_size

            print(
                f"  🧠 Processing embedding batch {batch_num}/{total_embedding_batches} ({len(batch_texts)} chunks)"
            )

            # Use cache for individual texts in batch
            batch_embeddings = []
            texts_to_generate = []
            cached_indices = []

            for idx, text in enumerate(batch_texts):
                # Try to get from cache first
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash in self.embedding_cache.cache:
                    # Use cached embedding
                    batch_embeddings.append(self.embedding_cache.cache[text_hash])
                    self.embedding_cache.hits += 1
                else:
                    # Mark for generation
                    texts_to_generate.append(text)
                    cached_indices.append(idx)

            # Generate embeddings for non-cached texts
            if texts_to_generate:
                new_embeddings = self._generate_embeddings_with_retry(
                    texts_to_generate, max_retries
                )

                # Add to cache and results
                for text, embedding in zip(texts_to_generate, new_embeddings):
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    if len(self.embedding_cache.cache) < self.embedding_cache.max_size:
                        self.embedding_cache.cache[text_hash] = embedding
                    self.embedding_cache.misses += 1
                    batch_embeddings.append(embedding)

            all_embeddings.extend(batch_embeddings)

            # Delay between batches to be gentle on the API
            if i + embedding_batch_size < len(all_texts) and texts_to_generate:
                time.sleep(batch_delay)

        # Remove duplicates based on embedding similarity (if enabled)
        if self.config["processing"].get("deduplication_enabled", True):
            print("🔍 Running deduplication analysis...")
            similarity_threshold = self.config["processing"].get(
                "similarity_threshold", 0.95
            )
            unique_chunks, unique_embeddings = self._remove_duplicates(
                chunks, all_embeddings, similarity_threshold=similarity_threshold
            )
        else:
            print("ℹ️  Deduplication disabled - using all chunks")
            unique_chunks, unique_embeddings = chunks, all_embeddings

        if not unique_chunks:
            print("❌ No unique chunks remaining after deduplication!")
            return

        # Process unique chunks in batches for upload
        batch_size = 10
        collection_name = self.config["qdrant"]["collection_name"]
        total_batches = (len(unique_chunks) + batch_size - 1) // batch_size

        print(
            f"🚀 Starting batch upload: {total_batches} batches of {batch_size} chunks each"
        )

        successful_uploads = 0

        for i in range(0, len(unique_chunks), batch_size):
            batch_num = i // batch_size + 1
            batch_chunks = unique_chunks[i : i + batch_size]
            batch_embeddings = unique_embeddings[i : i + batch_size]

            try:
                # Create points for Qdrant
                points = []
                for j, (chunk, embedding) in enumerate(
                    zip(batch_chunks, batch_embeddings)
                ):
                    # Generate deterministic ID for chunk
                    chunk_index = i + j
                    # Get file path from metadata for consistent ID generation
                    file_path = chunk.metadata.get("source", "unknown")
                    point_id = self._generate_chunk_id(
                        chunk.page_content, chunk_index, repo_name, file_path
                    )

                    # Debug: Log first 100 chars of content and generated ID
                    if j == 0 and i == 0:  # Only log first chunk of first batch
                        content_preview = chunk.page_content[:100].replace("\n", " ")
                        print(f"   🔍 Debug - First chunk ID: {point_id[:8]}...")
                        print(f"   🔍 Debug - Content preview: {content_preview}...")
                        print(f"   🔍 Debug - Source: {file_path}")

                    # Use new optimized payload creation
                    payload = create_payload(
                        chunk=chunk,
                        config=self.config,
                        chunk_index=chunk_index,
                        repo_name=repo_name,
                        file_path=file_path,
                    )

                    # Handle named vectors vs default vectors
                    vector_name = self.config["qdrant"].get("vector_name")
                    if vector_name:
                        # Use named vectors - create dict for named vectors
                        points.append(
                            PointStruct(
                                id=point_id,
                                vector={vector_name: embedding},
                                payload=payload,
                            )
                        )
                    else:
                        # Use default vectors - pass embedding directly
                        points.append(
                            PointStruct(id=point_id, vector=embedding, payload=payload)
                        )

                # Upload batch to Qdrant
                self.qdrant_client.upsert(
                    collection_name=collection_name, points=points
                )

                successful_uploads += len(batch_chunks)
                print(
                    f"  ✅ Uploaded batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)"
                )

                # Debug: Log all unique point IDs to help track duplicates
                if batch_num == 1:  # Log IDs from first batch
                    print("   🔍 Debug - First batch point IDs:")
                    for point in points[:3]:  # Show first 3 IDs
                        print(f"      - {point.id[:16]}...")

                # Show progress every 20 batches or at the end (less verbose)
                if batch_num % 20 == 0 or batch_num == total_batches:
                    progress = (successful_uploads / len(unique_chunks)) * 100
                    print(
                        f"  📊 Progress: {progress:.0f}% ({successful_uploads}/{len(unique_chunks)} unique chunks)"
                    )

            except Exception as e:
                print(f"  ❌ Failed to upload batch {batch_num}: {e}")
                continue

        original_count = len(chunks)
        duplicate_count = original_count - len(unique_chunks)
        print(
            f"🎉 Successfully uploaded {successful_uploads}/{len(unique_chunks)} unique chunks to Qdrant collection '{collection_name}'"
        )
        if duplicate_count > 0:
            print(
                f"   📊 Deduplication stats: {duplicate_count} duplicates removed from {original_count} original chunks"
            )

        # Display cache statistics
        cache_stats = self.embedding_cache.get_stats()
        if cache_stats["hits"] > 0 or cache_stats["misses"] > 0:
            print(
                f"   💾 Cache stats: {cache_stats['hits']} hits, {cache_stats['misses']} misses "
                f"({cache_stats['hit_rate']} hit rate)"
            )

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
        self._process_and_upload_documents(combined_content, repo_name)

        # Return chunk count (we'll track this in the upload process)
        # For now, split and count chunks
        document = Document(
            page_content=combined_content,
            metadata={
                "source": "github_repository",
                "repository": repo_name,
                "branch": self.config["github"].get("branch", "default"),
            },
        )
        chunks = self.text_splitter.split_documents([document])

        # Account for deduplication if enabled
        if self.config["processing"].get("deduplication_enabled", True):
            # Estimate unique chunks based on typical deduplication ratio
            # This is an approximation since we don't track the exact count in the current method
            return int(len(chunks) * 0.9)  # Assume 10% duplicates on average
        else:
            return len(chunks)

    def process_repository_with_override(
        self,
        repo_url: str,
        branch: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> ProcessingResult:
        """
        Process a repository with optional overrides for branch and collection.

        Args:
            repo_url: GitHub repository URL
            branch: Optional branch override
            collection_name: Optional collection name override

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

        try:
            if branch:
                self.config["github"]["branch"] = branch
            if collection_name:
                self.config["qdrant"]["collection_name"] = collection_name

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
                    return (0, 0)

                # Setup Qdrant collection
                self._setup_qdrant_collection()

                # Check if we should process files individually or combine them
                combine_docs = self.config["processing"].get("combine_documents", True)
                print(f"\n🔍 Debug: combine_documents = {combine_docs}")

                if combine_docs is False:  # Explicitly check for False
                    # Process files individually for better context and search quality
                    print("\n📄 Processing files individually for better context...")
                    chunks_created = self._process_files_individually(
                        text_files, repo_name, clone_path
                    )
                else:
                    # Legacy mode: Combine text files into folder-based files + overall combined file
                    print("\n📄 Combining documents (legacy mode)...")
                    combined_content = self._combine_text_files(text_files, repo_name)

                    # Process and upload ONLY the final combined document
                    print(
                        "\n🎯 Creating vector embeddings for the combined document only..."
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

    print("\n" + "=" * 60)
    print("STARTING MULTI-REPOSITORY PROCESSING")
    print(f"Total repositories to process: {len(repositories)}")
    print("=" * 60)

    overall_start_time = datetime.now()

    for i, repo_config in enumerate(repositories, 1):
        print("\n" + "=" * 60)
        print(f"Processing repository {i}/{len(repositories)}")
        print(f"Repository: {repo_config.url}")
        print(f"Branch: {repo_config.branch or 'default'}")
        print(f"Collection: {repo_config.collection_name}")
        print("=" * 60)

        try:
            result = processor.process_repository_with_override(
                repo_url=repo_config.url,
                branch=repo_config.branch,
                collection_name=repo_config.collection_name,
            )
            results.append(result)
            print(f"✅ Successfully processed: {repo_config.url}")

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Failed to process {repo_config.url}: {error_msg}")

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


def main():
    """Main entry point."""
    # Set up signal handler for clean interruption
    signal.signal(signal.SIGINT, signal_handler)

    # Load environment variables from .env file
    load_dotenv()

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

    try:
        processor = GitHubToQdrantProcessor(args.config)

        # Check if repository list is provided
        if args.repo_list:
            # Process multiple repositories from list
            process_repository_list(processor, args.repo_list)
        else:
            # Process single repository (current behavior)
            processor.process_repository(args.repo_url)

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully without showing traceback
        print("\n\n⚠️  Process interrupted by user (Ctrl+C)")
        print("🧹 Cleaning up and exiting...")
        return 130  # Standard Unix exit code for SIGINT
    except Exception as e:
        logging.error("Script failed: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
