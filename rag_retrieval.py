#!/usr/bin/env python3
"""
Simple retrieval CLI for Qdrant collections produced by github_to_qdrant.py.

Features (configurable):
- Query top-K from Qdrant
- Group/cap results per file (source)
- Optional parent/window expansion (implemented in a follow-up step)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml
from qdrant_client import QdrantClient
from qdrant_client.http import models as m

from github_to_qdrant import (
    ConfigLoader,
    MistralEmbeddingClient,
    SentenceTransformerClient,
    create_qdrant_client_from_config,
    get_metadata_structure,
    marker_exclusion_filter,
    payload_field_path,
)
from langchain_openai import AzureOpenAIEmbeddings


SENSITIVE_METADATA_PARTS = ("api_key", "token", "secret", "password", "authorization")


@dataclass
class CollectionTarget:
    """One Qdrant collection selected for retrieval."""

    collection_name: str
    repository_url: str = ""
    repository_name: str = ""
    branch: str = ""


@dataclass
class CollectionCompatibility:
    """Compatibility result for one selected Qdrant collection."""

    target: CollectionTarget
    status: str
    reason: str
    exists: bool = False
    usable: bool = False
    warning: bool = False
    expected_provider: str = ""
    expected_model: str = ""
    actual_provider: str = ""
    actual_model: str = ""
    expected_vector_size: Optional[int] = None
    actual_vector_size: Optional[int] = None
    expected_distance: str = ""
    actual_distance: str = ""
    expected_vector_name: str = ""
    actual_vector_name: str = ""

    @property
    def collection_name(self) -> str:
        """Return the Qdrant collection name."""
        return self.target.collection_name

    def warning_message(self) -> str:
        """Return a human-readable compatibility warning."""
        if self.usable:
            return f"Collection '{self.collection_name}': {self.reason}"
        return f"Collection '{self.collection_name}' skipped: {self.reason}"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "collection": self.collection_name,
            "repository_url": self.target.repository_url,
            "repository_name": self.target.repository_name,
            "branch": self.target.branch,
            "status": self.status,
            "reason": self.reason,
            "exists": self.exists,
            "usable": self.usable,
            "warning": self.warning,
            "expected_provider": self.expected_provider,
            "expected_model": self.expected_model,
            "actual_provider": self.actual_provider,
            "actual_model": self.actual_model,
            "expected_vector_size": self.expected_vector_size,
            "actual_vector_size": self.actual_vector_size,
            "expected_distance": self.expected_distance,
            "actual_distance": self.actual_distance,
            "expected_vector_name": self.expected_vector_name,
            "actual_vector_name": self.actual_vector_name,
        }


@dataclass
class QueryHit:
    """Display-ready retrieval hit."""

    score: float
    file_path: str
    content: str
    metadata: Dict[str, Any]
    preview: str = ""
    expanded_context: str = ""
    collection: str = ""
    repository_url: str = ""
    repository_name: str = ""
    repository_branch: str = ""


@dataclass
class QueryTimings:
    """Timing measurements for a query execution."""

    embed_seconds: float
    search_seconds: float
    group_seconds: float


@dataclass
class QueryResponse:
    """Structured retrieval response for CLI renderers."""

    query: str
    collection: str
    hits: List[QueryHit]
    timings: QueryTimings
    candidates: int
    collections: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    skipped_collections: List[CollectionCompatibility] = field(default_factory=list)
    collection_statuses: List[CollectionCompatibility] = field(default_factory=list)


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _resolve_env_vars(cfg)


def load_repository_targets(repo_list_path: str) -> List[CollectionTarget]:
    """Load collection targets from the existing repositories.yaml schema."""
    try:
        with open(repo_list_path, "r", encoding="utf-8") as repo_list_file:
            data = yaml.safe_load(repo_list_file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Repository list file not found: {repo_list_path}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML format in repository list: {exc}") from exc

    if not data or "repositories" not in data:
        raise ValueError(
            "Repository list file must contain a 'repositories' key with a list of repositories"
        )

    repositories = data["repositories"]
    if not isinstance(repositories, list):
        raise ValueError("'repositories' must be a list")

    targets: List[CollectionTarget] = []
    seen: set[str] = set()
    for index, repo in enumerate(repositories, 1):
        if not isinstance(repo, dict):
            raise ValueError(
                f"Repository {index}: Each repository must be a dictionary"
            )
        if "url" not in repo:
            raise ValueError(f"Repository {index}: 'url' field is required")
        if "collection_name" not in repo:
            raise ValueError(f"Repository {index}: 'collection_name' field is required")

        collection_name = str(repo["collection_name"]).strip()
        if not collection_name or collection_name in seen:
            continue
        seen.add(collection_name)
        targets.append(
            CollectionTarget(
                collection_name=collection_name,
                repository_url=str(repo.get("url") or ""),
                repository_name=str(repo.get("name") or ""),
                branch=str(repo.get("branch") or ""),
            )
        )

    if not targets:
        raise ValueError("Repository list does not contain any collection_name values")
    return targets


def resolve_collection_targets(
    cfg: Dict[str, Any],
    collection: Optional[str] = None,
    repo_list: Optional[str] = None,
) -> List[CollectionTarget]:
    """Resolve config, collection override, and repo-list into retrieval targets."""
    if repo_list:
        repo_targets = load_repository_targets(repo_list)
        if collection:
            selected = str(collection).strip()
            if not selected:
                raise ValueError("--collection cannot be blank")
            for target in repo_targets:
                if target.collection_name == selected:
                    return [target]
            return [CollectionTarget(collection_name=selected)]
        return repo_targets

    if collection:
        selected = str(collection).strip()
        if not selected:
            raise ValueError("--collection cannot be blank")
        return [CollectionTarget(collection_name=selected)]

    qcfg = cfg.get("qdrant", {})
    if "collection_name" not in qcfg:
        raise ValueError("Missing required 'qdrant.collection_name' in config")
    default_collection = str(qcfg["collection_name"]).strip()
    if not default_collection:
        raise ValueError("qdrant.collection_name cannot be blank")
    return [CollectionTarget(collection_name=default_collection)]


def _resolve_env_vars(obj: Any) -> Any:
    """Resolve ${VAR} and ${VAR:-default} in loaded YAML config."""
    import re

    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(x) for x in obj]
    if isinstance(obj, str):
        pattern = r"\$\{([^}]+)\}"

        def repl(match):
            expr = match.group(1)
            if ":-" in expr:
                var, default = expr.split(":-", 1)
                return os.getenv(var, default)
            val = os.getenv(expr)
            return val if val is not None else match.group(0)

        return re.sub(pattern, repl, obj)
    return obj


def _field_path(name: str, metadata_structure: str) -> str:
    return payload_field_path(name, metadata_structure)


def _build_filter(
    metadata_structure: str, raw_filters: Optional[Dict[str, Any]]
) -> Optional[m.Filter]:
    must: List[m.FieldCondition] = []
    marker_filter = marker_exclusion_filter(metadata_structure)
    must_not = list(marker_filter.must_not or [])
    if not raw_filters:
        return m.Filter(must_not=must_not) if must_not else None
    for key, value in raw_filters.items():
        must.append(
            m.FieldCondition(
                key=_field_path(key, metadata_structure),
                match=m.MatchValue(value=value),
            )
        )
    return m.Filter(must=must, must_not=must_not) if must or must_not else None


def _config_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _build_search_params(qdrant_config: Dict[str, Any]) -> Optional[m.SearchParams]:
    """Build optional Qdrant search params, including quantized search tuning."""
    quantization = qdrant_config.get("quantization") or {}
    if not isinstance(quantization, dict):
        return None
    search = quantization.get("search") or {}
    if not isinstance(search, dict) or not search:
        return None

    quantization_kwargs: Dict[str, Any] = {}
    if "ignore" in search:
        quantization_kwargs["ignore"] = _config_bool(search["ignore"])
    if "rescore" in search:
        quantization_kwargs["rescore"] = _config_bool(search["rescore"])
    if "oversampling" in search and search["oversampling"] is not None:
        quantization_kwargs["oversampling"] = float(search["oversampling"])

    if not quantization_kwargs:
        return None
    return m.SearchParams(
        quantization=m.QuantizationSearchParams(**quantization_kwargs)
    )


def _group_by_file(
    hits: List[m.ScoredPoint],
    metadata_structure: str,
    max_per_file: int,
) -> List[m.ScoredPoint]:
    grouped: Dict[str, List[m.ScoredPoint]] = defaultdict(list)
    for hit in hits:
        payload = hit.payload or {}
        meta = (
            payload.get("metadata", {}) if metadata_structure == "nested" else payload
        )
        file_path = meta.get("file_path") or meta.get("source") or "unknown"
        grouped[str(file_path)].append(hit)

    # Sort each file group by score desc, then interleave up to max_per_file per file
    for k in grouped:
        grouped[k].sort(key=lambda h: float(h.score or 0.0), reverse=True)

    files = list(grouped.keys())
    out: List[m.ScoredPoint] = []
    idx = 0
    while True:
        progressed = False
        for f in files:
            if idx < len(grouped[f]) and idx < max_per_file:
                out.append(grouped[f][idx])
                progressed = True
        if not progressed:
            break
        idx += 1
    return out


def _init_embedder(cfg: Dict[str, Any]):
    provider = cfg.get("embedding_provider", "azure_openai")
    if provider == "mistral_ai":
        mcfg = cfg["mistral_ai"]
        return MistralEmbeddingClient(
            api_key=mcfg["api_key"],
            model=mcfg.get("model", "codestral-embed"),
            dimensions=mcfg.get("dimensions", mcfg.get("output_dimension", 1536)),
        )
    if provider == "sentence_transformers":
        scfg = cfg["sentence_transformers"]
        return SentenceTransformerClient(
            model_name=scfg["model"],
            dimensions=scfg.get("dimensions"),
        )

    # Default to Azure OpenAI embeddings (LangChain wrapper)
    acfg = cfg["azure_openai"]
    params = {
        "azure_endpoint": acfg["endpoint"],
        "api_key": acfg["api_key"],
        "api_version": acfg["api_version"],
        # support both config styles in this repo
        "azure_deployment": acfg.get("model") or acfg.get("deployment_name"),
    }
    if "dimensions" in acfg:
        params["dimensions"] = acfg["dimensions"]
    return AzureOpenAIEmbeddings(**params)


def _embed_query(embedder, query: str) -> List[float]:
    # Both our custom clients and LangChain embedder implement embed_query
    return embedder.embed_query(query)


def _extract_meta(payload: Dict[str, Any], metadata_structure: str) -> Dict[str, Any]:
    return payload.get("metadata", {}) if metadata_structure == "nested" else payload


def _value_name(value: Any) -> str:
    """Return a stable string for Qdrant enum/string values."""
    if value is None:
        return ""
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _expected_embedding_identity(cfg: Dict[str, Any]) -> Tuple[str, str]:
    """Return the embedding provider/model that the active config will use."""
    provider = str(cfg.get("embedding_provider") or "azure_openai").strip()
    if provider == "mistral_ai":
        model = cfg.get("mistral_ai", {}).get("model", "codestral-embed")
    elif provider == "sentence_transformers":
        model = cfg.get("sentence_transformers", {}).get("model", "")
    else:
        azure = cfg.get("azure_openai", {})
        model = azure.get("model") or azure.get("deployment_name") or ""
    return provider, str(model or "")


def _vector_params_and_name(
    info: Any, expected_vector_name: str
) -> Tuple[Any, str, str]:
    """Return Qdrant vector params, actual vector name, and an error reason."""
    config = getattr(info, "config", None)
    params = getattr(config, "params", None)
    vectors = getattr(params, "vectors", None)
    if isinstance(vectors, dict):
        vector_names = sorted(str(name) for name in vectors)
        if expected_vector_name:
            if expected_vector_name not in vectors:
                return (
                    None,
                    "",
                    (
                        f"config expects named vector '{expected_vector_name}', "
                        f"collection has {', '.join(vector_names) or 'none'}."
                    ),
                )
            return vectors[expected_vector_name], expected_vector_name, ""
        return (
            None,
            vector_names[0] if len(vector_names) == 1 else "",
            (
                "config uses an unnamed vector but collection uses named vector(s): "
                f"{', '.join(vector_names) or 'none'}."
            ),
        )

    if expected_vector_name:
        return (
            None,
            "",
            f"config expects named vector '{expected_vector_name}', collection is unnamed.",
        )
    return vectors, "", ""


def _sample_payload_for_compatibility(
    client: Any, collection: str, metadata_structure: str
) -> Dict[str, Any]:
    """Return a non-marker payload sample for embedding metadata checks."""
    records, _next_page = client.scroll(
        collection_name=collection,
        limit=16,
        with_payload=True,
        with_vectors=False,
    )
    fallback_payload: Dict[str, Any] = {}
    for record in records or []:
        payload = record.payload or {}
        if not fallback_payload:
            fallback_payload = payload
        meta = _extract_meta(payload, metadata_structure)
        if meta.get("record_type") == "file_marker":
            continue
        if any(payload.get(field) for field in ("page_content", "content", "text")):
            return payload
        if meta.get("embedding_provider") or meta.get("embedding_model"):
            return payload
    return fallback_payload


def _payload_embedding_identity(
    payload: Dict[str, Any], metadata_structure: str
) -> Tuple[str, str]:
    """Return embedding provider/model metadata from a Qdrant payload sample."""
    meta = _extract_meta(payload, metadata_structure)
    provider = str(meta.get("embedding_provider") or "").strip()
    model = str(meta.get("embedding_model") or "").strip()
    return provider, model


def redact_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Return metadata with obvious secret fields redacted for JSON output."""
    safe: Dict[str, Any] = {}
    for key, value in metadata.items():
        normalized_key = str(key).lower()
        if any(part in normalized_key for part in SENSITIVE_METADATA_PARTS):
            safe[key] = "<redacted>"
            continue
        if isinstance(value, str):
            safe[key] = re.sub(
                r"https://[^\s/@]+@github\.com",
                "https://<redacted>@github.com",
                value,
            )
        else:
            safe[key] = value
    return safe


def _compatibility_from_error(
    target: CollectionTarget,
    status: str,
    reason: str,
    cfg: Dict[str, Any],
    exists: bool = True,
    usable: bool = False,
    warning: bool = False,
) -> CollectionCompatibility:
    expected_provider, expected_model = _expected_embedding_identity(cfg)
    qcfg = cfg.get("qdrant", {})
    return CollectionCompatibility(
        target=target,
        status=status,
        reason=reason,
        exists=exists,
        usable=usable,
        warning=warning,
        expected_provider=expected_provider,
        expected_model=expected_model,
        expected_vector_size=_int_or_none(qcfg.get("vector_size")),
        expected_distance=str(qcfg.get("distance", "Cosine") or ""),
        expected_vector_name=str(qcfg.get("vector_name") or ""),
    )


def inspect_collection_compatibility(
    cfg: Dict[str, Any],
    targets: List[CollectionTarget],
    client: Any,
) -> List[CollectionCompatibility]:
    """Inspect selected Qdrant collections before retrieval."""
    qcfg = cfg.get("qdrant", {})
    expected_provider, expected_model = _expected_embedding_identity(cfg)
    expected_size = _int_or_none(qcfg.get("vector_size"))
    expected_distance = str(qcfg.get("distance", "Cosine") or "").lower()
    expected_vector_name = str(qcfg.get("vector_name") or "")
    metadata_structure = get_metadata_structure(cfg)
    statuses: List[CollectionCompatibility] = []

    for target in targets:
        name = target.collection_name
        try:
            exists = bool(client.collection_exists(collection_name=name))
        except Exception as exc:  # pragma: no cover - defensive network wrapping
            statuses.append(
                _compatibility_from_error(
                    target,
                    "missing",
                    f"Qdrant collection check failed: {exc}",
                    cfg,
                    exists=False,
                )
            )
            continue

        if not exists:
            statuses.append(
                _compatibility_from_error(
                    target,
                    "missing",
                    "collection does not exist. Run ingestion first.",
                    cfg,
                    exists=False,
                )
            )
            continue

        try:
            info = client.get_collection(collection_name=name)
        except Exception as exc:  # pragma: no cover - best-effort compatibility check
            statuses.append(
                _compatibility_from_error(
                    target,
                    "metadata_unknown",
                    f"compatibility could not be fully inspected: {exc}",
                    cfg,
                    exists=True,
                    usable=True,
                    warning=True,
                )
            )
            continue

        params, actual_vector_name, vector_error = _vector_params_and_name(
            info, expected_vector_name
        )
        if vector_error:
            status = _compatibility_from_error(
                target,
                "vector_name_mismatch",
                vector_error,
                cfg,
                exists=True,
            )
            status.actual_vector_name = actual_vector_name
            statuses.append(status)
            continue
        if params is None:
            statuses.append(
                _compatibility_from_error(
                    target,
                    "vector_name_mismatch",
                    "vector configuration was not found.",
                    cfg,
                    exists=True,
                )
            )
            continue

        actual_size = _int_or_none(getattr(params, "size", None))
        actual_distance = _value_name(getattr(params, "distance", "")).lower()
        base = CollectionCompatibility(
            target=target,
            status="usable",
            reason="collection is compatible with the active embedding config.",
            exists=True,
            usable=True,
            expected_provider=expected_provider,
            expected_model=expected_model,
            expected_vector_size=expected_size,
            actual_vector_size=actual_size,
            expected_distance=str(qcfg.get("distance", "Cosine") or ""),
            actual_distance=_value_name(getattr(params, "distance", "")),
            expected_vector_name=expected_vector_name,
            actual_vector_name=actual_vector_name,
        )

        if expected_size is not None and actual_size != expected_size:
            base.status = "vector_mismatch"
            base.reason = (
                f"vector size is {actual_size}, config expects {expected_size}."
            )
            base.usable = False
            statuses.append(base)
            continue
        if (
            expected_distance
            and actual_distance
            and expected_distance not in actual_distance
        ):
            base.status = "distance_mismatch"
            base.reason = (
                f"distance is {base.actual_distance or 'unknown'}, "
                f"config expects {qcfg.get('distance')}."
            )
            base.usable = False
            statuses.append(base)
            continue

        try:
            payload = _sample_payload_for_compatibility(
                client, name, metadata_structure
            )
        except Exception as exc:  # pragma: no cover - best-effort payload check
            base.status = "metadata_unknown"
            base.reason = f"embedding metadata could not be sampled: {exc}"
            base.warning = True
            statuses.append(base)
            continue

        actual_provider, actual_model = _payload_embedding_identity(
            payload, metadata_structure
        )
        base.actual_provider = actual_provider
        base.actual_model = actual_model
        if not actual_provider and not actual_model:
            base.status = "metadata_unknown"
            base.reason = (
                "embedding provider/model metadata is missing; vector settings match, "
                "so retrieval will continue with a warning."
            )
            base.warning = True
            statuses.append(base)
            continue
        if actual_provider and actual_provider != expected_provider:
            base.status = "embedding_mismatch"
            base.reason = (
                f"embedded with {actual_provider}/{actual_model or 'unknown'}, "
                f"config uses {expected_provider}/{expected_model or 'unknown'}."
            )
            base.usable = False
            statuses.append(base)
            continue
        if actual_model and expected_model and actual_model != expected_model:
            base.status = "embedding_mismatch"
            base.reason = (
                f"embedded with {actual_provider or expected_provider}/{actual_model}, "
                f"config uses {expected_provider}/{expected_model}."
            )
            base.usable = False
            statuses.append(base)
            continue
        if not actual_model:
            base.status = "metadata_unknown"
            base.reason = (
                f"embedding model metadata is missing; provider {actual_provider} "
                "matches the active config."
            )
            base.warning = True
        statuses.append(base)

    return statuses


def inspect_collection_targets(
    config_path: str,
    targets: List[CollectionTarget],
    quiet: bool = True,
) -> List[CollectionCompatibility]:
    """Load config and inspect selected collection compatibility."""
    ConfigLoader.load_env_for_config(config_path, quiet=quiet)
    cfg = _load_config(config_path)
    if "qdrant" not in cfg:
        raise ValueError("Missing required 'qdrant' section in config")
    client = _init_qdrant_client(cfg["qdrant"])
    return inspect_collection_compatibility(cfg, targets, client)


def _expand_parent_window(
    client: QdrantClient,
    collection: str,
    metadata_structure: str,
    hit: m.ScoredPoint,
    window: int,
) -> str:
    """
    Expand context window around a hit by retrieving neighboring chunks.
    Returns empty string if parent metadata is missing.
    """
    logger = logging.getLogger(__name__)
    payload = hit.payload or {}
    meta = _extract_meta(payload, metadata_structure)
    parent_id = meta.get("parent_id")
    idx = meta.get("chunk_index_within_file")
    if parent_id is None:
        logger.warning("Cannot expand parent window: missing 'parent_id' metadata")
        return ""
    if idx is None:
        logger.warning(
            "Cannot expand parent window: missing 'chunk_index_within_file' metadata"
        )
        return ""

    parent_field = _field_path("parent_id", metadata_structure)
    idx_field = _field_path("chunk_index_within_file", metadata_structure)

    flt = m.Filter(
        must=[
            m.FieldCondition(key=parent_field, match=m.MatchValue(value=parent_id)),
            m.FieldCondition(
                key=idx_field,
                range=m.Range(gte=int(idx) - window, lte=int(idx) + window),
            ),
        ]
    )

    # Retrieve window with buffer (+10) to ensure we get all chunks in range
    # even if some indices are missing
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=flt,
        limit=(2 * window) + 10,
        with_payload=True,
        with_vectors=False,
    )

    # Sort by chunk index so text is ordered
    def _get_idx(p: m.Record) -> int:
        pl = p.payload or {}
        mt = _extract_meta(pl, metadata_structure)
        return int(mt.get("chunk_index_within_file", 0))

    points_sorted = sorted(points, key=_get_idx)
    texts: List[str] = []
    for p in points_sorted:
        pl = p.payload or {}
        # Prefer page_content if present in configured content fields
        texts.append(str(pl.get("page_content") or pl.get("content") or ""))
    return "\n\n".join([t for t in texts if t])


def _execute_search(
    client: QdrantClient,
    collection: str,
    query_text: str,
    query_vec: List[float],
    qcfg: Dict[str, Any],
    retrieval: Dict[str, Any],
    qdrant_filter: Optional[m.Filter],
    fetch_k: int,
    search_params: Optional[m.SearchParams] = None,
) -> List[m.ScoredPoint]:
    """Execute dense-only or hybrid dense+sparse search."""
    vector_name = qcfg.get("vector_name")
    retrieval_mode = str(retrieval.get("mode", "dense")).lower()

    if retrieval_mode == "hybrid":
        sparse_cfg = qcfg.get("sparse_vector", {})
        if not vector_name:
            raise ValueError(
                "Hybrid retrieval requires qdrant.vector_name for the dense vector "
                "(for example: vector_name: dense)."
            )
        if not sparse_cfg.get("enabled", False):
            raise ValueError(
                "Hybrid retrieval requires qdrant.sparse_vector.enabled: true."
            )

        fusion_name = str(retrieval.get("fusion", "rrf")).lower()
        fusion = m.Fusion.RRF if fusion_name == "rrf" else m.Fusion.DBSF
        sparse_name = sparse_cfg.get("name", "sparse")
        sparse_model = sparse_cfg.get("model", "qdrant/bm25")

        query_kwargs: Dict[str, Any] = {
            "collection_name": collection,
            "prefetch": [
                m.Prefetch(
                    query=query_vec,
                    using=vector_name,
                    filter=qdrant_filter,
                    limit=fetch_k,
                ),
                m.Prefetch(
                    query=m.Document(text=query_text, model=sparse_model),
                    using=sparse_name,
                    filter=qdrant_filter,
                    limit=fetch_k,
                ),
            ],
            "query": m.FusionQuery(fusion=fusion),
            "query_filter": qdrant_filter,
            "limit": fetch_k,
            "with_payload": True,
            "with_vectors": False,
        }
        if search_params is not None:
            query_kwargs["search_params"] = search_params
        return client.query_points(**query_kwargs).points

    query_kwargs = {
        "collection_name": collection,
        "query": query_vec,
        "query_filter": qdrant_filter,
        "limit": fetch_k,
        "with_payload": True,
        "with_vectors": False,
    }
    if vector_name:
        query_kwargs["using"] = vector_name
    if search_params is not None:
        query_kwargs["search_params"] = search_params
    return client.query_points(**query_kwargs).points


def _init_qdrant_client(qcfg: Dict[str, Any]) -> QdrantClient:
    """Initialize Qdrant client using the same config conventions as ingestion."""
    return create_qdrant_client_from_config(qcfg, client_class=QdrantClient)


@dataclass
class _CandidateHit:
    """Raw search candidate annotated with its source collection."""

    target: CollectionTarget
    point: m.ScoredPoint


def _candidate_file_key(
    candidate: _CandidateHit, metadata_structure: str
) -> tuple[str, str]:
    payload = candidate.point.payload or {}
    meta = _extract_meta(payload, metadata_structure)
    file_path = meta.get("file_path") or meta.get("source") or "unknown"
    return candidate.target.collection_name, str(file_path)


def _select_global_candidates(
    candidates: List[_CandidateHit],
    metadata_structure: str,
    max_per_file: int,
    top_k: int,
) -> List[_CandidateHit]:
    """Sort candidates globally by score while respecting max chunks per file."""
    counts: Dict[tuple[str, str], int] = defaultdict(int)
    selected: List[_CandidateHit] = []
    for candidate in sorted(
        candidates,
        key=lambda item: float(item.point.score or 0.0),
        reverse=True,
    ):
        key = _candidate_file_key(candidate, metadata_structure)
        if counts[key] >= max_per_file:
            continue
        counts[key] += 1
        selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected


def _query_hit_from_candidate(
    candidate: _CandidateHit,
    client: QdrantClient,
    metadata_structure: str,
    with_parent_window: bool,
    parent_window: int,
) -> QueryHit:
    payload = candidate.point.payload or {}
    meta = _extract_meta(payload, metadata_structure)
    expanded_context = ""
    if with_parent_window:
        expanded_context = _expand_parent_window(
            client=client,
            collection=candidate.target.collection_name,
            metadata_structure=metadata_structure,
            hit=candidate.point,
            window=parent_window,
        )

    return QueryHit(
        score=float(candidate.point.score or 0.0),
        file_path=meta.get("file_path") or meta.get("source") or "unknown",
        content=payload.get("page_content") or payload.get("content") or "",
        metadata=meta,
        preview=meta.get("preview", ""),
        expanded_context=expanded_context,
        collection=candidate.target.collection_name,
        repository_url=candidate.target.repository_url,
        repository_name=candidate.target.repository_name,
        repository_branch=candidate.target.branch,
    )


def check_collection_targets(
    config_path: str,
    targets: List[CollectionTarget],
    quiet: bool = True,
) -> Dict[str, bool]:
    """Return existence status for collection targets in Qdrant."""
    return {
        status.collection_name: status.exists
        for status in inspect_collection_targets(config_path, targets, quiet=quiet)
    }


def execute_query(
    config_path: str,
    query: str,
    limit: Optional[int] = None,
    with_parent_window: bool = False,
    verbose: bool = False,
    quiet: bool = False,
    progress: Optional[Callable[[str], None]] = None,
    collection: Optional[str] = None,
    repo_list: Optional[str] = None,
) -> QueryResponse:
    """Execute a retrieval query and return structured results."""
    log_level = (
        logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    )
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
        force=True,
    )
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    ConfigLoader.load_env_for_config(config_path, quiet=quiet)
    cfg = _load_config(config_path)

    # Validate required config sections
    if "qdrant" not in cfg:
        raise ValueError("Missing required 'qdrant' section in config")
    if "embedding_provider" not in cfg:
        raise ValueError("Missing required 'embedding_provider' section in config")

    qcfg = cfg["qdrant"]
    targets = resolve_collection_targets(
        cfg, collection=collection, repo_list=repo_list
    )
    retrieval = cfg.get("retrieval", {})
    metadata_structure = cfg.get("payload", {}).get("metadata_structure", "nested")

    top_k = int(limit or retrieval.get("top_k", 10))
    # fetch_k should be meaningfully larger than top_k for grouping to work properly
    fetch_k = int(retrieval.get("fetch_k", max(top_k * 4, 40)))
    max_per_file = int(retrieval.get("max_chunks_per_file", 3))
    raw_filters = retrieval.get("filters")
    parent_window = int(retrieval.get("parent_window", 2))

    client = _init_qdrant_client(qcfg)
    collection_statuses = inspect_collection_compatibility(cfg, targets, client)
    skipped_collections = [
        status for status in collection_statuses if not status.usable
    ]
    searchable_targets = [
        status.target for status in collection_statuses if status.usable
    ]
    warnings = [
        status.warning_message()
        for status in collection_statuses
        if status.warning or not status.usable
    ]

    if not searchable_targets:
        details = "; ".join(
            f"{status.collection_name}: {status.reason}"
            for status in collection_statuses
        )
        raise LookupError(
            "None of the requested collections are usable for the active embedding "
            f"config. {details}"
        )

    collection_names = [target.collection_name for target in searchable_targets]
    collection_label = (
        collection_names[0]
        if len(collection_names) == 1
        else f"{len(collection_names)} collections"
    )

    logger.info("Querying collection(s): %s", ", ".join(collection_names))
    logger.debug("Query: %s", query)

    if progress:
        progress("Encoding query")
    t0 = time.time()
    embedder = _init_embedder(cfg)
    query_vec = _embed_query(embedder, query)
    t_embed = time.time() - t0
    logger.debug(f"Query embedding generated in {t_embed:.2f}s")

    if progress:
        progress("Searching Qdrant")
    qdrant_filter = _build_filter(metadata_structure, raw_filters)
    search_params = _build_search_params(qcfg)
    t0 = time.time()
    candidates: List[_CandidateHit] = []
    query_failures: List[CollectionCompatibility] = []
    for target in searchable_targets:
        try:
            points = _execute_search(
                client=client,
                collection=target.collection_name,
                query_text=query,
                query_vec=query_vec,
                qcfg=qcfg,
                retrieval=retrieval,
                qdrant_filter=qdrant_filter,
                fetch_k=fetch_k,
                search_params=search_params,
            )
        except Exception as exc:  # pragma: no cover - defensive Qdrant wrapping
            failure = _compatibility_from_error(
                target,
                "query_failed",
                f"Qdrant search failed: {exc}",
                cfg,
                exists=True,
            )
            query_failures.append(failure)
            warnings.append(failure.warning_message())
            continue
        candidates.extend(_CandidateHit(target=target, point=point) for point in points)
    if query_failures:
        skipped_collections.extend(query_failures)
        searchable_targets = [
            target
            for target in searchable_targets
            if target.collection_name
            not in {failure.collection_name for failure in query_failures}
        ]
        collection_names = [target.collection_name for target in searchable_targets]
        collection_label = (
            collection_names[0]
            if len(collection_names) == 1
            else f"{len(collection_names)} collections"
        )
        collection_statuses.extend(query_failures)
        if not searchable_targets:
            details = "; ".join(
                f"{failure.collection_name}: {failure.reason}"
                for failure in query_failures
            )
            raise LookupError(
                "None of the requested collections could be searched. " + details
            )
    t_search = time.time() - t0
    logger.debug(
        f"Vector search completed in {t_search:.2f}s, retrieved {len(candidates)} candidates"
    )

    if not candidates:
        logger.warning("No results found for query.")
        logger.info("Try:")
        logger.info("  - Using different search terms")
        logger.info("  - Removing or loosening filters")
        logger.info(
            f"  - Checking collections have data: {', '.join(collection_names)}"
        )
        return QueryResponse(
            query=query,
            collection=collection_label,
            hits=[],
            timings=QueryTimings(
                embed_seconds=t_embed,
                search_seconds=t_search,
                group_seconds=0.0,
            ),
            candidates=0,
            collections=collection_names,
            warnings=warnings,
            skipped_collections=skipped_collections,
            collection_statuses=collection_statuses,
        )

    if progress:
        progress("Ranking matches")
    t0 = time.time()
    if len(searchable_targets) == 1:
        target = searchable_targets[0]
        grouped = _group_by_file(
            [candidate.point for candidate in candidates],
            metadata_structure,
            max_per_file=max_per_file,
        )
        selected_candidates = [
            _CandidateHit(target=target, point=point) for point in grouped[:top_k]
        ]
    else:
        selected_candidates = _select_global_candidates(
            candidates,
            metadata_structure=metadata_structure,
            max_per_file=max_per_file,
            top_k=top_k,
        )
    t_group = time.time() - t0

    logger.info(f"Returning {len(selected_candidates)} results")

    query_hits = [
        _query_hit_from_candidate(
            candidate=candidate,
            client=client,
            metadata_structure=metadata_structure,
            with_parent_window=with_parent_window,
            parent_window=parent_window,
        )
        for candidate in selected_candidates
    ]

    return QueryResponse(
        query=query,
        collection=collection_label,
        hits=query_hits,
        timings=QueryTimings(
            embed_seconds=t_embed,
            search_seconds=t_search,
            group_seconds=t_group,
        ),
        candidates=len(candidates),
        collections=collection_names,
        warnings=warnings,
        skipped_collections=skipped_collections,
        collection_statuses=collection_statuses,
    )


def _render_json_response(response: QueryResponse) -> None:
    results = []
    for hit in response.hits:
        result = {
            "score": hit.score,
            "file_path": hit.file_path,
            "collection": hit.collection or response.collection,
            "content": hit.content,
            "metadata": redact_metadata(hit.metadata),
        }
        if hit.repository_url:
            result["repository_url"] = hit.repository_url
        if hit.repository_name:
            result["repository_name"] = hit.repository_name
        if hit.repository_branch:
            result["repository_branch"] = hit.repository_branch
        if hit.expanded_context:
            result["expanded_context"] = hit.expanded_context
        results.append(result)

    output = {
        "query": response.query,
        "collection": response.collection,
        "collections": response.collections or [response.collection],
        "total_results": len(response.hits),
        "warnings": response.warnings,
        "skipped_collections": [
            status.to_dict() for status in response.skipped_collections
        ],
        "collection_statuses": [
            status.to_dict() for status in response.collection_statuses
        ],
        "results": results,
        "timing": {
            "embed_seconds": response.timings.embed_seconds,
            "search_seconds": response.timings.search_seconds,
            "group_seconds": response.timings.group_seconds,
        },
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


def _is_multi_collection_response(response: QueryResponse) -> bool:
    collections = response.collections or [response.collection]
    return len({collection for collection in collections if collection}) > 1


def _render_text_response(response: QueryResponse) -> None:
    for warning in response.warnings:
        print(f"warning: {warning}")
    for status in response.skipped_collections:
        print(
            "skipped collection: "
            f"{status.collection_name} ({status.status}) {status.reason}"
        )

    show_collection = _is_multi_collection_response(response)
    for n, hit in enumerate(response.hits, 1):
        print(f"\n#{n} score={hit.score:.4f} file={hit.file_path}")
        if show_collection:
            print(f"collection: {hit.collection or response.collection}")
        if hit.preview:
            print(f"matched snippet: {hit.preview}")

        if hit.expanded_context:
            truncated = hit.expanded_context[:2000]
            if len(hit.expanded_context) > 2000:
                truncated += "\n... (truncated at 2000 chars)"
            print("\n[expanded_context]\n" + truncated)


def run_query(
    config_path: str,
    query: str,
    limit: Optional[int] = None,
    with_parent_window: bool = False,
    verbose: bool = False,
    quiet: bool = False,
    output_format: str = "text",
    collection: Optional[str] = None,
    repo_list: Optional[str] = None,
) -> int:
    """Run a retrieval query against a populated Qdrant collection."""
    if output_format not in {"text", "json"}:
        raise ValueError("output_format must be 'text' or 'json'")

    try:
        response = execute_query(
            config_path=config_path,
            query=query,
            limit=limit,
            with_parent_window=with_parent_window,
            verbose=verbose,
            quiet=quiet,
            collection=collection,
            repo_list=repo_list,
        )
    except (LookupError, ValueError) as exc:
        logging.getLogger(__name__).error(str(exc))
        return 1

    if output_format == "json":
        _render_json_response(response)
    else:
        _render_text_response(response)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config.yaml used for ingestion")
    parser.add_argument("--query", required=True, help="Query text")
    parser.add_argument(
        "--limit", type=int, default=None, help="Override retrieval.top_k"
    )
    parser.add_argument(
        "--with-parent-window",
        action="store_true",
        help="Include expanded context window around each selected hit (requires parent_id metadata).",
    )
    parser.add_argument("--collection", help="Qdrant collection override")
    parser.add_argument(
        "--repo-list",
        help="YAML repository list; searches all listed collections by default",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress info messages, show only results and errors",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format: text (human-readable) or json (machine-readable)",
    )
    args = parser.parse_args()

    return run_query(
        config_path=args.config,
        query=args.query,
        limit=args.limit,
        with_parent_window=args.with_parent_window,
        verbose=args.verbose,
        quiet=args.quiet,
        output_format=args.format,
        collection=args.collection,
        repo_list=args.repo_list,
    )


if __name__ == "__main__":
    sys.exit(main())
