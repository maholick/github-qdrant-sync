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
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as m

from github_to_qdrant import (
    MistralEmbeddingClient,
    SentenceTransformerClient,
    create_qdrant_client_from_config,
    get_metadata_structure,
    marker_exclusion_filter,
    payload_field_path,
)
from langchain_openai import AzureOpenAIEmbeddings


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _resolve_env_vars(cfg)


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
    metadata_structure: str,
    raw_filters: Optional[Dict[str, Any]],
    *,
    exclude_markers: bool = True,
) -> Optional[m.Filter]:
    must: List[m.FieldCondition] = []
    must_not = []
    if exclude_markers:
        marker_filter = marker_exclusion_filter(metadata_structure)
        must_not.extend(marker_filter.must_not or [])

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
) -> List[m.ScoredPoint]:
    """Execute dense-only or hybrid dense+sparse search."""
    vector_name = qcfg.get("vector_name")
    retrieval_mode = str(retrieval.get("mode", "dense")).lower()

    if retrieval_mode == "hybrid":
        sparse_cfg = qcfg.get("sparse_vector", {})
        if not vector_name:
            raise SystemExit(
                "Hybrid retrieval requires qdrant.vector_name for the dense vector "
                "(for example: vector_name: dense)."
            )
        if not sparse_cfg.get("enabled", False):
            raise SystemExit(
                "Hybrid retrieval requires qdrant.sparse_vector.enabled: true."
            )

        fusion_name = str(retrieval.get("fusion", "rrf")).lower()
        fusion = m.Fusion.RRF if fusion_name == "rrf" else m.Fusion.DBSF
        sparse_name = sparse_cfg.get("name", "sparse")
        sparse_model = sparse_cfg.get("model", "qdrant/bm25")

        return client.query_points(
            collection_name=collection,
            prefetch=[
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
            query=m.FusionQuery(fusion=fusion),
            query_filter=qdrant_filter,
            limit=fetch_k,
            with_payload=True,
            with_vectors=False,
        ).points

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

    return client.query_points(**query_kwargs).points


def main() -> None:
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

    # Configure logging
    log_level = (
        logging.WARNING
        if args.quiet
        else (logging.DEBUG if args.verbose else logging.INFO)
    )
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    load_dotenv()
    cfg = _load_config(args.config)

    # Validate required config sections
    if "qdrant" not in cfg:
        logger.error("Missing required 'qdrant' section in config")
        sys.exit(1)
    if "embedding_provider" not in cfg:
        logger.error("Missing required 'embedding_provider' section in config")
        sys.exit(1)

    qcfg = cfg["qdrant"]
    if "collection_name" not in qcfg:
        logger.error("Missing required 'qdrant.collection_name' in config")
        sys.exit(1)

    collection = qcfg["collection_name"]
    retrieval = cfg.get("retrieval", {})
    metadata_structure = get_metadata_structure(cfg)

    top_k = int(args.limit or retrieval.get("top_k", 10))
    # fetch_k should be meaningfully larger than top_k for grouping to work properly
    fetch_k = int(retrieval.get("fetch_k", max(top_k * 4, 40)))
    max_per_file = int(retrieval.get("max_chunks_per_file", 3))
    raw_filters = retrieval.get("filters")
    parent_window = int(retrieval.get("parent_window", 2))

    # Initialize Qdrant client using the same config conventions as ingestion.
    client = create_qdrant_client_from_config(qcfg)

    # Verify collection exists before querying
    if not client.collection_exists(collection_name=collection):
        logger.error(f"Collection '{collection}' does not exist.")
        logger.error(
            "Run github_to_qdrant.py first to create and populate the collection."
        )
        sys.exit(1)

    logger.info(f"Querying collection: {collection}")
    logger.info(f"Query: {args.query}")

    # Generate embedding for query
    t0 = time.time()
    embedder = _init_embedder(cfg)
    query_vec = _embed_query(embedder, args.query)
    t_embed = time.time() - t0
    logger.debug(f"Query embedding generated in {t_embed:.2f}s")

    # Execute vector or hybrid search
    qdrant_filter = _build_filter(metadata_structure, raw_filters)
    t0 = time.time()
    hits = _execute_search(
        client=client,
        collection=collection,
        query_text=args.query,
        query_vec=query_vec,
        qcfg=qcfg,
        retrieval=retrieval,
        qdrant_filter=qdrant_filter,
        fetch_k=fetch_k,
    )
    t_search = time.time() - t0
    logger.debug(
        f"Vector search completed in {t_search:.2f}s, retrieved {len(hits)} candidates"
    )

    if not hits:
        logger.warning(f"No results found for query: {args.query}")
        logger.info("Try:")
        logger.info("  - Using different search terms")
        logger.info("  - Removing or loosening filters")
        logger.info(f"  - Checking collection has data: collection '{collection}'")
        sys.exit(0)

    # Group and select results
    t0 = time.time()
    grouped = _group_by_file(hits, metadata_structure, max_per_file=max_per_file)
    selected = grouped[:top_k]
    t_group = time.time() - t0
    logger.debug(
        f"Grouping completed in {t_group:.2f}s, selected {len(selected)} results from {len(set(_extract_meta(h.payload, metadata_structure).get('file_path') or _extract_meta(h.payload, metadata_structure).get('source') for h in hits))} files"
    )

    logger.info(f"Returning {len(selected)} results")

    # Format output based on requested format
    if args.format == "json":
        # JSON output for programmatic use
        results = []
        for hit in selected:
            payload = hit.payload or {}
            meta = _extract_meta(payload, metadata_structure)
            result = {
                "score": hit.score,
                "file_path": meta.get("file_path") or meta.get("source") or "unknown",
                "content": payload.get("page_content") or payload.get("content") or "",
                "metadata": meta,
            }
            if args.with_parent_window:
                expanded = _expand_parent_window(
                    client=client,
                    collection=collection,
                    metadata_structure=metadata_structure,
                    hit=hit,
                    window=parent_window,
                )
                if expanded:
                    result["expanded_context"] = expanded
            results.append(result)

        output = {
            "query": args.query,
            "total_results": len(selected),
            "results": results,
            "timing": {
                "embed_seconds": t_embed,
                "search_seconds": t_search,
                "group_seconds": t_group,
            },
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        # Text output for human readability
        for n, hit in enumerate(selected, 1):
            payload = hit.payload or {}
            meta = _extract_meta(payload, metadata_structure)
            file_path = meta.get("file_path") or meta.get("source") or "unknown"
            preview = meta.get("preview", "")
            print(f"\n#{n} score={hit.score:.4f} file={file_path}")
            if preview:
                print(f"preview: {preview}")

            if args.with_parent_window:
                expanded = _expand_parent_window(
                    client=client,
                    collection=collection,
                    metadata_structure=metadata_structure,
                    hit=hit,
                    window=parent_window,
                )
                if expanded:
                    truncated = expanded[:2000]
                    if len(expanded) > 2000:
                        truncated += "\n... (truncated at 2000 chars)"
                    print("\n[expanded_context]\n" + truncated)


if __name__ == "__main__":
    main()
