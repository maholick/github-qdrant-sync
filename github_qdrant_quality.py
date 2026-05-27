#!/usr/bin/env python3
"""Retrieval quality, Qdrant health, and safe improvement helpers."""

from __future__ import annotations

import json
import shutil
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import yaml
from qdrant_client.http import models as qdrant_models
from rich import box
from rich.table import Table
from ruamel.yaml import YAML

from github_to_qdrant import ConfigLoader
from rag_retrieval import (
    CollectionTarget,
    QueryResponse,
    _init_qdrant_client,
    execute_query,
    inspect_collection_compatibility,
    resolve_collection_targets,
)


DEFAULT_PAYLOAD_INDEX_FIELDS = [
    {"name": "repo_id", "type": "keyword"},
    {"name": "file_id", "type": "keyword"},
    {"name": "file_upload_id", "type": "keyword"},
    {"name": "repository", "type": "keyword"},
    {"name": "source", "type": "keyword"},
    {"name": "source_type", "type": "keyword"},
    {"name": "extraction_method", "type": "keyword"},
    {"name": "content_hash", "type": "keyword"},
    {"name": "file_hash", "type": "keyword"},
    {"name": "embedding_provider", "type": "keyword"},
    {"name": "embedding_model", "type": "keyword"},
    {"name": "page_number", "type": "integer"},
]
CONTENT_FIELDS = ("page_content", "content", "document", "text")
METADATA_SOURCE_FIELDS = ("source", "file_path")
DEFAULT_BENCHMARK_CASE_CANDIDATES = (Path("eval.yaml"), Path("eval.yml"))

yaml_rt = YAML()
yaml_rt.preserve_quotes = True


@dataclass
class QualityThresholds:
    """Benchmark thresholds for aggregate and per-case pass/fail decisions."""

    pass_rate: float = 0.8
    expected_source_top_k: int = 5
    min_top_score: float = 0.4
    min_keyword_coverage: float = 0.5


@dataclass
class BenchmarkCase:
    """One retrieval benchmark case."""

    case_id: str
    query: str
    collection: Optional[str] = None
    expected_sources: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


@dataclass
class BenchmarkCaseResult:
    """Metrics for one benchmark case."""

    case_id: str
    query: str
    collection: str
    passed: bool
    top_score: float
    hit_at_1: bool
    hit_at_5: bool
    hit_at_10: bool
    mrr: float
    expected_source_rank: Optional[int]
    keyword_coverage: float
    latency_seconds: float
    sources: List[str]
    error: str = ""
    warnings: List[str] = field(default_factory=list)
    skipped_collections: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    """Aggregate benchmark result."""

    cases_path: str
    thresholds: QualityThresholds
    results: List[BenchmarkCaseResult]
    pass_rate: float
    passed: bool


@dataclass
class DoctorFinding:
    """One Qdrant/index health finding."""

    level: str
    check: str
    message: str
    collection: str = ""
    fixable: bool = False


@dataclass
class DoctorReport:
    """Qdrant/index health report."""

    config_path: str
    targets: List[CollectionTarget]
    findings: List[DoctorFinding]

    @property
    def ok(self) -> bool:
        """Return whether no blocking health errors were found."""
        return not any(finding.level == "ERROR" for finding in self.findings)


@dataclass
class ImproveAction:
    """One proposed or applied improvement action."""

    category: str
    action: str
    status: str
    message: str
    requires_reingest: bool = False


@dataclass
class ImproveReport:
    """Safe improvement report."""

    config_path: str
    applied: bool
    backup_path: Optional[str]
    actions: List[ImproveAction]
    doctor: DoctorReport
    benchmark: Optional[BenchmarkReport] = None


def load_config(config_path: str, quiet: bool = True) -> Dict[str, Any]:
    """Load YAML/JSON config with environment variables resolved."""
    ConfigLoader.load_env_for_config(config_path, quiet=quiet)
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        if path.suffix.lower() == ".json":
            loaded = json.load(config_file)
        else:
            loaded = yaml.safe_load(config_file)
    return ConfigLoader._resolve_env_vars(loaded or {})  # pylint: disable=protected-access


def resolve_benchmark_cases(
    cases_path: Optional[Union[str, Path]],
    config_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve an explicit or default benchmark YAML file."""
    if cases_path is not None:
        resolved = Path(cases_path).expanduser()
        if not resolved.exists() or not resolved.is_file():
            raise ValueError(f"Benchmark file not found: {resolved}")
        return resolved

    candidates: List[Path] = []
    if config_path is not None:
        config_dir = Path(config_path).expanduser().parent
        candidates.extend(
            config_dir / candidate.name
            for candidate in DEFAULT_BENCHMARK_CASE_CANDIDATES
        )
    candidates.extend(DEFAULT_BENCHMARK_CASE_CANDIDATES)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists() and resolved.is_file():
            return resolved

    labels = ", ".join(
        candidate.name for candidate in DEFAULT_BENCHMARK_CASE_CANDIDATES
    )
    raise ValueError(
        f"No benchmark cases file found. Expected {labels} in the current "
        "directory or next to the config, or pass --cases PATH."
    )


def _load_roundtrip_config(config_path: Path) -> Any:
    if config_path.suffix.lower() == ".json":
        with config_path.open("r", encoding="utf-8") as config_file:
            return json.load(config_file)
    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml_rt.load(config_file) or {}


def _write_roundtrip_config(config_path: Path, config: Any) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.suffix.lower() == ".json":
        with config_path.open("w", encoding="utf-8") as config_file:
            json.dump(_plain_config(config), config_file, indent=2, ensure_ascii=False)
            config_file.write("\n")
        return
    with config_path.open("w", encoding="utf-8") as config_file:
        yaml_rt.dump(config, config_file)


def _plain_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_config(item) for item in value]
    return value


def _metadata_structure(config: Dict[str, Any]) -> str:
    return str(config.get("payload", {}).get("metadata_structure", "nested"))


def _field_path(name: str, metadata_structure: str) -> str:
    return f"metadata.{name}" if metadata_structure == "nested" else name


def _payload_index_fields(config: Dict[str, Any]) -> List[Dict[str, str]]:
    configured = config.get("qdrant", {}).get("payload_indexes", {}).get("fields")
    if isinstance(configured, list) and configured:
        return [
            {"name": str(item.get("name")), "type": str(item.get("type", "keyword"))}
            for item in configured
            if isinstance(item, dict) and item.get("name")
        ]
    return deepcopy(DEFAULT_PAYLOAD_INDEX_FIELDS)


def _schema_field_names(info: Any) -> set[str]:
    schema = getattr(info, "payload_schema", None)
    if isinstance(schema, dict):
        return set(schema)
    return set()


def _vector_params(info: Any, vector_name: Optional[str]) -> Any:
    config = getattr(info, "config", None)
    params = getattr(config, "params", None)
    vectors = getattr(params, "vectors", None)
    if isinstance(vectors, dict):
        if vector_name:
            return vectors.get(vector_name)
        if len(vectors) == 1:
            return next(iter(vectors.values()))
        return None
    return vectors


def _value_name(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _point_count(info: Any) -> Optional[int]:
    for name in ("points_count", "vectors_count", "indexed_vectors_count"):
        value = getattr(info, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _sample_payload(client: Any, collection: str) -> Dict[str, Any]:
    records, _next_page = client.scroll(
        collection_name=collection,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not records:
        return {}
    return records[0].payload or {}


def _payload_has_content(payload: Dict[str, Any]) -> bool:
    return any(bool(payload.get(field)) for field in CONTENT_FIELDS)


def _payload_has_source(payload: Dict[str, Any], metadata_structure: str) -> bool:
    metadata = (
        payload.get("metadata", {}) if metadata_structure == "nested" else payload
    )
    return any(bool(metadata.get(field)) for field in METADATA_SOURCE_FIELDS)


def _schema_type(type_name: str) -> Optional[qdrant_models.PayloadSchemaType]:
    type_map = {
        "keyword": qdrant_models.PayloadSchemaType.KEYWORD,
        "integer": qdrant_models.PayloadSchemaType.INTEGER,
        "float": qdrant_models.PayloadSchemaType.FLOAT,
        "bool": qdrant_models.PayloadSchemaType.BOOL,
        "datetime": qdrant_models.PayloadSchemaType.DATETIME,
        "text": qdrant_models.PayloadSchemaType.TEXT,
        "uuid": qdrant_models.PayloadSchemaType.UUID,
    }
    return type_map.get(type_name.lower())


def _create_payload_indexes(
    client: Any,
    collection: str,
    config: Dict[str, Any],
    existing_fields: Optional[set[str]] = None,
) -> int:
    metadata_structure = _metadata_structure(config)
    existing = existing_fields or set()
    created = 0
    for item in _payload_index_fields(config):
        field_name = _field_path(item["name"], metadata_structure)
        if field_name in existing:
            continue
        schema = _schema_type(item["type"])
        if schema is None:
            continue
        client.create_payload_index(
            collection_name=collection,
            field_name=field_name,
            field_schema=schema,
            wait=True,
        )
        created += 1
    return created


def run_doctor(
    config_path: str,
    repo_list: Optional[str] = None,
    collection: Optional[str] = None,
    apply_indexes: bool = False,
    yes: bool = False,
) -> DoctorReport:
    """Check Qdrant collection and payload-index health."""
    config = load_config(config_path)
    targets = resolve_collection_targets(
        config, collection=collection, repo_list=repo_list
    )
    client = _init_qdrant_client(config.get("qdrant", {}))
    qdrant_config = config.get("qdrant", {})
    expected_size = qdrant_config.get("vector_size")
    expected_distance = str(qdrant_config.get("distance", "Cosine")).lower()
    vector_name = qdrant_config.get("vector_name")
    metadata_structure = _metadata_structure(config)
    indexes_enabled = bool(qdrant_config.get("payload_indexes", {}).get("enabled"))
    findings: List[DoctorFinding] = []
    compatibility_by_name = {
        status.collection_name: status
        for status in inspect_collection_compatibility(config, targets, client)
    }

    for target in targets:
        name = target.collection_name
        try:
            exists = client.collection_exists(collection_name=name)
        except Exception as exc:  # pragma: no cover - defensive network wrapping
            findings.append(
                DoctorFinding(
                    "ERROR", "connection", f"Qdrant check failed: {exc}", name
                )
            )
            continue

        if not exists:
            findings.append(
                DoctorFinding(
                    "ERROR",
                    "collection",
                    "Collection does not exist. Run ingestion first.",
                    name,
                )
            )
            continue

        findings.append(DoctorFinding("OK", "collection", "Collection exists.", name))
        info = client.get_collection(collection_name=name)

        count = _point_count(info)
        if count is None:
            findings.append(
                DoctorFinding("WARN", "points", "Point count is unavailable.", name)
            )
        elif count <= 0:
            findings.append(
                DoctorFinding(
                    "ERROR",
                    "points",
                    "Collection is empty. Run ingestion before benchmarking.",
                    name,
                )
            )
        else:
            findings.append(
                DoctorFinding("OK", "points", f"Collection has {count} points.", name)
            )

        params = _vector_params(info, str(vector_name) if vector_name else None)
        if params is None:
            findings.append(
                DoctorFinding(
                    "ERROR",
                    "vectors",
                    "Vector configuration was not found or named vector is missing.",
                    name,
                )
            )
        else:
            actual_size = getattr(params, "size", None)
            actual_distance = _value_name(getattr(params, "distance", "")).lower()
            if expected_size is not None and int(actual_size or 0) != int(
                expected_size
            ):
                findings.append(
                    DoctorFinding(
                        "ERROR",
                        "vector_size",
                        f"Vector size is {actual_size}, config expects {expected_size}.",
                        name,
                    )
                )
            else:
                findings.append(
                    DoctorFinding(
                        "OK", "vector_size", "Vector size matches config.", name
                    )
                )
            if expected_distance and expected_distance not in actual_distance:
                findings.append(
                    DoctorFinding(
                        "ERROR",
                        "distance",
                        f"Distance is {_value_name(getattr(params, 'distance', 'unknown'))}, "
                        f"config expects {qdrant_config.get('distance')}.",
                        name,
                    )
                )
            else:
                findings.append(
                    DoctorFinding(
                        "OK", "distance", "Distance metric matches config.", name
                    )
                )

        try:
            payload = _sample_payload(client, name)
        except Exception as exc:  # pragma: no cover - defensive network wrapping
            payload = {}
            findings.append(
                DoctorFinding(
                    "WARN", "payload", f"Could not sample payload: {exc}", name
                )
            )
        if payload:
            if _payload_has_content(payload):
                findings.append(
                    DoctorFinding(
                        "OK", "payload_content", "Sample payload has content.", name
                    )
                )
            else:
                findings.append(
                    DoctorFinding(
                        "WARN",
                        "payload_content",
                        "Sample payload lacks page_content/content/document/text.",
                        name,
                    )
                )
            if _payload_has_source(payload, metadata_structure):
                findings.append(
                    DoctorFinding(
                        "OK",
                        "payload_source",
                        "Sample payload has source metadata.",
                        name,
                    )
                )
            else:
                findings.append(
                    DoctorFinding(
                        "WARN",
                        "payload_source",
                        "Sample payload lacks source/file_path metadata.",
                        name,
                    )
                )

        compatibility = compatibility_by_name.get(name)
        if compatibility and compatibility.status == "embedding_mismatch":
            findings.append(
                DoctorFinding(
                    "ERROR",
                    "embedding_compatibility",
                    compatibility.reason,
                    name,
                )
            )
        elif compatibility and compatibility.status == "metadata_unknown":
            findings.append(
                DoctorFinding(
                    "WARN",
                    "embedding_compatibility",
                    compatibility.reason,
                    name,
                )
            )
        elif (
            compatibility
            and compatibility.status == "usable"
            and (compatibility.actual_provider or compatibility.actual_model)
        ):
            findings.append(
                DoctorFinding(
                    "OK",
                    "embedding_compatibility",
                    "Embedding provider/model metadata matches config.",
                    name,
                )
            )

        schema_fields = _schema_field_names(info)
        expected_fields = {
            _field_path(item["name"], metadata_structure)
            for item in _payload_index_fields(config)
        }
        missing_indexes = sorted(expected_fields - schema_fields)
        if not indexes_enabled:
            findings.append(
                DoctorFinding(
                    "WARN",
                    "payload_indexes",
                    "Payload indexes are disabled in config.",
                    name,
                    fixable=True,
                )
            )
        elif missing_indexes:
            findings.append(
                DoctorFinding(
                    "WARN",
                    "payload_indexes",
                    f"Missing payload indexes: {', '.join(missing_indexes[:8])}"
                    + ("..." if len(missing_indexes) > 8 else ""),
                    name,
                    fixable=True,
                )
            )
        else:
            findings.append(
                DoctorFinding(
                    "OK", "payload_indexes", "Configured payload indexes exist.", name
                )
            )

        if apply_indexes:
            if not yes:
                findings.append(
                    DoctorFinding(
                        "WARN",
                        "payload_indexes_apply",
                        "Indexes not created because --yes was not passed.",
                        name,
                        fixable=True,
                    )
                )
            else:
                created = _create_payload_indexes(client, name, config, schema_fields)
                findings.append(
                    DoctorFinding(
                        "OK",
                        "payload_indexes_apply",
                        f"Created {created} payload index(es).",
                        name,
                    )
                )

    return DoctorReport(config_path=config_path, targets=targets, findings=findings)


def load_benchmark_file(
    cases_path: str,
) -> tuple[QualityThresholds, List[BenchmarkCase]]:
    """Load benchmark thresholds and cases from YAML."""
    with open(cases_path, "r", encoding="utf-8") as cases_file:
        data = yaml.safe_load(cases_file) or {}
    if not isinstance(data, dict):
        raise ValueError("Benchmark file must contain a mapping")
    raw_thresholds = data.get("thresholds") or {}
    if not isinstance(raw_thresholds, dict):
        raise ValueError("benchmark.thresholds must be a mapping")
    thresholds = QualityThresholds(
        pass_rate=float(raw_thresholds.get("pass_rate", 0.8)),
        expected_source_top_k=int(raw_thresholds.get("expected_source_top_k", 5)),
        min_top_score=float(raw_thresholds.get("min_top_score", 0.4)),
        min_keyword_coverage=float(raw_thresholds.get("min_keyword_coverage", 0.5)),
    )
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Benchmark file must contain a non-empty cases list")

    cases: List[BenchmarkCase] = []
    for index, item in enumerate(raw_cases, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Benchmark case {index} must be a mapping")
        query = str(item.get("query") or "").strip()
        if not query:
            raise ValueError(f"Benchmark case {index} is missing query")
        case_id = str(item.get("id") or f"case-{index}")
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                query=query,
                collection=item.get("collection"),
                expected_sources=[
                    str(source) for source in item.get("expected_sources", [])
                ],
                keywords=[str(keyword) for keyword in item.get("keywords", [])],
            )
        )
    return thresholds, cases


def _first_expected_source_rank(
    response: QueryResponse, expected_sources: Iterable[str]
) -> Optional[int]:
    expected = [source.lower() for source in expected_sources if source]
    if not expected:
        return None
    for index, hit in enumerate(response.hits, 1):
        haystack = " ".join(
            [
                hit.file_path,
                str(hit.metadata.get("source", "")),
                str(hit.metadata.get("file_path", "")),
            ]
        ).lower()
        if any(source in haystack for source in expected):
            return index
    return None


def _keyword_coverage(response: QueryResponse, keywords: Iterable[str]) -> float:
    keyword_list = [keyword.lower() for keyword in keywords if keyword]
    if not keyword_list:
        return 1.0
    haystack = "\n".join(
        f"{hit.file_path}\n{hit.preview}\n{hit.content}" for hit in response.hits
    ).lower()
    found = sum(1 for keyword in keyword_list if keyword in haystack)
    return found / len(keyword_list)


def _case_result(
    case: BenchmarkCase,
    response: QueryResponse,
    thresholds: QualityThresholds,
    latency_seconds: float,
) -> BenchmarkCaseResult:
    rank = _first_expected_source_rank(response, case.expected_sources)
    coverage = _keyword_coverage(response, case.keywords)
    top_score = float(response.hits[0].score) if response.hits else 0.0
    source_ok = not case.expected_sources or (
        rank is not None and rank <= thresholds.expected_source_top_k
    )
    score_ok = top_score >= thresholds.min_top_score
    keyword_ok = coverage >= thresholds.min_keyword_coverage
    passed = bool(response.hits) and source_ok and score_ok and keyword_ok
    return BenchmarkCaseResult(
        case_id=case.case_id,
        query=case.query,
        collection=response.collection,
        passed=passed,
        top_score=top_score,
        hit_at_1=bool(rank is not None and rank <= 1),
        hit_at_5=bool(rank is not None and rank <= 5),
        hit_at_10=bool(rank is not None and rank <= 10),
        mrr=(1 / rank) if rank else 0.0,
        expected_source_rank=rank,
        keyword_coverage=coverage,
        latency_seconds=latency_seconds,
        sources=[hit.file_path for hit in response.hits[:10]],
        warnings=response.warnings,
        skipped_collections=[
            status.to_dict() for status in response.skipped_collections
        ],
    )


def run_benchmark(
    config_path: str,
    cases_path: str,
    repo_list: Optional[str] = None,
    collection: Optional[str] = None,
    limit: Optional[int] = None,
    fail_under: Optional[float] = None,
) -> BenchmarkReport:
    """Run retrieval benchmark cases and return aggregate metrics."""
    thresholds, cases = load_benchmark_file(cases_path)
    effective_limit = max(limit or 10, thresholds.expected_source_top_k, 10)
    results: List[BenchmarkCaseResult] = []

    for case in cases:
        started = time.time()
        try:
            response = execute_query(
                config_path=config_path,
                query=case.query,
                limit=effective_limit,
                with_parent_window=False,
                quiet=True,
                collection=collection or case.collection,
                repo_list=repo_list,
            )
            results.append(
                _case_result(case, response, thresholds, time.time() - started)
            )
        except (LookupError, ValueError, SystemExit) as exc:
            results.append(
                BenchmarkCaseResult(
                    case_id=case.case_id,
                    query=case.query,
                    collection=collection or case.collection or "",
                    passed=False,
                    top_score=0.0,
                    hit_at_1=False,
                    hit_at_5=False,
                    hit_at_10=False,
                    mrr=0.0,
                    expected_source_rank=None,
                    keyword_coverage=0.0,
                    latency_seconds=time.time() - started,
                    sources=[],
                    error=str(exc),
                )
            )

    pass_rate = (
        sum(1 for result in results if result.passed) / len(results) if results else 0.0
    )
    required_pass_rate = thresholds.pass_rate if fail_under is None else fail_under
    return BenchmarkReport(
        cases_path=cases_path,
        thresholds=thresholds,
        results=results,
        pass_rate=pass_rate,
        passed=pass_rate >= required_pass_rate,
    )


def _ensure_payload_index_config(config: Any) -> bool:
    qdrant = config.setdefault("qdrant", {})
    payload_indexes = qdrant.setdefault("payload_indexes", {})
    changed = False
    if payload_indexes.get("enabled") is not True:
        payload_indexes["enabled"] = True
        changed = True
    if payload_indexes.get("apply_to_existing_collections") is not True:
        payload_indexes["apply_to_existing_collections"] = True
        changed = True
    fields = payload_indexes.get("fields")
    if not isinstance(fields, list) or not fields:
        payload_indexes["fields"] = deepcopy(DEFAULT_PAYLOAD_INDEX_FIELDS)
        changed = True
    return changed


def _safe_set(config: Any, dotted_path: str, value: Any) -> bool:
    parts = dotted_path.split(".")
    current = config
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    if current.get(parts[-1]) == value:
        return False
    current[parts[-1]] = value
    return True


def _benchmark_needs_keyword_help(report: Optional[BenchmarkReport]) -> bool:
    if report is None:
        return False
    for result in report.results:
        if len(result.query.strip()) <= 4 or result.keyword_coverage < 0.5:
            return True
    return False


def run_improve(
    config_path: str,
    cases_path: Optional[str] = None,
    repo_list: Optional[str] = None,
    collection: Optional[str] = None,
    apply: bool = False,  # pylint: disable=redefined-builtin
    yes: bool = False,
) -> ImproveReport:
    """Generate or apply safe quality improvements."""
    doctor = run_doctor(
        config_path=config_path,
        repo_list=repo_list,
        collection=collection,
        apply_indexes=False,
        yes=False,
    )
    benchmark = (
        run_benchmark(
            config_path=config_path,
            cases_path=cases_path,
            repo_list=repo_list,
            collection=collection,
        )
        if cases_path
        else None
    )
    actions: List[ImproveAction] = []
    config_file = Path(config_path)
    working_config = _load_roundtrip_config(config_file)
    changed_config = False

    has_index_findings = any(
        finding.check == "payload_indexes" and finding.fixable
        for finding in doctor.findings
    )
    if has_index_findings:
        changed_config = _ensure_payload_index_config(working_config) or changed_config
        actions.append(
            ImproveAction(
                "index",
                "payload_indexes",
                "applied" if apply and yes else "preview",
                "Enable/configure payload indexes for faster filtered retrieval.",
            )
        )

    retrieval = working_config.get("retrieval", {})
    top_k = int(retrieval.get("top_k", 10))
    desired_fetch_k = max(int(retrieval.get("fetch_k", 0) or 0), top_k * 6, 80)
    if _safe_set(working_config, "retrieval.fetch_k", desired_fetch_k):
        changed_config = True
        actions.append(
            ImproveAction(
                "retrieval",
                "retrieval.fetch_k",
                "applied" if apply and yes else "preview",
                f"Set retrieval.fetch_k to {desired_fetch_k} for better ranking depth.",
            )
        )

    desired_max_chunks = max(int(retrieval.get("max_chunks_per_file", 3) or 3), 5)
    if _safe_set(working_config, "retrieval.max_chunks_per_file", desired_max_chunks):
        changed_config = True
        actions.append(
            ImproveAction(
                "retrieval",
                "retrieval.max_chunks_per_file",
                "applied" if apply and yes else "preview",
                f"Set max_chunks_per_file to {desired_max_chunks} to avoid over-pruning.",
            )
        )

    if top_k < 10 and _safe_set(working_config, "retrieval.top_k", 10):
        changed_config = True
        actions.append(
            ImproveAction(
                "retrieval",
                "retrieval.top_k",
                "applied" if apply and yes else "preview",
                "Set retrieval.top_k to 10 for broader benchmark evidence.",
            )
        )

    if _benchmark_needs_keyword_help(benchmark):
        actions.append(
            ImproveAction(
                "retrieval",
                "hybrid_search",
                "manual",
                "Short/acronym queries would likely benefit from future hybrid lexical + vector search.",
            )
        )

    if any(
        finding.check
        in {
            "vector_size",
            "distance",
            "vectors",
            "embedding_compatibility",
        }
        and finding.level == "ERROR"
        for finding in doctor.findings
    ):
        actions.append(
            ImproveAction(
                "index",
                "reingestion_required",
                "manual",
                "Embedding/vector compatibility issues require explicit reingestion.",
                requires_reingest=True,
            )
        )

    backup_path = None
    applied = False
    if apply:
        if not yes:
            actions.append(
                ImproveAction(
                    "safety",
                    "confirm",
                    "blocked",
                    "No changes applied because --yes was not passed.",
                )
            )
        elif changed_config:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup = config_file.with_name(f"{config_file.name}.{timestamp}.bak")
            shutil.copy2(config_file, backup)
            _write_roundtrip_config(config_file, working_config)
            backup_path = str(backup)
            applied = True
            # Apply indexes after the config has been updated, still idempotently.
            run_doctor(
                config_path=config_path,
                repo_list=repo_list,
                collection=collection,
                apply_indexes=True,
                yes=True,
            )

    if not actions:
        actions.append(
            ImproveAction(
                "quality",
                "none",
                "ok",
                "No safe automatic improvements were found.",
            )
        )

    return ImproveReport(
        config_path=config_path,
        applied=applied,
        backup_path=backup_path,
        actions=actions,
        doctor=doctor,
        benchmark=benchmark,
    )


def doctor_to_dict(report: DoctorReport) -> Dict[str, Any]:
    """Return a JSON-safe doctor report."""
    return {
        "ok": report.ok,
        "config_path": report.config_path,
        "targets": [target.collection_name for target in report.targets],
        "findings": [finding.__dict__ for finding in report.findings],
    }


def benchmark_to_dict(report: BenchmarkReport) -> Dict[str, Any]:
    """Return a JSON-safe benchmark report."""
    return {
        "passed": report.passed,
        "pass_rate": report.pass_rate,
        "cases_path": report.cases_path,
        "thresholds": report.thresholds.__dict__,
        "results": [result.__dict__ for result in report.results],
    }


def improve_to_dict(report: ImproveReport) -> Dict[str, Any]:
    """Return a JSON-safe improvement report."""
    return {
        "config_path": report.config_path,
        "applied": report.applied,
        "backup_path": report.backup_path,
        "actions": [action.__dict__ for action in report.actions],
        "doctor": doctor_to_dict(report.doctor),
        "benchmark": benchmark_to_dict(report.benchmark) if report.benchmark else None,
    }


def doctor_table(report: DoctorReport) -> Table:
    """Build a Rich doctor report table."""
    table = Table(
        title="Index Doctor",
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("Level", no_wrap=True)
    table.add_column("Collection", style="cyan")
    table.add_column("Check", style="bold")
    table.add_column("Message")
    for finding in report.findings:
        style = {
            "OK": "green",
            "WARN": "yellow",
            "ERROR": "red",
        }.get(finding.level, "")
        table.add_row(
            f"[{style}]{finding.level}[/{style}]" if style else finding.level,
            finding.collection,
            finding.check,
            finding.message,
        )
    return table


def benchmark_table(report: BenchmarkReport) -> Table:
    """Build a Rich benchmark report table."""
    table = Table(
        title=f"Retrieval Benchmark ({report.pass_rate:.0%} pass)",
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("Case", style="cyan")
    table.add_column("Pass", no_wrap=True)
    table.add_column("Score", justify="right")
    table.add_column("Rank", justify="right")
    table.add_column("Keywords", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Top Source")
    table.add_column("Notes")
    for result in report.results:
        skipped = "; ".join(
            f"{item.get('collection')}:{item.get('status')}"
            for item in result.skipped_collections
        )
        notes = result.error or skipped or "; ".join(result.warnings)
        table.add_row(
            result.case_id,
            "[green]yes[/green]" if result.passed else "[red]no[/red]",
            f"{result.top_score:.4f}",
            str(result.expected_source_rank or "-"),
            f"{result.keyword_coverage:.0%}",
            f"{result.latency_seconds:.2f}s",
            result.error or (result.sources[0] if result.sources else "-"),
            notes,
        )
    return table


def improve_table(report: ImproveReport) -> Table:
    """Build a Rich improvement report table."""
    table = Table(
        title="Safe Improvements",
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("Category", style="cyan")
    table.add_column("Action", style="bold")
    table.add_column("Status")
    table.add_column("Message")
    for action in report.actions:
        table.add_row(
            action.category,
            action.action,
            action.status,
            action.message
            + (
                " [yellow](requires reingestion)[/yellow]"
                if action.requires_reingest
                else ""
            ),
        )
    return table
