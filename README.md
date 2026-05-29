# 🚀 GitHub to Qdrant Vector Processing Pipeline

**High-performance document processing pipeline that transforms GitHub repositories containing markdown, PDFs, and 150+ text file types into searchable vector databases for AI applications. Now with multi-repository batch processing, multiple embedding providers, and state-of-the-art deduplication algorithms.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version 0.5.1](https://img.shields.io/badge/version-0.5.1-0A7CFF.svg)](#)
[![Qdrant](https://img.shields.io/badge/Vector_DB-Qdrant-red.svg)](https://qdrant.tech/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Overview

This project automatically processes GitHub repositories containing documentation (markdown, PDFs, code, and text files) and creates optimized vector embeddings for **Retrieval-Augmented Generation (RAG)**, semantic search, and AI chat applications. It supports multiple embedding providers including cloud-based APIs and local models, featuring cutting-edge deduplication algorithms and intelligent PDF extraction.

### ✨ Key Features

- 🔄 **Multi-Repository Processing**: Process multiple repos sequentially with one command
- 🤖 **Multi-Provider Support**: Azure OpenAI, Mistral AI & Sentence Transformers
- 📑 **PDF Processing**: PyMuPDF (60x faster), PyPDFLoader, and Mistral OCR API
- ⚡ **5-15x Faster Processing**: Vectorized duplicate detection algorithms
- 🎯 **Smart Deduplication**: Two-stage content hash + semantic similarity
- 📊 **Real-time Progress**: Detailed processing reports with summary statistics
- 🛡️ **Production Ready**: Error handling, rate limiting, retry logic
- 🎛️ **Highly Configurable**: YAML configs with environment variable support
- 🧭 **CLI + TUI**: Script-compatible commands plus a fixed-pane Textual terminal UI
- 🩺 **Doctor & Benchmark**: Check collection health, compatibility, and retrieval quality
- 🧊 **TurboQuant Ready**: Optional Qdrant TurboQuant config for supported clusters
- 📚 **150+ File Types**: Process code, docs, configs, PDFs, and more
- 📄 **Individual File Processing**: Maintain document boundaries for better search (v0.3.3)
- 🔒 **Deterministic IDs**: Consistent vector IDs across runs prevent duplicates (v0.3.3)
- 💾 **Embedding Cache**: LRU cache reduces API calls by 20-30% (v0.3.2)
- 🎨 **Semantic Chunking**: Context-aware text splitting for better retrieval (v0.3.1)
- 📈 **Quality Scoring**: Rank chunks by information density and relevance (v0.3.2)
- 🔧 **Configurable Payloads**: Choose content fields for compatibility (v0.3.2)
- 🧩 **Hybrid Retrieval**: Optional dense + sparse BM25 search with Qdrant Query API (v0.5)
- 🗜️ **TurboQuant Ready**: Optional Qdrant 1.18+ TurboQuant collection setup (v0.5)

### 🎯 Perfect For

- **AI Chatbots** - Create knowledge bases from documentation
- **Semantic Search** - Enable intelligent document discovery  
- **RAG Applications** - Augment LLMs with domain-specific knowledge
- **Technical Documentation** - Process markdown, PDFs, and code with specialized embeddings
- **Content Processing** - Handles 150+ file types including PDFs, HTML, TXT, code files

## 🏃‍♂️ Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd github-qdrant-sync
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
github-qdrant-sync --version
```

### 2. Configure API Keys

Set up your API keys for your chosen embedding provider (see Configuration Guide below)

### 3. Configure

Copy the example config and add your API keys:

```bash
cp config.yaml.example config.yaml
# Edit config.yaml with your settings

# Set API keys in .env file for security
cp .env.example .env
# Edit .env with your API keys
```

Or use the interactive config wizard:

```bash
github-qdrant-sync wizard --output config.yaml
github-qdrant-sync validate-config config.yaml
```

If the target config file already exists, the wizard asks whether to overwrite it.
Answer `n` to choose a different filename such as `config.local.yaml`.

### 4. Run

```bash
github-qdrant-sync ingest config.yaml
```

## 🛠️ Installation & Dependencies

### Installation Manual

Use the editable install during local development so the `github-qdrant-sync`
command always points at your checkout:

```bash
git clone https://github.com/maholick/github-qdrant-sync.git
cd github-qdrant-sync
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
github-qdrant-sync --version
```

On Windows PowerShell:

```powershell
git clone https://github.com/maholick/github-qdrant-sync.git
cd github-qdrant-sync
py -3.9 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
github-qdrant-sync --version
```

For script-only usage without installing the console command, install the
requirements and call the original Python files directly:

```bash
pip install -r requirements.txt
python github_to_qdrant.py config.yaml
python rag_retrieval.py config.yaml --query "How do I configure authentication?"
```

### System Requirements

- **Python 3.10+**
- **4GB+ RAM** (for large repositories)
- **Internet connection** (for API calls)

### Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `langchain` - Document processing
- `qdrant-client>=1.18.0` - Vector database client with TurboQuant support
- `openai` - Azure OpenAI embeddings
- `numpy` - Vectorized operations
- `typer`, `rich`, and `textual` - Modern commands, output, and terminal UI
- `mistralai` - Mistral AI embeddings
- `sentence-transformers` - Local embedding models (optional)

### Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

## ⚙️ Configuration Guide

### 🔧 Basic Configuration

The project uses YAML configuration files with environment variable support. Start with `config.yaml.example`:

```yaml
# Embedding provider selection
embedding_provider: azure_openai  # or mistral_ai, sentence_transformers

github:
  repository_url: https://github.com/your-org/your-repo.git
  branch: main
  token: ${GITHUB_TOKEN}  # For private repos (from .env file)

qdrant:
  url: ${QDRANT_URL}  # e.g., https://your-cluster.qdrant.io
  api_key: ${QDRANT_API_KEY}
  collection_name: your-collection
  vector_size: 3072  # Must match embedding model
  upload_batch_size: 64
```

### 🤖 Embedding Providers

#### Azure OpenAI
```yaml
embedding_provider: azure_openai

azure_openai:
  api_key: ${AZURE_OPENAI_API_KEY}
  endpoint: ${AZURE_OPENAI_ENDPOINT}
  deployment_name: text-embedding-3-large
  api_version: "2024-02-01"

qdrant:
  vector_size: 3072  # for text-embedding-3-large
```

#### Mistral AI
```yaml
embedding_provider: mistral_ai

mistral_ai:
  api_key: ${MISTRAL_API_KEY}
  model: codestral-embed  # or mistral-embed
  output_dimension: 3072

answering:
  provider: mistral_ai
  model: mistral-large-2512
  temperature: 0.2
  max_context_chars: 12000

qdrant:
  vector_size: 3072
```

#### Sentence Transformers (Local)
```yaml
embedding_provider: sentence_transformers

sentence_transformers:
  model: intfloat/multilingual-e5-large
  vector_size: 1024

qdrant:
  vector_size: 1024
  vector_name: intfloat/multilingual-e5-large  # Optional: for MCP compatibility
```

**Sentence Transformers Benefits:**
- ✅ **No API Keys Required** - Runs locally
- ✅ **No Rate Limits** - Process any amount of data  
- ✅ **Privacy** - Data never leaves your machine
- ✅ **Cost Effective** - No per-token charges
- ✅ **Offline Capable** - Works without internet

### 🎛️ Performance Tuning

```yaml
processing:
  chunk_size: 1000              # Characters per chunk
  chunk_overlap: 200            # Overlap for context
  chunking_strategy: semantic   # 'semantic' or 'recursive' (v0.3.1+)
  embedding_batch_size: 50      # Optimized batch size
  batch_delay_seconds: 1        # Required for Azure OpenAI
  deduplication_enabled: true   # Enable smart deduplication
  similarity_threshold: 0.95    # Duplicate detection threshold
  file_mode: all_text          # Process all text files (or markdown_only)
  detect_text_content: true     # Also include readable text files with unknown extensions

# New in v0.3.2: Configurable payload fields
payload:
  metadata_structure: nested     # nested or flat
  content_fields:
    - content      # For n8n compatibility
    - page_content # For LangChain
    - document     # For MCP compatibility
    - text         # Alternative field name
  preview_length: 200           # Preview snippet length
  minimal_mode: false           # Reduce payload size by 50%
```

### 🧩 Qdrant 1.18+ Options

TurboQuant is opt-in and requires Qdrant server or Cloud 1.18+ plus
`qdrant-client>=1.18.0`. Start with `bits4` on a test collection before
enabling it for production data. Sparse BM25 vectors use Qdrant `Document`
inference; set `qdrant.cloud_inference: true` for Qdrant Cloud/server
inference or install `qdrant-client[fastembed]` for local inference.

```yaml
qdrant:
  vector_name: dense  # Required when sparse_vector.enabled is true
  cloud_inference: false

  quantization:
    enabled: false
    method: turbo
    bits: bits4       # bits4, bits2, bits1_5, or bits1
    always_ram: true
    apply_to_existing_collections: false
    search:
      ignore: false
      rescore: true
      oversampling: 2.0

  sparse_vector:
    enabled: false
    name: sparse
    model: qdrant/bm25
```

When `enabled` is true, new collections are created with TurboQuant. Existing
collections are left unchanged unless `apply_to_existing_collections: true` is
set. Search commands pass `qdrant.quantization.search` to Qdrant so you can
disable quantized search temporarily (`ignore: true`) or increase
`oversampling` for quality-sensitive queries.

TurboQuant settings are index-affecting. After changing `enabled`, `method`,
`bits`, `always_ram`, or the collection/vector settings, run:

```bash
github-qdrant-sync collections config.yaml --check-qdrant
github-qdrant-sync doctor config.yaml
github-qdrant-sync benchmark config.yaml --cases eval.yaml
```

In the TUI, use `/config`, `/set qdrant.quantization.enabled true`, and related
`qdrant.quantization.*` paths to stage and validate these settings before
saving.

Other recent Qdrant features are mainly operational wins for this project: 1.18
adds memory monitoring, per-collection metrics, strict-mode guardrails,
audit-log querying, and in-place named-vector schema changes; 1.17 improves
write-load search latency and observability; 1.16 improves filtered search and
disk-efficient storage. Hybrid retrieval and TurboQuant are the pieces wired
into this repo because they directly affect ingestion and RAG quality.

## 🚀 Usage Examples

### Basic Usage
```bash
# Process with default config
github-qdrant-sync ingest config.yaml

# Process different repository
github-qdrant-sync ingest config.yaml --repo-url https://github.com/other/repo.git

# Legacy script usage remains supported
python github_to_qdrant.py config.yaml
python rag_retrieval.py config.yaml --query "How do I configure authentication?"
```

### 🔄 Multi-Repository Processing (New!)

Process multiple repositories sequentially with a single command:

```bash
# Process multiple repositories from a list file
github-qdrant-sync ingest config.yaml --repo-list repositories.yaml
```

**Repository List File Format (`repositories.yaml`):**
```yaml
repositories:
  # Basic repository
  - url: https://github.com/langchain-ai/langchain.git
    collection_name: langchain-docs

  # Repository with specific branch
  - url: https://github.com/openai/openai-python.git
    branch: main
    collection_name: openai-python-docs

  # Private repository using SSH
  - url: git@github.com:myorg/private-repo.git
    branch: develop
    collection_name: private-docs

  # Multiple versions of the same project
  - url: https://github.com/facebook/react.git
    branch: main
    collection_name: react-latest

  - url: https://github.com/facebook/react.git
    branch: 18.x
    collection_name: react-v18
```

**Features:**
- ✅ Sequential processing with progress tracking
- ✅ Individual collection names per repository
- ✅ Continues processing if one repository fails
- ✅ Comprehensive summary report at the end
- ✅ All global settings from `config.yaml` apply
- ✅ The same `repositories.yaml` can be reused for multi-collection search and answers

**Example Output:**
```
============================================================
Processing repository 2/5
Repository: https://github.com/openai/openai-python.git
Branch: main
Collection: openai-python-docs
============================================================
[... processing output ...]

============================================================
MULTI-REPOSITORY PROCESSING SUMMARY
============================================================
Total repositories: 5
✅ Successful: 4
❌ Failed: 1

Details:
------------------------------------------------------------
✅ langchain → langchain-docs
   Files: 234, Chunks: 1,234
   Time: 45.2s
✅ openai-python → openai-python-docs
   Files: 89, Chunks: 567
   Time: 23.1s
❌ private-repo → Failed
   Error: Authentication error
✅ react → react-latest
   Files: 456, Chunks: 2,345
   Time: 89.3s
✅ react → react-v18
   Files: 423, Chunks: 2,123
   Time: 82.7s
------------------------------------------------------------

Totals:
   Files processed: 1,202
   Chunks created: 6,269
   Processing time: 240.3s (4m 0s)

Embedding Cache Performance:
   💾 Total hits: 1,543
   💾 Total misses: 4,726
   💾 Hit rate: 24.6%
   💾 Cache size: 500/500
============================================================
```

### 🔍 RAG Retrieval & Querying

Query your vector database using the included retrieval CLI:

**Basic Query:**
```bash
github-qdrant-sync query config.yaml --query "How do I configure authentication?"
```

By default, `query` and `ask` search `qdrant.collection_name` from the config.
If you ingested multiple repositories with `repositories.yaml`, pass that same
file to search all listed collections and merge the best matches:

```bash
github-qdrant-sync query config.yaml \
  --repo-list repositories.yaml \
  --query "How do I configure authentication?"

github-qdrant-sync ask config.yaml \
  --repo-list repositories.yaml \
  --question "How do I configure authentication?"
```

To narrow retrieval to one collection, use `--collection`. This works by itself
or together with `--repo-list`:

```bash
github-qdrant-sync query config.yaml \
  --collection openai-python-docs \
  --query "How do I configure authentication?"

github-qdrant-sync ask config.yaml \
  --repo-list repositories.yaml \
  --collection langchain-docs \
  --question "How do I configure callbacks?"
```

You can inspect the collections known to a config or repo list:

```bash
github-qdrant-sync collections config.yaml
github-qdrant-sync collections config.yaml --repo-list repositories.yaml
github-qdrant-sync collections config.yaml --repo-list repositories.yaml --check-qdrant
```

With `--check-qdrant`, the collection list verifies whether each collection is
usable with the active embedding config. Collections embedded with a different
provider, model, vector size, distance metric, or vector name are shown as
unusable instead of being queried accidentally.

By default, query results render with a compact Rich brand header, transient
search spinner, Vector Search summary, and ranked source table. Use
`--format text` for plain terminal output, `--format json` for scripts, or
`--no-banner` when you want the Rich table without the decorative header.

To generate an AI answer from retrieved context, use `ask`:

```bash
github-qdrant-sync ask config.yaml --question "How do I configure authentication?"
```

The `ask` command uses `answering.provider` and `answering.model`. Supported
answer providers are `mistral_ai` and `azure_openai`; `sentence_transformers`
remains retrieval-only because it does not provide a chat model.

Before `query` or `ask` runs, the CLI checks the selected collections against
the active embedding configuration. Compatible collections are searched,
incompatible collections are skipped with a clear reason, and the command fails
only when no selected collection is usable. If an older collection has matching
vector settings but lacks embedding metadata, retrieval continues with a
warning.

### 🖥️ Terminal UI

For repeated asking, searching, validation, and scoped collection changes, run
the CLI without arguments or use `interactive`. Both forms open the Textual
terminal UI on normal TTY terminals and use `config.yaml`, `config.yml`, or
`config.json` from the current directory when no config path is provided:

```bash
github-qdrant-sync
github-qdrant-sync interactive
github-qdrant-sync interactive config.yaml
github-qdrant-sync interactive config.yaml --repo-list repositories.yaml
```

The terminal UI keeps the app header, current output, sources table,
collections table, activity/progress area, and command input in fixed regions
instead of appending a new menu after every action. Long-running operations run
in the background, animate the activity pane with the current phase, stream
curated progress into the main output pane, then leave the final result on
screen.

On first start, if no `config.yaml`, `config.yml`, or `config.json` exists and
the terminal supports Textual, `github-qdrant-sync` opens the TUI setup wizard
instead of failing with a missing-config error. The wizard creates `config.yaml`
by default, uses environment-variable placeholders for secrets, validates the
generated config, and then leaves you in the TUI. Non-TTY runs and
`interactive --classic` keep the script-friendly error and can use:

```bash
github-qdrant-sync wizard --output config.yaml
```

Typing `/` in the command input opens a grouped command palette. The palette is
organized by workflow so the command list stays readable as the CLI grows:

- **Search & Ask**: `/ask`, `/search`, `/limit`, `/parent`
- **Collections**: `/collections`, `/scope`, `/repo-list`
- **Config**: `/config`, `/get`, `/set`, `/secret`, `/changes`, `/save-config`
- **Quality**: `/doctor`, `/benchmark`, `/improve`, `/validate`
- **Ingest**: `/ingest config`, `/ingest repo-url`, `/ingest repo-list`, `/pause`, `/resume`, `/stop`
- **Session**: `/clear`, `/help`, `/quit`

Use category help to drill into one group:

```text
/help search
/help collections
/help config
/help quality
/help ingest
/help session
```

Common commands inside the TUI:

```text
/ask How do I configure authentication?
/search auth middleware
/collections
/scope all
/scope openai-python-docs
/config config.local.yaml
/load-config configs/support.yaml
/repo-list repositories.yaml
/repo-list clear
/config
/get qdrant.collection_name
/set retrieval.top_k 5
/secret qdrant.api_key QDRANT_API_KEY
/changes
/save-config
/save-config --confirm
/save-config-as config.local.yaml --confirm
/discard-config-changes
/doctor
/benchmark
/benchmark eval.yaml
/improve
/improve eval.yaml --apply
/limit 5
/parent on
/validate
/ingest config
/ingest repo-url https://github.com/example/project.git
/ingest repo-list repositories.yaml
/pause
/resume
/stop
/help
/quit
```

Continue typing after `/`, for example `/ben` or `/sec`, to filter grouped
commands before submitting one.

When `--repo-list repositories.yaml` is provided, the initial scope is all
collections listed in that file. Use `/scope COLLECTION` to narrow to one
collection or `/scope all` to return to merged multi-collection retrieval. If no
repository list is loaded, `/scope all` simply resets to the config default
collection. Use `/config PATH` to switch config files without restarting the UI.
The collections pane is updated after search, ask, doctor, benchmark, and
`/collections` so unusable collections remain visible with their status and
reason.

During TUI ingestion, the main pane shows a live ingestion dashboard with the
current stage, repo/collection context, progress bars when totals are known, and
a recent event timeline. The activity pane remains a compact heartbeat with
elapsed time and the phase trail. Use `/pause` to pause at the next progress
checkpoint, `/resume` to continue, and `/stop` to request cooperative
cancellation without leaving the TUI. Long blocking backend calls such as
`git clone` finish their current call before pause or stop can take effect.

The TUI also supports safe config editing. `/config` shows the editable config
summary, `/set` stages non-secret values, and `/secret` writes env-var
placeholders like `${QDRANT_API_KEY}` instead of raw keys. Unsaved changes are
used immediately by TUI search/answer/validation through a temporary config
file, but the real config is only written after `/save-config --confirm`.
Saving validates first, blocks errors, and creates a timestamped `.bak` backup.
Index-affecting edits such as embedding provider, dimensions, vector size,
distance, vector name, and collection name are marked because they may require
reingestion.

### 🩺 Retrieval Quality Loop

The CLI includes a safe quality loop for checking index health and measuring
retrieval quality before changing ingestion settings:

```bash
github-qdrant-sync doctor config.yaml --repo-list repositories.yaml
github-qdrant-sync benchmark config.yaml --repo-list repositories.yaml
github-qdrant-sync benchmark config.yaml --cases eval.yaml --repo-list repositories.yaml
github-qdrant-sync improve config.yaml --repo-list repositories.yaml
github-qdrant-sync improve config.yaml --cases eval.yaml --repo-list repositories.yaml
```

`doctor` checks whether selected collections exist, contain points, match the
configured vector size/distance/vector name, expose expected payload fields, and
have configured payload indexes. It also checks sampled payload metadata for
embedding provider/model compatibility. Missing payload indexes can be created
idempotently:

```bash
github-qdrant-sync doctor config.yaml --apply-indexes --yes
```

`benchmark` uses explicit YAML cases so quality checks are reproducible. If
`--cases` is omitted, the CLI looks for `eval.yaml` and then `eval.yml` in the
current directory and next to the config file. This repository includes a
starter `eval.yaml`; edit the queries, expected sources, and keywords so they
match your indexed repositories:

```yaml
version: 1
thresholds:
  pass_rate: 0.8
  expected_source_top_k: 5
  min_top_score: 0.4
  min_keyword_coverage: 0.5

cases:
  - id: sso
    query: sso
    collection: veviad_app
    expected_sources:
      - app/Http/Controllers/Auth/Sso
      - resources/js/Pages/Admin/Integrations
    keywords:
      - sso
      - authentication
```

`improve` produces a conservative action report. It can safely tune retrieval
settings and configure payload indexes with explicit confirmation:

```bash
github-qdrant-sync improve config.yaml --apply --yes
```

The improvement flow never recreates collections, changes embedding dimensions,
or reingests automatically. Those actions are reported as follow-up decisions
because they can be destructive or expensive.

If `doctor`, `benchmark`, or `ask` reports an embedding mismatch, reingest that
repository with the active config or switch to the config that originally
created the collection. The CLI will not mix sources embedded with different
embedding models in one answer.

Vector scores are raw Qdrant similarity values, not percentages. Short acronym
queries such as `sso` may have useful results even when scores look lower than
human-friendly percentages, so benchmark pass/fail combines source rank, keyword
coverage, score floor, and aggregate pass rate.

The classic prompt-loop menu is still available for simple terminals or muscle
memory:

```bash
github-qdrant-sync interactive --classic
github-qdrant-sync interactive --no-tui
```

Non-TTY output and `TERM=dumb` automatically use the classic mode. One-shot
commands such as `query`, `ask`, `ingest`, `collections`, and `validate-config`
remain the best interface for scripts and automation.

`query` is the retrieval/evidence command: it shows matched snippets. `ask` is
the answer command: it retrieves snippets first, then uses the configured chat
provider to generate a grounded answer with sources.

The legacy script interfaces remain available for automation that already calls
the original files directly:

```bash
python github_to_qdrant.py config.yaml --repo-list repositories.yaml
python rag_retrieval.py config.yaml --query "How do I configure authentication?"
```

**With Filters:**
```yaml
# In config.yaml:
retrieval:
  mode: dense
  top_k: 10
  fetch_k: 40  # Retrieves more candidates for grouping
  max_chunks_per_file: 3  # Caps results per file
  filters:
    repository: my-repo-name  # Optional filtering
```

**Hybrid Dense + Sparse Retrieval (Qdrant 1.18+):**
```yaml
qdrant:
  vector_name: dense
  cloud_inference: true  # Or install qdrant-client[fastembed] for local sparse inference
  sparse_vector:
    enabled: true
    name: sparse
    model: qdrant/bm25

retrieval:
  mode: hybrid
  fusion: rrf
  top_k: 10
  fetch_k: 40
```

Hybrid mode stores a named dense vector and a sparse BM25 vector per chunk, then uses Qdrant Query API prefetches with reciprocal-rank fusion. Existing unnamed-vector collections should be recreated or migrated intentionally before enabling hybrid mode.

**JSON Output (for programmatic use):**
```bash
github-qdrant-sync query config.yaml --query "setup guide" --format json
```

**Verbose Logging:**
```bash
github-qdrant-sync query config.yaml --query "api reference" --verbose
```

**Features:**
- ✅ **Smart Grouping**: Caps results per file for better context diversity
- ✅ **Multi-Collection Retrieval**: Search all collections from `repositories.yaml`
- ✅ **Collection Selection**: Override the config default with `--collection`
- ✅ **Parent Window Expansion**: Retrieve surrounding context around matched chunks
- ✅ **Multiple Output Formats**: Rich terminal UI, plain text, or machine-readable JSON
- ✅ **Polished Terminal UX**: Width-safe panels, grouped help, and optional banner suppression
- ✅ **Flexible Filtering**: Filter by repository, file type, or any metadata field
- ✅ **Marker Exclusion**: Internal incremental-sync markers are hidden by default
- ✅ **Hybrid Search**: Optional dense+sparse retrieval for better keyword recall
- ✅ **Timing Information**: Debug mode shows embedding, search, and grouping times
- ✅ **Robust Error Handling**: Clear error messages and suggestions

**Parent Window Context:**
```bash
# Retrieve surrounding chunks for expanded context
python rag_retrieval.py config.yaml --query "installation" --with-parent-window
```

**Example Output:**
```
INFO: Querying collection: my-docs
INFO: Query: How do I configure authentication?
INFO: Returning 10 results

#1 score=0.8234 file=docs/authentication.md
preview: Configure authentication using OAuth2 or API keys...

#2 score=0.7891 file=guides/setup.md
preview: Authentication setup requires the following steps...

#3 score=0.7654 file=api/reference.md
preview: API authentication methods include bearer tokens...
```

**Configuration Options:**
```yaml
retrieval:
  mode: dense                 # dense or hybrid
  fusion: rrf                 # hybrid only: rrf or dbsf
  top_k: 10                  # Final number of results to return
  fetch_k: 40                # Candidates to fetch before grouping (should be 3-4x top_k)
  max_chunks_per_file: 3     # Maximum results per file
  parent_window: 2           # Chunks before/after for context expansion
  filters:                   # Optional metadata filters
    repository: my-repo
    source_type: markdown
```

**CLI Arguments:**
- `--query`: Your search query (required)
- `--limit`: Override config's top_k
- `--collection`: Search one Qdrant collection instead of the config default
- `--repo-list`: Search all collections listed in `repositories.yaml`
- `--format`: Output format (rich, text, or json)
- `--no-banner`: Hide the decorative brand header
- `--with-parent-window`: Enable context expansion
- `--verbose` / `-v`: Enable debug logging
- `--quiet` / `-q`: Suppress info messages

**Use Cases:**
- 🤖 **AI Chatbots**: Retrieve relevant context for LLM responses
- 🔍 **Semantic Search**: Find similar content across documentation
- 📚 **Documentation Q&A**: Answer questions from your knowledge base
- 🧪 **Testing Retrieval Quality**: Evaluate embedding and chunking strategies

---

### Multiple Configurations
```bash
# Different repository configurations
python github_to_qdrant.py config_technical.yaml

# Multi-language documentation
python github_to_qdrant.py config_multilang.yaml

# Large documentation projects
python github_to_qdrant.py config_enterprise.yaml
```

### 📑 PDF Processing

The pipeline includes state-of-the-art PDF processing with three modes:

**1. Local Mode (Offline, Fast)**
```yaml
pdf_processing:
  enabled: true
  mode: local  # Uses PyMuPDF (60x faster) with PyPDFLoader fallback
```

**2. Cloud Mode (Mistral OCR API)**
```yaml
pdf_processing:
  enabled: true
  mode: cloud  # Best quality, handles scanned PDFs, $0.001/page
  cloud:
    max_pages_per_doc: 100  # Cost control
```

**3. Hybrid Mode (Smart Selection)**
```yaml
pdf_processing:
  enabled: true
  mode: hybrid  # Local first, cloud for complex/scanned PDFs
  hybrid:
    force_cloud_patterns:  # Always use OCR for these
      - "*scan*.pdf"
      - "*ocr*.pdf"
```

### Advanced Examples

**Process specific branch:**
```yaml
github:
  repository_url: https://github.com/your-org/your-repo.git
  branch: develop
```

**Switch to Mistral AI:**
```yaml
embedding_provider: mistral_ai

mistral_ai:
  api_key: ${MISTRAL_API_KEY}
  model: codestral-embed
  output_dimension: 3072

answering:
  provider: mistral_ai
  model: mistral-large-2512
  temperature: 0.2
  max_context_chars: 12000
```

**Use Azure OpenAI for answers:**
```yaml
answering:
  provider: azure_openai
  model: ${AZURE_OPENAI_CHAT_DEPLOYMENT}
  temperature: 0.2
  max_context_chars: 12000
```

**Use Local Models:**
```yaml
embedding_provider: sentence_transformers

sentence_transformers:
  model: intfloat/multilingual-e5-large
  vector_size: 1024
```

## 📊 Embedding Models Comparison

| Provider | Model | Dimensions | Best For | Context |
|----------|-------|------------|----------|---------|
| **Azure OpenAI** | text-embedding-ada-002 | 1536 | General text | 2,048 tokens |
| **Azure OpenAI** | text-embedding-3-small | 1536 | Efficient processing | 8,191 tokens |
| **Azure OpenAI** | text-embedding-3-large | 3072 | **Best quality** | 8,191 tokens |
| **Mistral AI** | mistral-embed | 1024 | General text | 8,000 tokens |
| **Mistral AI** | codestral-embed | 3072 | **Technical docs** | 8,000 tokens |
| **Sentence Transformers** | all-MiniLM-L6-v2 | 384 | **Lightweight/Fast** | 256 tokens |
| **Sentence Transformers** | multilingual-e5-large | 1024 | **Multilingual** | 512 tokens |

### 💡 Recommendations

- **Technical Documentation**: Use `codestral-embed` (Mistral AI)
- **General Documentation**: Use `text-embedding-3-large` (Azure OpenAI) 
- **Cost-Effective**: Use `text-embedding-3-small` (Azure OpenAI)
- **Code Repositories**: Use `codestral-embed` (Mistral AI)
- **Privacy/Offline**: Use `multilingual-e5-large` (Sentence Transformers)
- **Fast/Lightweight**: Use `all-MiniLM-L6-v2` (Sentence Transformers)
- **Multilingual Content**: Use `multilingual-e5-large` (Sentence Transformers)
- **No API Costs**: Use any Sentence Transformers model

## 🚄 Performance & Optimization

### ⚡ Deduplication Performance

This project features **cutting-edge deduplication** that's **5-15x faster** than traditional methods:

#### **Traditional Approach (Slow)**
- O(n²) complexity: Each chunk compared to ALL previous chunks
- Individual similarity calculations
- No progress reporting
- **Hours for large repositories**

#### **Our Optimized Approach (Fast)**
- ✅ **Content hash pre-filtering**: Instant exact duplicate removal
- ✅ **Vectorized similarity**: Batch NumPy operations  
- ✅ **Progress reporting**: Real-time feedback
- ✅ **Memory optimization**: Batched processing
- ✅ **Smart thresholding**: Configurable similarity detection

### 📈 Processing Speed

| Repository Size | Traditional | Optimized | Speedup |
|----------------|-------------|-----------|---------|
| Small (100 files) | 5 minutes | 1 minute | **5x** |
| Medium (500 files) | 45 minutes | 5 minutes | **9x** |
| Large (1000+ files) | 3+ hours | 15 minutes | **12x+** |

### ⚙️ Rate Limiting

**Azure OpenAI:**
- Batch size: 50 chunks
- Delay: 1 second between batches
- Auto-retry with exponential backoff

**Mistral AI:**
- Batch size: 50 chunks  
- Delay: 1 second between batches
- Shorter retry delays

**Sentence Transformers:**
- No rate limits (local processing)
- Batch size: 50 chunks (for memory management)
- Processing speed depends on hardware (CPU/GPU)

### 🧠 Memory Usage

- **Batched processing**: Prevents memory overflow
- **Streaming embeddings**: Process chunks incrementally
- **Automatic cleanup**: Temporary files removed

**Local Models (Sentence Transformers):**
- Model loaded once, reused for all chunks
- Additional VRAM usage for GPU acceleration
- Faster processing with dedicated GPU

## 📦 Data Structure & Metadata

### 🏗️ Payload Structure (Updated v0.3.2)

This project uses a **configurable payload structure** for maximum compatibility with n8n, LangChain, MCP, and other frameworks. Each document chunk is stored with the following structure:

### ⚡ Qdrant Payload Indexes (Optional)

For large collections, enabling payload indexes improves performance for filtered queries (e.g. “only PDFs”, “only a specific repository”, “only a specific file path”).

- **Nested metadata (`payload.metadata_structure: nested`)**: index fields use paths like `metadata.repository`.
- **Flat metadata (`payload.metadata_structure: flat`)**: index fields use paths like `repository`.

Configure in `qdrant.payload_indexes` in `config.yaml` (see `config.yaml.example`).

### 🔁 Incremental Sync (track_file_changes) and Shared Collections

When `processing.track_file_changes: true`, the pipeline computes a **SHA-256** `file_hash` per file and uses deterministic identifiers to safely support either:
- **One repo per collection**, or
- **Multiple repos/branches sharing a collection**.

Key metadata fields:
- `repo_id`: SHA-256 of `repo_url@branch`
- `file_id`: SHA-256 of `repo_id:file_path`
- `file_upload_id`: SHA-256 of `file_id:file_hash`

This prevents accidental cross-repo deletes when different repos contain the same `file_path`, and it enables auto-repair of partial uploads by writing a per-file marker only after upload succeeds.

```json
{
  // Configurable content fields (choose which to include via config)
  "content": "Full document text content here...",      // n8n
  "page_content": "Full document text content here...",  // LangChain
  "document": "Full document text content here...",      // MCP
  "text": "Full document text content here...",          // Alternative

  // Flattened metadata (v0.3.2) - no nesting for better performance
  "doc_id": "repo-name_file.md_123",
  "chunk_id": 123,
  "source": "path/to/file.md",
  "source_type": "markdown",  // pdf, code, config, etc.
  "repository": "your-repo-name",
  "branch": "main",
  "preview": "First 200 characters of content...",
  "chunk_size": 850,
  "token_count": 213,          // NEW in v0.3.2
  "quality_score": 0.87,       // NEW in v0.3.2 (0-1 scale)
  "timestamp": 1705147200,     // Unix timestamp (smaller)
  "content_hash": "abc123de",
  "extraction_method": "default",

  // PDF-specific fields (when applicable)
  "page_number": 5,
  "total_pages": 42
}
```

### 📋 Field Descriptions

**Content Fields (configurable via `payload.content_fields`):**
- **`content`** - Full text for n8n compatibility
- **`page_content`** - Full text for LangChain compatibility
- **`document`** - Full text for MCP server compatibility
- **`text`** - Alternative field name some systems use

**Metadata Fields (flattened in v0.3.2):**
- **`doc_id`** - Unique document identifier (repo_file_chunk)
- **`chunk_id`** - Sequential chunk number
- **`source`** - Path to source file
- **`source_type`** - File type (pdf, markdown, code, config, etc.)
- **`preview`** - Configurable preview snippet (default 200 chars)
- **`token_count`** - Token count for LLM context management (v0.3.2)
- **`quality_score`** - Content quality score 0-1 (v0.3.2)
- **`timestamp`** - Unix timestamp (more compact than ISO)
- **`content_hash`** - MD5 hash for duplicate detection
- **`extraction_method`** - How content was extracted

### ✅ Benefits

- **🔧 Standard Compatible** - Works seamlessly with LangChain, n8n, and most Qdrant clients
- **🤖 MCP Server Ready** - Includes `document` field for Qdrant MCP server compatibility
- **💾 Efficient Storage** - Clean separation between standard and compatibility fields
- **🔍 Easy Filtering** - Structured metadata enables precise search and filtering
- **📊 Debug Friendly** - Preview field allows quick content inspection without full retrieval

## 📁 Project Structure

```
github-qdrant-sync/
├── github_to_qdrant.py      # 🌟 Main processing script
├── github_qdrant_cli.py     # 🖥️ Typer/Rich command-line interface
├── github_qdrant_tui.py     # 🧭 Textual fixed-pane terminal UI
├── rag_retrieval.py         # 🔍 Retrieval and semantic search helpers
├── pdf_processor.py         # 📑 Advanced PDF processing module
├── config.yaml.example      # 📝 Configuration template with docs
├── config.yaml              # 🔧 Your configuration (gitignored)
├── eval.yaml                # 🧪 Starter benchmark cases
├── repositories.yaml.example # 📋 Multi-repo list template
├── repositories.yaml        # 📋 Your repository list (gitignored)
├── .env.example             # 🔐 Environment variables template
├── .env                     # 🔑 Your API keys (gitignored)
├── pyproject.toml           # 📦 Package metadata and CLI entrypoint
├── requirements.txt         # 📦 Python dependencies
├── .gitignore              # 🚫 Git exclusions
├── README.md               # 📖 This documentation
├── CLAUDE.md               # 🤖 AI assistant context (gitignored)
├── venv/                   # 🐍 Virtual environment (gitignored)
└── markdown/               # 📄 Generated markdown output (gitignored)
    ├── repo-name/
    │   ├── __combined_markdown.md
    │   ├── folder1.md
    │   └── folder2.md
    └── ...
```

### 📄 Configuration Files

- **`config.yaml.example`** - Template with inline documentation and examples
- **`config.yaml`** - Your custom configuration (gitignored)
- **`repositories.yaml.example`** - Template for multi-repository processing
- **`repositories.yaml`** - Your repository list (gitignored)
- **`.env.example`** - Template for environment variables
- **`.env`** - Your API keys and sensitive data (gitignored)

#### Why YAML?

✅ **Cleaner syntax** - More readable than JSON
✅ **Comments support** - Inline documentation
✅ **Environment variables** - Secure API key management via `${VAR_NAME}` syntax
✅ **Multi-line strings** - Better for long text values
✅ **Default values** - Support for `${VAR:-default}` pattern

## 🔧 Troubleshooting

### Common Issues

#### 🚫 "ModuleNotFoundError: No module named 'mistralai'"
```bash
# Reinstall requirements to ensure all dependencies are available
pip install -r requirements.txt
```

#### 🔑 "Authentication Error"  
- Check API keys in config file
- Verify endpoint URLs
- Ensure API keys have proper permissions

#### ⏱️ "Rate Limit Exceeded"
- Increase `batch_delay_seconds` in config
- Reduce `embedding_batch_size`
- Check API quotas

#### 🧠 "Out of Memory"
- Reduce `chunk_size` in config
- Increase `batch_delay_seconds`
- Process smaller repositories

#### 🔗 "Connection Failed"
- Check internet connection
- Verify Qdrant URL and API key
- Test with smaller batch size

### 🐛 Debug Mode

Enable detailed logging:
```yaml
logging:
  level: DEBUG
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### 📞 Getting Help

1. **Check the logs** - Enable DEBUG logging
2. **Verify configuration** - Use `config.yaml.example` 
3. **Test connections** - Run with minimal config
4. **Check dependencies** - Reinstall requirements
5. **API quotas** - Verify account limits

## 🔒 Security & Best Practices

### 🔐 API Key Management

❌ **Never commit API keys to Git!**

✅ **Safe practices:**
```bash
# Use .env file (recommended)
cp .env.example .env
# Edit .env with your API keys:
# AZURE_OPENAI_API_KEY=your-key
# MISTRAL_API_KEY=your-key
# QDRANT_API_KEY=your-key
# GITHUB_TOKEN=your-token

# Then use environment variables in config.yaml:
# api_key: ${AZURE_OPENAI_API_KEY}

# Or use separate config files (gitignored)
cp config.yaml.example config.local.yaml
# Edit config.local.yaml with real keys
python github_to_qdrant.py config.local.yaml
```

### 🛡️ Production Deployment

- Use environment variables for secrets
- Enable rate limiting
- Monitor API usage and costs
- Set up proper logging
- Use dedicated service accounts
- Prefer `/secret` in the TUI for API keys; it writes env-var placeholders
  instead of raw values.
- CLI/TUI errors, config summaries, and JSON metadata redact obvious secret
  fields and tokenized GitHub URLs, but retrieved repository content is still
  shown as indexed.
- Unsaved TUI config edits are written to temporary files with owner-only
  permissions and removed after backend commands finish.

## 🎯 Example Workflows

### Workflow 1: Technical Documentation Site
```bash
# 1. Configure for technical content in config.yaml
embedding_provider: mistral_ai
mistral_ai:
  model: codestral-embed
  output_dimension: 3072

# 2. Process repository
python github_to_qdrant.py config.yaml
```

### Workflow 2: Multi-Language Documentation
```bash
# Process different language versions
python github_to_qdrant.py config_english.yaml  # English docs
python github_to_qdrant.py config_german.yaml   # German docs
python github_to_qdrant.py config_french.yaml   # French docs

# Or use multi-repo processing with single config
python github_to_qdrant.py config.yaml --repo-list multilang_repos.yaml
```

### Workflow 3: Continuous Integration
```bash
#!/bin/bash
# Update vector database when docs change
git pull origin main
python github_to_qdrant.py config.yaml
echo "Vector database updated successfully"
```

## 📊 Performance Metrics

### Real-World Results

**Large Documentation Repository (1,200+ files):**
- **Processing time**: 12 minutes (vs 3+ hours traditional)
- **Duplicates removed**: 1,847 chunks (23% of total)
- **Final chunks**: 6,234 unique vectors
- **Accuracy**: 99.8% (manual validation)

**Performance breakdown:**
- Repository cloning: 30 seconds
- Markdown processing: 2 minutes  
- Embedding generation: 6 minutes
- Deduplication: 3 minutes
- Upload to Qdrant: 1 minute

## 📝 Changelog

### v0.5.1 (2026-05-30) - Live TUI Ingestion Progress
- ✨ **Live Ingestion Dashboard**: TUI ingestion now streams curated progress into the main pane with current stage, repo/collection context, timeline, and progress bars when totals are known
- ✨ **Pause/Resume/Stop Controls**: Added `/pause`, `/resume`, and `/stop` so long ingestions can be controlled without quitting the TUI
- 🔍 **All-Text Detection**: `file_mode: all_text` now includes readable text files even when their extension/name is not listed, while still respecting exclude patterns
- 🔧 **Cooperative Cancellation**: Ingestion progress callbacks can stop work cleanly with exit code `130`; signal handling remains main-thread safe for TUI workers
- ⚡ **Faster TUI Startup**: CLI/TUI startup now lazy-loads heavy ingestion, retrieval, and quality modules so the terminal UI opens before LangChain/Qdrant/PDF imports are needed
- 📚 **TUI Docs**: README now documents live ingestion progress and cooperative pause/stop behavior
- 🧪 **Tests**: Added coverage for ingestion progress callbacks, cancellation, signal handling, TUI progress rendering, and controls

### v0.5.0 (2026-05-27) - Qdrant 1.18, CLI & TUI Modernization
- ✨ **TurboQuant Support**: Optional Qdrant 1.18+ TurboQuant collection config
- ✨ **Hybrid Retrieval**: Optional dense + sparse BM25 ingestion and RRF querying
- ✨ **Typer/Rich CLI**: Installable `github-qdrant-sync` command with `ingest`, `query`, `ask`, `collections`, `doctor`, `benchmark`, `improve`, `interactive`, `validate-config`, and `wizard`
- ✨ **Textual TUI**: Fixed-pane interactive terminal UI with slash commands, activity progress, source/collection panes, and first-run setup when no config exists
- ✨ **AI Answers**: `ask` command builds grounded answers from retrieved chunks using configured Mistral AI or Azure OpenAI chat models
- ✨ **Multi-Collection Retrieval**: Query, ask, benchmark, and interactive mode can search all collections listed in `repositories.yaml`
- ✨ **Collection Compatibility Checks**: Skips missing or embedding-incompatible collections with explicit reasons instead of mixing incompatible sources
- ✨ **Safe TUI Config Editing**: Inspect, stage, validate, save, save-as, and discard whitelisted config changes with secret redaction
- ✨ **Quality Loop**: Added `doctor`, `benchmark`, and `improve` workflows plus starter `eval.yaml`
- 📚 **Install Manual**: README now documents editable install, Windows setup, script-only usage, CLI, TUI, and release workflows
- 🔧 **Shared Qdrant Config**: Ingestion and retrieval now share connection parsing
- 🐛 **Metadata Config Fix**: `payload.metadata_structure` is canonical, with legacy fallback
- 🐛 **Exclude Pattern Fixes**: Glob-aware excludes for paths like `*.pyc`
- 🧪 **Unit Tests**: Added pytest coverage for config, payloads, Qdrant setup, retrieval, CLI routing, TUI commands, quality checks, and setup flows

### v0.4.1 (2025-12-19) - Robustness & Reliability Improvements
- 🐛 **Retrieval Fixes**: Fixed `fetch_k` defaults, collection existence checks, empty result guidance, and debug logging
- ✨ **CLI Polish**: Added JSON output, verbose/quiet logging, timing details, and clearer help text
- 🔧 **Pipeline Reliability**: Added orphaned marker cleanup, config validation, and payload index mismatch warnings
- 📚 **Documentation**: Expanded retrieval docs and updated example configs

### v0.4.0 (2025-12-17) - Retrieval, Indexing & Robust Incremental Sync
- ✨ **Retrieval CLI**: Added `rag_retrieval.py` for querying Qdrant collections
- ✨ **Grouping by File**: Optionally cap/interleave results per file for better context diversity
- ✨ **Token-aware Chunking**: Optional `token_recursive` chunking strategy using `tiktoken`
- ✨ **Qdrant Payload Indexes**: Optional payload index management for faster filtered queries
- 🔁 **Robust Incremental Sync**: Dedup-safe skipping via per-file markers written after successful upload
- 🛡️ **Shared Collection Safety**: Multi-repo/branch safe scoping with `repo_id` / `file_id` / `file_upload_id`
- 🐛 **Track File Changes Fixes**: Safer deletes + legacy fallback handling for older collections

### v0.3.4 (2025-11-12) - Metadata & Embedding Config Standardization
- ✨ **Standardized embedding config**: Use `model` consistently across providers
- 🐛 **Metadata field fixes**: Improved repository/name mapping and attribution fields
- 📌 **Embedding tracking**: Persist `embedding_provider` and `embedding_model` in metadata

### v0.3.3 (2025-09-16) - Individual File Processing & Deduplication
- ✨ **Individual File Processing**: Process files separately for better context preservation
- ✨ **Deterministic ID Generation**: Fixed duplicate vector creation on repeated runs
- 🐛 **Fixed Duplicate Vectors**: Resolved issue where each run added exactly 1 duplicate vector
- 🐛 **Fixed combine_documents Setting**: Properly respects `combine_documents: false` configuration
- 🚀 **Clean Interrupt Handling**: Improved Ctrl+C handling without verbose tracebacks
- 🔧 **Type Safety**: Fixed type annotations for better IDE support
- 📊 **Better Search Quality**: Individual file processing maintains document boundaries

### v0.3.2 (2025-09-14) - Performance & Compatibility
- ✨ **Embedding Cache**: LRU cache reduces API calls by 20-30%
- ✨ **Configurable Payloads**: Choose content fields for n8n/LangChain/MCP compatibility
- ✨ **Quality Scoring**: Rank chunks by information density (0-1 scale)
- ✨ **Token Counting**: Track tokens for LLM context management
- 🚀 **Flattened Metadata**: Improved filtering performance
- 🚀 **CI/CD Optimization**: 10x faster (30s vs 3min)
- 🐛 Fixed formatting and import issues

### v0.3.1 (2025-09-14) - Semantic Intelligence
- ✨ **Semantic Chunking**: Context-aware text splitting for 30% better retrieval
- ✨ **Mistral Vision API**: Extract text/content from images in PDFs
- 🚀 Improved PDF processing with image extraction
- 🐛 Fixed PDF processing edge cases

### v0.3.0 (2025-09-12) - Multi-Repository Support
- ✨ **Multi-Repository Processing**: Process multiple repos with one command
- ✨ **Repository Lists**: YAML configuration for batch processing
- 📊 Comprehensive summary reports with statistics
- 🚀 Shared resources across repositories

### v0.2.0 (2025-09-10) - Enhanced Processing
- ✨ **PDF Processing**: Three modes (local, cloud, hybrid)
- ✨ **Mistral OCR API**: Cloud-based PDF extraction
- ✨ **150+ File Types**: Extended file type support
- 🚀 **5-15x Faster**: Vectorized deduplication

### v0.1.0 (2025-09-08) - Initial Release
- 🎉 Initial public release
- ✨ Azure OpenAI and Mistral AI support
- ✨ Basic markdown and text processing
- ✨ Qdrant integration

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Qdrant** - High-performance vector database
- **Azure OpenAI** - Advanced embedding models  
- **Mistral AI** - Specialized code embeddings
- **LangChain** - Document processing framework

---

**Made with ❤️ for the AI community**

*Transform your documentation into intelligent, searchable knowledge bases.*
