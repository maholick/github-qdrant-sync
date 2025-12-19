# Troubleshooting Guide

This guide helps resolve common issues with the GitHub to Qdrant Vector Processing Pipeline.

---

## Table of Contents

1. [Incremental Sync Issues](#incremental-sync-issues)
2. [RAG Retrieval Problems](#rag-retrieval-problems)
3. [Payload Index Issues](#payload-index-issues)
4. [Multi-Repo Shared Collections](#multi-repo-shared-collections)
5. [Migration from Pre-v0.4.0](#migration-from-pre-v040)
6. [Performance Issues](#performance-issues)
7. [Connection & Authentication](#connection--authentication)

---

## Incremental Sync Issues

### Problem: Files are being reprocessed on every run despite no changes

**Symptoms:**
- All files show as "Processing" instead of "Skipping unchanged file"
- Processing takes full time even when repository hasn't changed

**Solutions:**

1. **Verify config setting:**
   ```yaml
   processing:
     track_file_changes: true  # Must be enabled
   ```

2. **Check for marker upsert failures:**
   - Look for warning: `⚠️  Warning: Could not save incremental sync markers`
   - This indicates markers weren't written after last run
   - Check Qdrant connection and permissions

3. **Verify chunking configuration hasn't changed:**
   - Changes to `chunk_size`, `chunk_overlap`, `chunking_strategy`, etc. invalidate markers
   - This is intentional - chunks need regeneration

4. **Legacy collections (pre-v0.4.0):**
   - Old collections may not have `repo_id`, `file_id`, `file_upload_id` metadata
   - Solution: Enable payload indexes for these fields and re-ingest once

---

### Problem: Some files stuck as "incomplete" and reprocess every time

**Symptoms:**
- Specific files always reprocess
- Log shows: "Processing: <filename>" instead of "Skipping unchanged file"

**Causes & Solutions:**

1. **Deduplication reduced chunk count below expected:**
   - This should be handled by file markers (v0.4.0+)
   - If using pre-v0.4.0: Upgrade to v0.4.0 for marker-based tracking

2. **Partial upload (interrupted run):**
   - Marker wasn't written because upload didn't complete
   - Solution: Let current run complete fully

3. **Chunking signature mismatch:**
   - Check if config changed between runs
   - Use consistent chunking settings

---

### Problem: Orphaned markers accumulating in collection

**Symptoms:**
- Collection size growing despite deleting files from repository
- Markers for deleted files remain

**Solution:**

Enable orphaned marker cleanup in config:
```yaml
processing:
  track_file_changes: true
  cleanup_orphaned_markers: true  # Add this
```

This will remove markers for files that no longer exist in the repository.

---

## RAG Retrieval Problems

### Problem: rag_retrieval.py returns no results

**Symptoms:**
```
No results found for query: <your query>
```

**Solutions:**

1. **Verify collection exists and has data:**
   ```bash
   python rag_retrieval.py config.yaml --query "test" --verbose
   ```
   Check for error: `Collection '<name>' does not exist`

2. **Check filters in config:**
   ```yaml
   retrieval:
     filters:
       repository: exact-repo-name  # Must match exactly
   ```
   - Try removing filters temporarily
   - Use `--verbose` to see filter details

3. **Verify embeddings match:**
   - Query uses same embedding provider as ingestion
   - Check `embedding_provider` in config matches what was used for ingestion

4. **Collection is empty:**
   - Run ingestion first: `python github_to_qdrant.py config.yaml`

---

### Problem: Results are all from the same file

**Symptoms:**
- All top results come from single file despite `max_chunks_per_file: 3`

**Cause:** `fetch_k` is too small

**Solution:**

Ensure `fetch_k` is significantly larger than `top_k`:
```yaml
retrieval:
  top_k: 10
  fetch_k: 40  # Should be 3-4x top_k minimum
```

v0.4.1+ calculates this automatically if not specified.

---

### Problem: Parent window expansion returns empty

**Symptoms:**
- With `--with-parent-window`, context is empty
- Warning: `Cannot expand parent window: missing 'parent_id' metadata`

**Causes & Solutions:**

1. **Collection ingested before v0.4.0:**
   - Parent metadata wasn't added in older versions
   - Solution: Re-ingest with v0.4.0+

2. **Metadata structure mismatch:**
   - Verify `payload.metadata_structure` in config matches ingestion setting
   - Default is `nested`, but some collections use `flat`

---

### Problem: JSON output missing or malformed

**Symptoms:**
- `--format json` doesn't produce valid JSON
- Error parsing output

**Solutions:**

1. **Suppress log messages:**
   ```bash
   python rag_retrieval.py config.yaml --query "test" --format json --quiet
   ```

2. **Redirect logs to stderr (if needed):**
   - Use `--quiet` flag
   - JSON always goes to stdout, logs to stderr with proper logging

---

## Payload Index Issues

### Problem: Index creation fails with "already exists" error

**Symptoms:**
```
Failed to create payload index for 'metadata.repository': already exists
```

**This is normal!** The system is idempotent and skips existing indexes.

**If you need to change index types:**
1. You must recreate the collection (Qdrant doesn't support changing types)
2. Set `recreate_collection: true` in config
3. Re-run ingestion

---

### Problem: Index type mismatch warning

**Symptoms:**
```
WARNING: Index type mismatch for 'metadata.repository': existing=integer, requested=keyword
```

**Meaning:** Config specifies different type than what exists in collection

**Solutions:**

1. **To keep existing index:** Ignore warning, existing index remains

2. **To change index type:**
   - Must recreate collection
   - Update config: `recreate_collection: true`
   - Re-run ingestion

---

### Problem: Filtered queries are slow despite indexes

**Symptoms:**
- Queries with filters take long time
- Indexes exist but queries don't use them

**Solutions:**

1. **Verify indexes are created:**
   ```yaml
   qdrant:
     payload_indexes:
       enabled: true
       apply_to_existing_collections: true  # Important!
   ```

2. **Check field paths match metadata structure:**
   - Nested: Index on `metadata.repository`
   - Flat: Index on `repository`

3. **Ensure indexed fields are in payload:**
   - Fields must actually exist in ingested data
   - Check a sample point in Qdrant to verify

---

## Multi-Repo Shared Collections

### Problem: Deletes affecting wrong repository

**Symptoms:**
- Deleting file from Repo A removes chunks from Repo B with same path
- Cross-repository interference

**Cause:** Using legacy collection (pre-v0.4.0)

**Solution:**

1. **Upgrade to v0.4.0+** for `repo_id/file_id` scoping

2. **One-time migration:**
   - Back up collection first
   - Re-ingest all repos with v0.4.0+
   - New ingestion adds proper scoping metadata

3. **Avoid enabling `legacy_cleanup_delete_by_file_path`** in shared collections

---

### Problem: Cannot distinguish repos in queries

**Symptoms:**
- Results mix content from multiple repos
- No way to filter by specific repository

**Solution:**

Add repository filter in retrieval config:
```yaml
retrieval:
  filters:
    repository: specific-repo-name
```

Or use `repo_id` filter (more reliable):
```yaml
retrieval:
  filters:
    repo_id: <sha256-hash-of-repo-url-at-branch>
```

---

## Migration from Pre-v0.4.0

### Problem: Old collections not working with new features

**Symptoms:**
- Incremental sync not working
- File markers not found
- Parent window expansion fails

**Solution: Gradual Migration**

1. **Keep using old collection (limited functionality):**
   - Set `track_file_changes: false`
   - Set `cleanup_orphaned_markers: false`
   - Parent window won't work

2. **Full migration (recommended):**
   ```yaml
   qdrant:
     recreate_collection: true  # One-time
     payload_indexes:
       enabled: true
   processing:
     track_file_changes: true
   ```
   - Re-ingest all content
   - New features fully functional

---

## Performance Issues

### Problem: Ingestion is very slow

**Solutions:**

1. **Enable deduplication (if disabled):**
   ```yaml
   processing:
     deduplication_enabled: true
   ```

2. **Use incremental sync:**
   ```yaml
   processing:
     track_file_changes: true
   ```

3. **Adjust batch sizes:**
   ```yaml
   processing:
     batch_size: 100  # Increase for faster processing
     embedding_batch_size: 50  # Adjust based on API limits
   ```

4. **Enable embedding cache:**
   - Caching is automatic for sentence-transformers
   - For cloud providers, cache is in-memory only (per session)

5. **Check PDF processing mode:**
   ```yaml
   pdf_processing:
     mode: local  # Faster than cloud for most cases
   ```

---

### Problem: Query/retrieval is slow

**Solutions:**

1. **Enable payload indexes:**
   ```yaml
   qdrant:
     payload_indexes:
       enabled: true
   ```

2. **Reduce fetch_k if very large:**
   ```yaml
   retrieval:
     fetch_k: 40  # Don't set too high
   ```

3. **Use faster embedding model:**
   - sentence-transformers (local) fastest
   - Mistral AI (cloud) faster than Azure OpenAI

4. **Check Qdrant connection method:**
   ```yaml
   qdrant:
     connection_method: auto  # Usually optimal
   ```

---

## Connection & Authentication

### Problem: Cannot connect to Qdrant

**Symptoms:**
```
Error: Connection refused
Could not connect to Qdrant
```

**Solutions:**

1. **Verify Qdrant is running:**
   ```bash
   # If using Docker:
   docker ps | grep qdrant

   # Check Qdrant health:
   curl http://localhost:6333/health
   ```

2. **Check connection config:**
   ```yaml
   qdrant:
     url: ${QDRANT_URL}  # Or explicit URL
     api_key: ${QDRANT_API_KEY}  # If using auth
   ```

3. **Environment variables set:**
   ```bash
   # In .env file:
   QDRANT_URL=http://localhost:6333
   QDRANT_API_KEY=your-key  # If needed
   ```

4. **Firewall/network issues:**
   - Check if port 6333 (HTTP) or 6334 (gRPC) is accessible
   - Try `connection_method: url` for explicit HTTP

---

### Problem: Authentication fails

**Symptoms:**
```
401 Unauthorized
403 Forbidden
```

**Solutions:**

1. **Check API key:**
   ```bash
   echo $QDRANT_API_KEY  # Should output your key
   ```

2. **Verify .env file loaded:**
   - File must be named exactly `.env`
   - Must be in same directory as script
   - Or set variables in shell

3. **Cloud Qdrant:**
   ```yaml
   qdrant:
     url: https://your-cluster.qdrant.tech
     api_key: ${QDRANT_API_KEY}
     connection_method: url  # Important for cloud
   ```

---

## Getting More Help

### Enable Debug Logging

**For ingestion:**
```yaml
logging:
  level: DEBUG
```

**For retrieval:**
```bash
python rag_retrieval.py config.yaml --query "test" --verbose
```

### Check Versions

```bash
pip show qdrant-client langchain openai mistralai
python --version
```

### Report Issues

If problems persist:
1. Include full error message
2. Include relevant config (redact API keys!)
3. Include Python/package versions
4. Report at: https://github.com/maholick/github-qdrant-sync/issues

---

## Quick Diagnostic Checklist

Before reporting issues, verify:

- [ ] Qdrant is running and accessible
- [ ] Config file syntax is valid YAML
- [ ] Environment variables are set (`.env` file or shell)
- [ ] Collection exists (for retrieval)
- [ ] Using same embedding provider for ingest and retrieval
- [ ] No conflicting config options (e.g., `recreate_collection: true` in production)
- [ ] Logs show actual error messages (enable DEBUG if needed)

---

*Last updated: v0.4.1*
