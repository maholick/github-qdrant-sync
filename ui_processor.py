#!/usr/bin/env python3
"""
UI Processor wrapper for GitHub to Qdrant processing with Gradio integration
"""

import sys
import io
import threading
import queue
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import traceback
import time

from github_to_qdrant import (
    GitHubToQdrantProcessor,
    ProcessingResult,
    RepositoryConfig,
    load_repository_list,
    process_repository_list
)


class LogCapture:
    """Capture stdout/stderr and forward to UI"""

    def __init__(self, callback: Callable[[str], None], original_stream):
        self.callback = callback
        self.original = original_stream
        self.buffer = []

    def write(self, text):
        """Capture and forward text"""
        if text and text != '\n':
            self.callback(text)
            self.buffer.append(text)
        # Also write to original stream for debugging
        self.original.write(text)

    def flush(self):
        """Flush the stream"""
        self.original.flush()

    def get_full_log(self):
        """Get complete captured log"""
        return ''.join(self.buffer)


class UIProcessor:
    """Wrapper for GitHubToQdrantProcessor with UI callbacks"""

    def __init__(self):
        self.processor = None
        self.log_queue = queue.Queue()
        self.is_processing = False
        self.should_stop = False

    def process_single_repository(
        self,
        config_path: str,
        repo_url: str,
        branch: Optional[str] = None,
        collection_name: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        log_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process a single repository with UI feedback

        Args:
            config_path: Path to configuration file
            repo_url: Repository URL
            branch: Optional branch override
            collection_name: Optional collection name override
            progress_callback: Gradio progress callback
            log_callback: Function to update log display

        Returns:
            Processing results dictionary
        """
        self.is_processing = True
        self.should_stop = False

        # Initialize result
        result = {
            "status": "failed",
            "repo_url": repo_url,
            "collection_name": collection_name,
            "files_processed": 0,
            "chunks_created": 0,
            "processing_time": 0,
            "error": None,
            "log": ""
        }

        # Create log capture
        log_capture = LogCapture(
            lambda text: self._handle_log(text, log_callback),
            sys.stdout
        )
        error_capture = LogCapture(
            lambda text: self._handle_log(f"ERROR: {text}", log_callback),
            sys.stderr
        )

        try:
            # Update progress
            if progress_callback:
                progress_callback(0.1, desc="Initializing processor...")

            # Initialize processor
            self.processor = GitHubToQdrantProcessor(config_path)

            if progress_callback:
                progress_callback(0.2, desc="Starting repository processing...")

            # Redirect output
            with redirect_stdout(log_capture), redirect_stderr(error_capture):
                start_time = time.time()

                # Process repository
                if branch or collection_name:
                    # Use override method
                    processing_result = self.processor.process_repository_with_override(
                        repo_url=repo_url,
                        branch=branch,
                        collection_name=collection_name
                    )

                    result["status"] = processing_result.status
                    result["files_processed"] = processing_result.files_processed
                    result["chunks_created"] = processing_result.chunks_created
                    result["error"] = processing_result.error
                else:
                    # Use standard method
                    self.processor.process_repository(repo_url)
                    result["status"] = "success"

                result["processing_time"] = time.time() - start_time

            if progress_callback:
                progress_callback(1.0, desc="Processing complete!")

            # Get full log
            result["log"] = log_capture.get_full_log()

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            result["log"] = log_capture.get_full_log() if 'log_capture' in locals() else ""

            # Log the error
            if log_callback:
                log_callback(f"\n❌ Error: {str(e)}\n")
                log_callback(traceback.format_exc())

        finally:
            self.is_processing = False

        return result

    def process_repository_list(
        self,
        config_path: str,
        repo_list_path: str,
        progress_callback: Optional[Callable] = None,
        log_callback: Optional[Callable] = None,
        status_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process multiple repositories from a list

        Args:
            config_path: Path to configuration file
            repo_list_path: Path to repository list YAML
            progress_callback: Gradio progress callback
            log_callback: Function to update log display
            status_callback: Function to update status table

        Returns:
            Batch processing results
        """
        self.is_processing = True
        self.should_stop = False

        results = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "repositories": [],
            "total_time": 0,
            "log": ""
        }

        # Create log capture
        log_capture = LogCapture(
            lambda text: self._handle_log(text, log_callback),
            sys.stdout
        )

        try:
            start_time = time.time()

            # Load repository list
            if progress_callback:
                progress_callback(0.05, desc="Loading repository list...")

            repositories = load_repository_list(repo_list_path)
            results["total"] = len(repositories)

            # Initialize processor
            if progress_callback:
                progress_callback(0.1, desc="Initializing processor...")

            self.processor = GitHubToQdrantProcessor(config_path)

            # Process each repository
            with redirect_stdout(log_capture):
                for i, repo_config in enumerate(repositories):
                    if self.should_stop:
                        break

                    # Update progress
                    progress = (i + 0.5) / len(repositories)
                    if progress_callback:
                        progress_callback(
                            progress,
                            desc=f"Processing {i+1}/{len(repositories)}: {repo_config.url}"
                        )

                    # Update status table
                    if status_callback:
                        status_callback(repo_config.url, "processing", 0, 0, 0)

                    try:
                        # Process repository
                        repo_start = time.time()
                        result = self.processor.process_repository_with_override(
                            repo_url=repo_config.url,
                            branch=repo_config.branch,
                            collection_name=repo_config.collection_name
                        )

                        repo_time = time.time() - repo_start

                        # Update status
                        if result.status == "success":
                            results["successful"] += 1
                            if status_callback:
                                status_callback(
                                    repo_config.url,
                                    "✅ Complete",
                                    result.files_processed,
                                    result.chunks_created,
                                    round(repo_time, 1)
                                )
                        else:
                            results["failed"] += 1
                            if status_callback:
                                status_callback(
                                    repo_config.url,
                                    f"❌ Failed: {result.error[:50]}",
                                    0, 0, round(repo_time, 1)
                                )

                        results["repositories"].append({
                            "url": repo_config.url,
                            "status": result.status,
                            "files": result.files_processed,
                            "chunks": result.chunks_created,
                            "time": repo_time,
                            "error": result.error
                        })

                    except Exception as e:
                        results["failed"] += 1
                        if status_callback:
                            status_callback(
                                repo_config.url,
                                f"❌ Error: {str(e)[:50]}",
                                0, 0, 0
                            )
                        if log_callback:
                            log_callback(f"\n❌ Error processing {repo_config.url}: {e}\n")

            results["total_time"] = time.time() - start_time
            results["log"] = log_capture.get_full_log()

            if progress_callback:
                progress_callback(1.0, desc="Batch processing complete!")

        except Exception as e:
            results["error"] = str(e)
            if log_callback:
                log_callback(f"\n❌ Batch processing error: {e}\n")

        finally:
            self.is_processing = False

        return results

    def test_connection(
        self,
        config_path: str,
        connection_type: str = "all"
    ) -> Dict[str, Any]:
        """
        Test connections to various services

        Args:
            config_path: Path to configuration file
            connection_type: Type of connection to test ("qdrant", "embeddings", "all")

        Returns:
            Test results dictionary
        """
        results = {
            "qdrant": {"status": "not_tested", "message": ""},
            "embeddings": {"status": "not_tested", "message": ""},
            "github": {"status": "not_tested", "message": ""}
        }

        try:
            # Initialize processor to test connections
            processor = GitHubToQdrantProcessor(config_path)

            if connection_type in ["qdrant", "all"]:
                try:
                    # Test Qdrant connection
                    collections = processor.qdrant_client.get_collections()
                    results["qdrant"] = {
                        "status": "success",
                        "message": f"Connected! Found {len(collections.collections)} collections"
                    }
                except Exception as e:
                    results["qdrant"] = {
                        "status": "failed",
                        "message": f"Connection failed: {str(e)[:100]}"
                    }

            if connection_type in ["embeddings", "all"]:
                try:
                    # Test embedding provider
                    test_embedding = processor.embeddings.embed_query("test")
                    provider = processor.config.get("embedding_provider", "azure_openai")
                    results["embeddings"] = {
                        "status": "success",
                        "message": f"{provider} connected! Dimension: {len(test_embedding)}"
                    }
                except Exception as e:
                    results["embeddings"] = {
                        "status": "failed",
                        "message": f"Connection failed: {str(e)[:100]}"
                    }

        except Exception as e:
            # Failed to initialize processor
            return {
                "error": f"Failed to load configuration: {str(e)}",
                "qdrant": {"status": "failed", "message": "Could not initialize"},
                "embeddings": {"status": "failed", "message": "Could not initialize"}
            }

        return results

    def stop_processing(self):
        """Stop current processing"""
        self.should_stop = True

    def _handle_log(self, text: str, callback: Optional[Callable]):
        """Handle log output"""
        if callback:
            callback(text)
        self.log_queue.put(text)

    def get_logs(self) -> str:
        """Get accumulated logs"""
        logs = []
        while not self.log_queue.empty():
            try:
                logs.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        return ''.join(logs)