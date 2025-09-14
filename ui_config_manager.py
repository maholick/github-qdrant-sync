#!/usr/bin/env python3
"""
Configuration manager for the Gradio UI
Handles loading, saving, and validating configurations and repository lists
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import shutil


class ConfigManager:
    """Manage configurations and repository lists for the UI"""

    def __init__(self):
        self.config_dir = Path("configs")
        self.repo_lists_dir = Path("repo_lists")
        self.logs_dir = Path("logs")

        # Create directories if they don't exist
        for directory in [self.config_dir, self.repo_lists_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def list_configs(self) -> List[str]:
        """List all available configuration files"""
        configs = []

        # Add example config if exists
        if Path("config.yaml.example").exists():
            configs.append("config.yaml.example")

        # Add user's config.yaml if exists
        if Path("config.yaml").exists():
            configs.append("config.yaml")

        # Add saved configs
        for config_file in self.config_dir.glob("*.yaml"):
            if config_file.name != "templates":
                configs.append(f"configs/{config_file.name}")

        for config_file in self.config_dir.glob("*.yml"):
            configs.append(f"configs/{config_file.name}")

        return sorted(configs)

    def list_repo_lists(self) -> List[str]:
        """List all available repository list files"""
        lists = []

        # Add example if exists
        if Path("repositories.yaml.example").exists():
            lists.append("repositories.yaml.example")

        # Add user's repositories.yaml if exists
        if Path("repositories.yaml").exists():
            lists.append("repositories.yaml")

        # Add saved lists
        for list_file in self.repo_lists_dir.glob("*.yaml"):
            lists.append(f"repo_lists/{list_file.name}")

        for list_file in self.repo_lists_dir.glob("*.yml"):
            lists.append(f"repo_lists/{list_file.name}")

        return sorted(lists)

    def load_config(self, config_name: str) -> str:
        """Load a configuration file as string"""
        try:
            config_path = Path(config_name)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return f.read()
            return ""
        except Exception as e:
            return f"# Error loading config: {str(e)}"

    def save_config(self, config_name: str, content: str) -> Dict[str, Any]:
        """Save a configuration file"""
        try:
            # Validate YAML syntax first
            yaml.safe_load(content)

            # Determine save path
            if not config_name.endswith(('.yaml', '.yml')):
                config_name += '.yaml'

            # Save to configs directory unless it's the main config.yaml
            if config_name == "config.yaml":
                config_path = Path(config_name)
            else:
                config_path = self.config_dir / config_name

            # Write file
            with open(config_path, 'w') as f:
                f.write(content)

            return {
                "status": "success",
                "message": f"Configuration saved to {config_path}",
                "path": str(config_path)
            }

        except yaml.YAMLError as e:
            return {
                "status": "error",
                "message": f"Invalid YAML syntax: {str(e)}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to save configuration: {str(e)}"
            }

    def validate_config(self, content: str) -> Dict[str, Any]:
        """Validate a configuration"""
        try:
            config = yaml.safe_load(content)

            errors = []
            warnings = []

            # Check required sections
            required_sections = ["github", "qdrant", "processing"]
            for section in required_sections:
                if section not in config:
                    errors.append(f"Missing required section: {section}")

            # Check embedding provider
            if "embedding_provider" not in config:
                errors.append("Missing embedding_provider specification")
            else:
                provider = config["embedding_provider"]
                if provider not in ["azure_openai", "mistral_ai", "sentence_transformers"]:
                    errors.append(f"Unknown embedding provider: {provider}")

                # Check provider-specific config
                if provider in config:
                    if provider == "azure_openai":
                        if "api_key" not in config[provider]:
                            warnings.append("Azure OpenAI API key not configured")
                    elif provider == "mistral_ai":
                        if "api_key" not in config[provider]:
                            warnings.append("Mistral AI API key not configured")

            # Check Qdrant config
            if "qdrant" in config:
                if "vector_size" not in config["qdrant"]:
                    errors.append("Missing qdrant.vector_size")
                if "collection_name" not in config["qdrant"]:
                    warnings.append("No default collection_name specified")

            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings
            }

        except yaml.YAMLError as e:
            return {
                "valid": False,
                "errors": [f"Invalid YAML syntax: {str(e)}"],
                "warnings": []
            }

    def load_repo_list(self, list_name: str) -> str:
        """Load a repository list file as string"""
        try:
            list_path = Path(list_name)
            if list_path.exists():
                with open(list_path, 'r') as f:
                    return f.read()
            return ""
        except Exception as e:
            return f"# Error loading repository list: {str(e)}"

    def save_repo_list(self, list_name: str, content: str) -> Dict[str, Any]:
        """Save a repository list file"""
        try:
            # Validate YAML syntax
            data = yaml.safe_load(content)

            # Validate structure
            if not isinstance(data, dict) or "repositories" not in data:
                return {
                    "status": "error",
                    "message": "Repository list must contain a 'repositories' key"
                }

            # Validate each repository
            for i, repo in enumerate(data["repositories"]):
                if "url" not in repo:
                    return {
                        "status": "error",
                        "message": f"Repository {i+1}: Missing 'url' field"
                    }
                if "collection_name" not in repo:
                    return {
                        "status": "error",
                        "message": f"Repository {i+1}: Missing 'collection_name' field"
                    }

            # Determine save path
            if not list_name.endswith(('.yaml', '.yml')):
                list_name += '.yaml'

            # Save to repo_lists directory unless it's the main repositories.yaml
            if list_name == "repositories.yaml":
                list_path = Path(list_name)
            else:
                list_path = self.repo_lists_dir / list_name

            # Write file
            with open(list_path, 'w') as f:
                f.write(content)

            return {
                "status": "success",
                "message": f"Repository list saved to {list_path}",
                "path": str(list_path),
                "count": len(data["repositories"])
            }

        except yaml.YAMLError as e:
            return {
                "status": "error",
                "message": f"Invalid YAML syntax: {str(e)}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to save repository list: {str(e)}"
            }

    def add_repository_to_list(
        self,
        list_content: str,
        url: str,
        branch: Optional[str] = None,
        collection_name: str = None
    ) -> str:
        """Add a repository to an existing list"""
        try:
            # Parse existing content or create new
            if list_content.strip():
                data = yaml.safe_load(list_content)
            else:
                data = {"repositories": []}

            if "repositories" not in data:
                data["repositories"] = []

            # Create new repository entry
            new_repo = {
                "url": url,
                "collection_name": collection_name
            }
            if branch:
                new_repo["branch"] = branch

            # Add to list
            data["repositories"].append(new_repo)

            # Convert back to YAML
            return yaml.dump(data, default_flow_style=False, sort_keys=False)

        except Exception as e:
            return list_content  # Return unchanged on error

    def load_config_template(self) -> str:
        """Load the default configuration template"""
        # Use the original example file
        if Path("config.yaml.example").exists():
            with open("config.yaml.example", 'r') as f:
                return f.read()

        # Return a minimal template
        return """# GitHub to Qdrant Configuration
embedding_provider: azure_openai

github:
  repository_url: https://github.com/user/repo.git
  branch: main
  clone_depth: 1
  cleanup_after_processing: true

azure_openai:
  api_key: ${AZURE_OPENAI_API_KEY}
  endpoint: ${AZURE_OPENAI_ENDPOINT}
  deployment_name: text-embedding-3-large
  api_version: "2024-02-01"

qdrant:
  url: ${QDRANT_URL}
  api_key: ${QDRANT_API_KEY}
  collection_name: my-collection
  vector_size: 3072
  distance: Cosine

processing:
  chunk_size: 1000
  chunk_overlap: 200
  deduplication_enabled: true
  similarity_threshold: 0.95
"""

    def load_repo_list_template(self) -> str:
        """Load the repository list template"""
        # Use the original example file
        if Path("repositories.yaml.example").exists():
            with open("repositories.yaml.example", 'r') as f:
                return f.read()

        # Return a minimal template
        return """# Repository List
repositories:
  - url: https://github.com/user/repo1.git
    branch: main
    collection_name: collection-1

  - url: https://github.com/user/repo2.git
    collection_name: collection-2
"""

    def list_sessions(self) -> List[str]:
        """List all log sessions"""
        sessions = []
        for log_file in self.logs_dir.glob("*.log"):
            sessions.append(log_file.stem)
        return sorted(sessions, reverse=True)

    def save_session_log(self, log_content: str) -> str:
        """Save a session log"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"session_{timestamp}.log"
        with open(log_file, 'w') as f:
            f.write(log_content)
        return str(log_file)

    def load_session_log(self, session_name: str) -> str:
        """Load a session log"""
        log_file = self.logs_dir / f"{session_name}.log"
        if log_file.exists():
            with open(log_file, 'r') as f:
                return f.read()
        return "Log file not found"

    def export_config_with_env(self, config_content: str) -> str:
        """Export configuration with environment variables resolved"""
        try:
            # This would resolve environment variables if needed
            # For now, just return the content
            return config_content
        except Exception as e:
            return f"# Error exporting config: {str(e)}"