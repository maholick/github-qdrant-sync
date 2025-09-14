#!/usr/bin/env python3
"""
Gradio Web UI for GitHub to Qdrant Vector Processing Pipeline
"""

import gradio as gr
import os
import yaml
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Any, Optional

# Import the UI components (to be created)
from ui_processor import UIProcessor
from ui_config_manager import ConfigManager

# Create necessary directories
Path("configs").mkdir(exist_ok=True)
Path("repo_lists").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)


class GitHubQdrantUI:
    """Main UI Application for GitHub to Qdrant processing"""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.processor = UIProcessor()
        self.current_config = None
        self.current_repo_list = None

    def create_interface(self):
        """Create the main Gradio interface"""

        with gr.Blocks(title="GitHub → Qdrant Processor", theme=gr.themes.Soft()) as app:
            gr.Markdown(
                """
                # 🚀 GitHub to Qdrant Vector Processing Pipeline
                Transform GitHub repositories into searchable vector databases for AI applications.
                """
            )

            with gr.Tabs():
                # Tab 1: Quick Process (Single Repository)
                with gr.Tab("⚡ Quick Process"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.Markdown("### Single Repository Processing")

                            # Config selection
                            config_dropdown = gr.Dropdown(
                                label="Configuration",
                                choices=self.config_manager.list_configs(),
                                value="config.yaml" if "config.yaml" in self.config_manager.list_configs() else None,
                                interactive=True
                            )

                            # Repository settings
                            repo_url = gr.Textbox(
                                label="Repository URL",
                                placeholder="https://github.com/user/repo.git",
                                lines=1
                            )

                            with gr.Row():
                                branch = gr.Textbox(
                                    label="Branch (optional)",
                                    placeholder="main",
                                    lines=1
                                )
                                collection_name = gr.Textbox(
                                    label="Collection Name",
                                    placeholder="my-collection",
                                    lines=1
                                )

                            # Process button
                            process_btn = gr.Button(
                                "🔄 Process Repository",
                                variant="primary",
                                size="lg"
                            )

                        with gr.Column(scale=2):
                            gr.Markdown("### Quick Settings")

                            # Quick config display
                            config_info = gr.JSON(
                                label="Current Configuration",
                                visible=True
                            )

                            # Test connections
                            with gr.Row():
                                test_qdrant_btn = gr.Button("🔌 Test Qdrant", size="sm")
                                test_embed_btn = gr.Button("🤖 Test Embeddings", size="sm")

                            connection_status = gr.Textbox(
                                label="Connection Status",
                                lines=2,
                                interactive=False
                            )

                    # Progress and output
                    progress_bar = gr.Progress()

                    with gr.Row():
                        # Real-time log output
                        log_output = gr.Textbox(
                            label="Processing Log",
                            lines=20,
                            max_lines=30,
                            interactive=False,
                            autoscroll=True
                        )

                    # Results
                    process_results = gr.JSON(
                        label="Processing Results",
                        visible=False
                    )

                # Tab 2: Batch Process (Multiple Repositories)
                with gr.Tab("📦 Batch Process"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.Markdown("### Multi-Repository Processing")

                            # Repository list selection
                            repo_list_dropdown = gr.Dropdown(
                                label="Repository List",
                                choices=self.config_manager.list_repo_lists(),
                                interactive=True
                            )

                            # Repository list editor
                            repo_list_editor = gr.Code(
                                label="Edit Repository List",
                                language="yaml",
                                lines=15,
                                value=self.config_manager.load_repo_list_template()
                            )

                            with gr.Row():
                                save_list_btn = gr.Button("💾 Save List", size="sm")
                                validate_list_btn = gr.Button("✓ Validate", size="sm")

                            # Batch process button
                            batch_process_btn = gr.Button(
                                "🚀 Start Batch Processing",
                                variant="primary",
                                size="lg"
                            )

                        with gr.Column(scale=2):
                            gr.Markdown("### Batch Status")

                            # Progress tracking
                            batch_progress = gr.Dataframe(
                                headers=["Repository", "Status", "Files", "Chunks", "Time (s)"],
                                label="Processing Status",
                                interactive=False
                            )

                            # Overall progress
                            overall_progress = gr.Progress()

                            # Summary statistics
                            batch_summary = gr.Textbox(
                                label="Batch Summary",
                                lines=5,
                                interactive=False
                            )

                    # Batch log output
                    batch_log = gr.Textbox(
                        label="Batch Processing Log",
                        lines=20,
                        max_lines=30,
                        interactive=False,
                        autoscroll=True
                    )

                # Tab 3: Configuration Manager
                with gr.Tab("⚙️ Configuration"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.Markdown("### Configuration Editor")

                            # Config selection and management
                            with gr.Row():
                                config_selector = gr.Dropdown(
                                    label="Select Configuration",
                                    choices=self.config_manager.list_configs(),
                                    interactive=True
                                )
                                config_name_input = gr.Textbox(
                                    label="Config Name",
                                    placeholder="my-config.yaml",
                                    lines=1
                                )

                            # Config editor
                            config_editor = gr.Code(
                                label="Configuration (YAML)",
                                language="yaml",
                                lines=25,
                                value=self.config_manager.load_config_template()
                            )

                            with gr.Row():
                                save_config_btn = gr.Button("💾 Save Config", variant="primary")
                                load_template_btn = gr.Button("📋 Load Template")
                                validate_config_btn = gr.Button("✓ Validate")

                        with gr.Column(scale=2):
                            gr.Markdown("### Configuration Options")

                            # Provider selection
                            provider_radio = gr.Radio(
                                label="Embedding Provider",
                                choices=["azure_openai", "mistral_ai", "sentence_transformers"],
                                value="azure_openai"
                            )

                            # API Keys (masked inputs)
                            with gr.Group():
                                gr.Markdown("#### API Keys")
                                azure_key = gr.Textbox(
                                    label="Azure OpenAI Key",
                                    type="password",
                                    placeholder="Enter key or use ${AZURE_OPENAI_API_KEY}"
                                )
                                mistral_key = gr.Textbox(
                                    label="Mistral AI Key",
                                    type="password",
                                    placeholder="Enter key or use ${MISTRAL_API_KEY}"
                                )
                                qdrant_key = gr.Textbox(
                                    label="Qdrant API Key",
                                    type="password",
                                    placeholder="Enter key or use ${QDRANT_API_KEY}"
                                )
                                github_token = gr.Textbox(
                                    label="GitHub Token (optional)",
                                    type="password",
                                    placeholder="For private repos"
                                )

                            # Test buttons
                            gr.Markdown("#### Test Connections")
                            with gr.Row():
                                test_all_btn = gr.Button("🔍 Test All Connections")

                            test_results = gr.Textbox(
                                label="Test Results",
                                lines=5,
                                interactive=False
                            )

                # Tab 4: Repository Lists
                with gr.Tab("📋 Repository Lists"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Manage Repository Lists")

                            # List management
                            list_selector = gr.Dropdown(
                                label="Select List",
                                choices=self.config_manager.list_repo_lists(),
                                interactive=True
                            )

                            # List editor
                            list_editor = gr.Code(
                                label="Repository List (YAML)",
                                language="yaml",
                                lines=20
                            )

                            # Add repository form
                            gr.Markdown("#### Add Repository")
                            with gr.Group():
                                add_url = gr.Textbox(label="Repository URL", lines=1)
                                add_branch = gr.Textbox(label="Branch (optional)", lines=1)
                                add_collection = gr.Textbox(label="Collection Name", lines=1)
                                add_repo_btn = gr.Button("➕ Add to List")

                            with gr.Row():
                                save_list_main_btn = gr.Button("💾 Save List", variant="primary")
                                validate_urls_btn = gr.Button("🔍 Validate URLs")
                                export_csv_btn = gr.Button("📤 Export CSV")

                # Tab 5: Logs & History
                with gr.Tab("📊 Logs & History"):
                    gr.Markdown("### Processing History")

                    # Session selector
                    session_dropdown = gr.Dropdown(
                        label="Select Session",
                        choices=self.config_manager.list_sessions(),
                        interactive=True
                    )

                    # Log viewer
                    log_viewer = gr.Textbox(
                        label="Session Log",
                        lines=25,
                        max_lines=50,
                        interactive=False
                    )

                    # History table
                    history_table = gr.Dataframe(
                        headers=["Timestamp", "Repository", "Status", "Files", "Chunks", "Duration"],
                        label="Processing History",
                        interactive=False
                    )

                    with gr.Row():
                        refresh_logs_btn = gr.Button("🔄 Refresh")
                        export_logs_btn = gr.Button("📤 Export Logs")
                        clear_logs_btn = gr.Button("🗑️ Clear Old Logs")

            # Footer
            gr.Markdown(
                """
                ---
                Made with ❤️ for the AI community | [GitHub](https://github.com/maholick/github-qdrant-sync)
                """
            )

            # Event handlers will be connected here
            self._connect_events(
                app, config_dropdown, repo_url, branch, collection_name,
                process_btn, log_output, process_results, config_info,
                test_qdrant_btn, test_embed_btn, connection_status,
                repo_list_dropdown, repo_list_editor, batch_process_btn,
                batch_progress, batch_log, batch_summary,
                config_selector, config_editor, save_config_btn,
                provider_radio, test_all_btn, test_results
            )

        return app

    def _connect_events(self, app, *components):
        """Connect event handlers to UI components"""
        # This will be implemented to connect all the buttons and interactions
        pass


def main():
    """Launch the Gradio UI"""
    ui = GitHubQdrantUI()
    app = ui.create_interface()

    # Enable queue for long-running operations
    app.queue()

    # Launch the app
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,  # Set to True for public URL
        show_error=True,
        inbrowser=True
    )


if __name__ == "__main__":
    main()