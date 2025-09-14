#!/usr/bin/env python3
"""
Simplified Gradio Web UI for GitHub to Qdrant Vector Processing Pipeline
"""

import gradio as gr
import os
import yaml
from datetime import datetime
from pathlib import Path
import threading
import time

from ui_processor import UIProcessor
from ui_config_manager import ConfigManager


class GitHubQdrantUI:
    """Simplified UI Application for GitHub to Qdrant processing"""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.processor = UIProcessor()
        self.current_log = []

    def process_single_repo(self, config_file, repo_url, branch, collection_name, progress=gr.Progress()):
        """Process a single repository"""
        if not config_file:
            return "❌ Please select a configuration file", {}

        if not repo_url:
            return "❌ Please enter a repository URL", {}

        self.current_log = []

        def log_callback(text):
            self.current_log.append(text)
            return '\n'.join(self.current_log[-100:])  # Keep last 100 lines

        try:
            # Process repository
            result = self.processor.process_single_repository(
                config_path=config_file,
                repo_url=repo_url,
                branch=branch if branch else None,
                collection_name=collection_name if collection_name else None,
                progress_callback=progress,
                log_callback=log_callback
            )

            # Create summary
            summary = f"""
✅ Processing Complete!

Repository: {repo_url}
Status: {result['status']}
Files Processed: {result['files_processed']}
Chunks Created: {result['chunks_created']}
Processing Time: {result['processing_time']:.1f} seconds
"""
            if result['error']:
                summary += f"\n⚠️ Error: {result['error']}"

            return '\n'.join(self.current_log), result

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            self.current_log.append(error_msg)
            return '\n'.join(self.current_log), {"error": str(e)}

    def process_batch(self, config_file, repo_list_file, progress=gr.Progress()):
        """Process multiple repositories"""
        if not config_file:
            return "❌ Please select a configuration file", [], ""

        if not repo_list_file:
            return "❌ Please select a repository list file", [], ""

        self.current_log = []
        batch_status = []

        def log_callback(text):
            self.current_log.append(text)

        def status_callback(repo_url, status, files, chunks, time):
            # Update batch status
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            batch_status.append([repo_name, status, files, chunks, time])
            return batch_status

        try:
            # Process repository list
            result = self.processor.process_repository_list(
                config_path=config_file,
                repo_list_path=repo_list_file,
                progress_callback=progress,
                log_callback=log_callback,
                status_callback=status_callback
            )

            # Create summary
            summary = f"""
Batch Processing Complete!
Total: {result['total']} repositories
Successful: {result['successful']}
Failed: {result['failed']}
Total Time: {result['total_time']:.1f} seconds
"""

            return '\n'.join(self.current_log), batch_status, summary

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            self.current_log.append(error_msg)
            return '\n'.join(self.current_log), [], error_msg

    def test_connections(self, config_file):
        """Test connections to services"""
        if not config_file:
            return "❌ Please select a configuration file"

        try:
            results = self.processor.test_connection(config_file, "all")

            output = "🔌 Connection Test Results:\n\n"

            # Qdrant
            if results['qdrant']['status'] == 'success':
                output += f"✅ Qdrant: {results['qdrant']['message']}\n"
            else:
                output += f"❌ Qdrant: {results['qdrant']['message']}\n"

            # Embeddings
            if results['embeddings']['status'] == 'success':
                output += f"✅ Embeddings: {results['embeddings']['message']}\n"
            else:
                output += f"❌ Embeddings: {results['embeddings']['message']}\n"

            return output

        except Exception as e:
            return f"❌ Error testing connections: {str(e)}"

    def save_config(self, config_name, config_content):
        """Save configuration file"""
        if not config_name:
            return "❌ Please enter a configuration name"

        result = self.config_manager.save_config(config_name, config_content)

        if result['status'] == 'success':
            return f"✅ {result['message']}"
        else:
            return f"❌ {result['message']}"

    def load_config(self, config_file):
        """Load configuration file"""
        if not config_file:
            return ""
        return self.config_manager.load_config(config_file)

    def validate_config(self, config_content):
        """Validate configuration"""
        result = self.config_manager.validate_config(config_content)

        output = "🔍 Validation Results:\n\n"

        if result['valid']:
            output += "✅ Configuration is valid!\n"
        else:
            output += "❌ Configuration has errors:\n"
            for error in result['errors']:
                output += f"  • {error}\n"

        if result['warnings']:
            output += "\n⚠️ Warnings:\n"
            for warning in result['warnings']:
                output += f"  • {warning}\n"

        return output

    def save_repo_list(self, list_name, list_content):
        """Save repository list"""
        if not list_name:
            return "❌ Please enter a list name"

        result = self.config_manager.save_repo_list(list_name, list_content)

        if result['status'] == 'success':
            return f"✅ {result['message']} ({result['count']} repositories)"
        else:
            return f"❌ {result['message']}"

    def add_repo_to_list(self, list_content, url, branch, collection):
        """Add repository to list"""
        if not url or not collection:
            return list_content, "❌ URL and collection name are required"

        new_content = self.config_manager.add_repository_to_list(
            list_content, url, branch, collection
        )
        return new_content, f"✅ Added {url} to list"

    def create_interface(self):
        """Create the Gradio interface"""

        with gr.Blocks(title="GitHub → Qdrant Processor", theme=gr.themes.Soft()) as app:
            gr.Markdown(
                """
                # 🚀 GitHub to Qdrant Vector Processing Pipeline
                Transform GitHub repositories into searchable vector databases for AI applications.
                """
            )

            with gr.Tabs():
                # Tab 1: Quick Process
                with gr.Tab("⚡ Quick Process"):
                    with gr.Row():
                        with gr.Column():
                            config_dropdown = gr.Dropdown(
                                label="Configuration File",
                                choices=self.config_manager.list_configs(),
                                value="config.yaml" if "config.yaml" in self.config_manager.list_configs() else None
                            )

                            repo_url = gr.Textbox(
                                label="Repository URL",
                                placeholder="https://github.com/user/repo.git"
                            )

                            with gr.Row():
                                branch = gr.Textbox(
                                    label="Branch (optional)",
                                    placeholder="main"
                                )
                                collection = gr.Textbox(
                                    label="Collection Name (optional)",
                                    placeholder="my-collection"
                                )

                            with gr.Row():
                                process_btn = gr.Button("🔄 Process Repository", variant="primary")
                                test_conn_btn = gr.Button("🔌 Test Connections")

                            connection_output = gr.Textbox(
                                label="Connection Status",
                                lines=4
                            )

                    log_output = gr.Textbox(
                        label="Processing Log",
                        lines=20,
                        max_lines=30
                    )

                    result_output = gr.JSON(label="Processing Results", visible=True)

                # Tab 2: Batch Process
                with gr.Tab("📦 Batch Process"):
                    with gr.Row():
                        with gr.Column():
                            batch_config = gr.Dropdown(
                                label="Configuration File",
                                choices=self.config_manager.list_configs(),
                                value="config.yaml" if "config.yaml" in self.config_manager.list_configs() else None
                            )

                            repo_list = gr.Dropdown(
                                label="Repository List",
                                choices=self.config_manager.list_repo_lists()
                            )

                            batch_btn = gr.Button("🚀 Start Batch Processing", variant="primary")

                    batch_status = gr.Dataframe(
                        headers=["Repository", "Status", "Files", "Chunks", "Time (s)"],
                        label="Processing Status"
                    )

                    batch_summary = gr.Textbox(label="Batch Summary", lines=5)
                    batch_log = gr.Textbox(label="Batch Log", lines=15)

                # Tab 3: Configuration
                with gr.Tab("⚙️ Configuration"):
                    with gr.Row():
                        with gr.Column():
                            config_selector = gr.Dropdown(
                                label="Load Configuration",
                                choices=self.config_manager.list_configs()
                            )

                            config_name = gr.Textbox(
                                label="Config Name (for saving)",
                                placeholder="my-config.yaml"
                            )

                            config_editor = gr.Code(
                                label="Configuration (YAML)",
                                language="yaml",
                                lines=20,
                                value=self.config_manager.load_config_template()
                            )

                            with gr.Row():
                                save_config_btn = gr.Button("💾 Save Config")
                                validate_btn = gr.Button("✓ Validate")
                                load_template_btn = gr.Button("📋 Load Template")

                            config_status = gr.Textbox(label="Status", lines=5)

                # Tab 4: Repository Lists
                with gr.Tab("📋 Repository Lists"):
                    with gr.Row():
                        with gr.Column():
                            list_selector = gr.Dropdown(
                                label="Load List",
                                choices=self.config_manager.list_repo_lists()
                            )

                            list_name = gr.Textbox(
                                label="List Name (for saving)",
                                placeholder="my-repos.yaml"
                            )

                            list_editor = gr.Code(
                                label="Repository List (YAML)",
                                language="yaml",
                                lines=15,
                                value=self.config_manager.load_repo_list_template()
                            )

                            gr.Markdown("### Add Repository")
                            with gr.Row():
                                add_url = gr.Textbox(
                                    label="Repository URL",
                                    placeholder="https://github.com/user/repo.git"
                                )
                                add_branch = gr.Textbox(
                                    label="Branch (optional)",
                                    placeholder="main"
                                )
                                add_collection = gr.Textbox(
                                    label="Collection Name",
                                    placeholder="collection-name"
                                )
                            add_btn = gr.Button("➕ Add Repository")

                            with gr.Row():
                                save_list_btn = gr.Button("💾 Save List")

                            list_status = gr.Textbox(label="Status", lines=2)

            # Connect event handlers

            # Quick Process Tab
            process_btn.click(
                fn=self.process_single_repo,
                inputs=[config_dropdown, repo_url, branch, collection],
                outputs=[log_output, result_output],
                queue=True
            )

            test_conn_btn.click(
                fn=self.test_connections,
                inputs=[config_dropdown],
                outputs=[connection_output]
            )

            # Batch Process Tab
            batch_btn.click(
                fn=self.process_batch,
                inputs=[batch_config, repo_list],
                outputs=[batch_log, batch_status, batch_summary],
                queue=True
            )

            # Configuration Tab
            config_selector.change(
                fn=self.load_config,
                inputs=[config_selector],
                outputs=[config_editor]
            )

            save_config_btn.click(
                fn=self.save_config,
                inputs=[config_name, config_editor],
                outputs=[config_status]
            )

            validate_btn.click(
                fn=self.validate_config,
                inputs=[config_editor],
                outputs=[config_status]
            )

            load_template_btn.click(
                fn=lambda: self.config_manager.load_config_template(),
                outputs=[config_editor]
            )

            # Repository Lists Tab
            list_selector.change(
                fn=lambda x: self.config_manager.load_repo_list(x) if x else "",
                inputs=[list_selector],
                outputs=[list_editor]
            )

            add_btn.click(
                fn=self.add_repo_to_list,
                inputs=[list_editor, add_url, add_branch, add_collection],
                outputs=[list_editor, list_status]
            )

            save_list_btn.click(
                fn=self.save_repo_list,
                inputs=[list_name, list_editor],
                outputs=[list_status]
            )

        return app


def main():
    """Launch the Gradio UI"""
    ui = GitHubQdrantUI()
    app = ui.create_interface()

    # Enable queue for long-running operations
    app.queue()  # Removed concurrency_count parameter for compatibility

    # Launch the app
    print("🚀 Starting GitHub to Qdrant UI...")
    print("📍 Access the UI at: http://localhost:7860")
    print("📍 To stop the server, press Ctrl+C")

    app.launch(
        server_name="127.0.0.1",  # Changed to localhost for better access
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True  # Automatically open browser
    )


if __name__ == "__main__":
    main()