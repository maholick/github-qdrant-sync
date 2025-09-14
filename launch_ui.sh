#!/bin/bash

# Launch script for GitHub to Qdrant UI

echo "🚀 GitHub to Qdrant Vector Processing UI Launcher"
echo "================================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Check if main requirements are installed
if ! python -c "import langchain" 2>/dev/null; then
    echo "📦 Installing main requirements..."
    pip install -r requirements.txt
fi

# Check if UI requirements are installed
if ! python -c "import gradio" 2>/dev/null; then
    echo "📦 Installing UI requirements..."
    pip install -r requirements_ui.txt
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p configs repo_lists logs

# Launch the UI
echo ""
echo "🌟 Starting GitHub to Qdrant UI..."
echo "📍 Access the UI at: http://localhost:7860"
echo "📍 For public URL, set share=True in ui_app_simple.py"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================================"

python ui_app_simple.py