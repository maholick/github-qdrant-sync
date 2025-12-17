#!/bin/bash

# GitHub to Qdrant Sync Runner
# Usage: ./run_sync.sh [config_file] [--repo-url <url>]

set -e  # Exit on error

# Default config file
CONFIG_FILE="${1:-config.yaml}"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found"
    echo "Usage: ./run_sync.sh [config_file] [--repo-url <url>]"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment 'venv' not found"
    echo "Please create it with: python -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Make sure environment variables are set."
fi

# Run the sync script with all arguments
echo "Running GitHub to Qdrant sync with config: $CONFIG_FILE"
echo ""

# If there are additional arguments beyond the config file, pass them through
if [ $# -gt 1 ]; then
    shift  # Remove first argument (config file) to pass remaining args
    python github_to_qdrant.py "$CONFIG_FILE" "$@"
else
    python github_to_qdrant.py "$CONFIG_FILE"
fi

echo "Sync completed successfully!"
