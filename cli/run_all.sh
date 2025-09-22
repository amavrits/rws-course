#!/usr/bin/env bash
set -e  # exit immediately on error

# Path to virtual environment
VENV_DIR=".venv"

# Check if .venv exists
if [ -d "$VENV_DIR" ]; then
    echo "✅ Virtual environment already exists. Activating..."
else
    echo "🚀 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "📦 Installing dependencies from requirements.txt..."
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Activate environment
echo "🔗 Activating virtual environment..."
source "$VENV_DIR/bin/activate"


