#!/usr/bin/env bash
set -e  # exit immediately on error

clear

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

DIR="results/lr_model"
if [ -d "$DIR" ]; then
  echo "Directory exists: $DIR, deleting..."
  rm -rf -- "$DIR"
  echo "Deleted: $DIR"
fi

python -m main.run_normal_lr_inference --true_intercept 5.0 --true_slope 0.2 --error_sigma 1.0

pyhton -m main.generate_rf --n_x 100 --n_y 50 --mean 5.0 --std 4.0
