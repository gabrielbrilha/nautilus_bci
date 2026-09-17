#!/bin/bash
# Script to launch the BCI Friday Night Funkin' Training Studio GUI

# Resolve the absolute path to the directory where this script resides
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Switch to the Python environment directory
cd "$SCRIPT_DIR/../../fnf_prot/python"

# Run the GUI script using uv
uv run python "$SCRIPT_DIR/train_and_run.py" --gui
