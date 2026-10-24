#!/usr/bin/env bash
# Opens the MultiQTL interface in your web browser. Press Ctrl+C (or close the window) to stop.
cd "$(dirname "$0")/.."
. .venv/bin/activate 2>/dev/null || { echo "Run install/install_mac_linux.sh first."; exit 1; }
python -m multiqtl gui
