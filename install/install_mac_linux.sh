#!/usr/bin/env bash
# ---- MultiQTL installer for macOS / Linux -------------------------------------------------
# Run:  bash install/install_mac_linux.sh      (needs Python 3.9+; macOS: python.org installer or Homebrew)
set -e
cd "$(dirname "$0")/.."
PY=$(command -v python3 || command -v python)
[ -z "$PY" ] && { echo "Python 3 not found. Install it from https://www.python.org/downloads/"; exit 1; }
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' || { echo "Python 3.9 or newer is required."; exit 1; }
echo "Creating a private environment in .venv ..."
"$PY" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
chmod +x install/start_multiqtl_mac_linux.command
python -m multiqtl --version
echo
echo "Installation complete. Start MultiQTL with:  install/start_multiqtl_mac_linux.command   (or double-click it on macOS)"
