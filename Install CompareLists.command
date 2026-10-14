#!/bin/bash
# CompareLists installer (macOS). Double-click this file in Finder.
cd "$(dirname "$0")" || exit 1

echo "==============================================="
echo "  CompareLists - installation"
echo "==============================================="
echo

# 1. Make sure uv is available.
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (the Python tool runner)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    echo
    echo "uv was installed but is not on the PATH yet."
    echo "Please close this window, then double-click this file again."
    read -n 1 -s -r -p "Press any key to close..."
    exit 1
  fi
fi

# 2. Install the Python dependencies.
echo
echo "Installing dependencies (a couple of minutes the first time)..."
if ! uv sync; then
  echo
  echo "Installation failed. See the messages above."
  read -n 1 -s -r -p "Press any key to close..."
  exit 1
fi

# 3. Download the matching model once (optional but makes the first run fast).
echo
echo "Downloading the model (first time only, ~1.3 GB)..."
uv run comparelists --prefetch || echo "(No problem - it will download on first use.)"

echo
echo "==============================================="
echo "  Done. You can now run the app by double-clicking"
echo "  'Run CompareLists.command'."
echo "==============================================="
read -n 1 -s -r -p "Press any key to close..."
