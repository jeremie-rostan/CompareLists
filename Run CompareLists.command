#!/bin/bash
# Launch the CompareLists web app (macOS). Double-click this file in Finder.
cd "$(dirname "$0")" || exit 1

if [ ! -d ".venv" ]; then
  echo "CompareLists is not installed yet. Running the installer first..."
  echo
  bash "Install CompareLists.command"
fi

echo "Starting CompareLists... your browser will open shortly."
echo "Keep this window open while you use the app. Press Ctrl-C here to stop it."
echo
export COMPARELISTS_OPEN_BROWSER=1
uv run comparelists-web
