#!/bin/bash
# BS2 Autotune launcher for macOS (also works on Linux).
# Double-click it in Finder or run ./start.command in a terminal.
# The first start creates a private Python environment in .venv and installs
# the requirements; later starts go straight to the app.
#
#   ./start.command            start and open the browser
#   ./start.command --demo     start with the built-in virtual synth
#   ./start.command --help     all options

set -u
cd "$(dirname "$0")" || exit 1

VENV=".venv"
READY="$VENV/.bs2autotune-ready"

fail() {
    echo
    echo "Setup failed: $1"
    echo "See the Troubleshooting section of README.md."
    # keep the Terminal window open when started by a double click
    if [ -t 0 ]; then read -r -p "Press Return to close ... " _; fi
    exit 1
}

if [ ! -f "$READY" ] || [ requirements.txt -nt "$READY" ]; then
    echo "Setting up BS2 Autotune (first start only, a few minutes) ..."
    if command -v uv >/dev/null 2>&1; then
        [ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV" || fail "uv could not create the environment"
        uv pip install --python "$VENV/bin/python" -r requirements.txt || fail "the packages could not be installed"
    else
        if [ ! -x "$VENV/bin/python" ]; then
            PY=""
            for candidate in python3.12 python3.11 python3.10 python3; do
                if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c \
                    'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)' 2>/dev/null; then
                    PY="$candidate"
                    break
                fi
            done
            if [ -z "$PY" ]; then
                fail "Python 3.10, 3.11 or 3.12 was not found. Install Python 3.12 from https://www.python.org/downloads/ (or 'brew install python@3.12') and start again"
            fi
            echo "Using $("$PY" --version)"
            "$PY" -m venv "$VENV" || fail "could not create the Python environment"
        fi
        "$VENV/bin/python" -m pip install --quiet --upgrade pip || fail "pip could not be updated"
        "$VENV/bin/python" -m pip install -r requirements.txt || fail "the packages could not be installed"
    fi
    touch "$READY"
    echo "Setup finished."
fi

exec "$VENV/bin/python" -m bs2autotune "$@"
