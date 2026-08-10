#!/bin/bash

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
REPO_ROOT="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"

exec python -m pytest "$REPO_ROOT/tests" "$@"
