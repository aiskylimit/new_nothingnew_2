#!/usr/bin/env bash
set -euo pipefail
# Backward-compatible entrypoint with the same configurable options.
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_eval.sh" "$@"
