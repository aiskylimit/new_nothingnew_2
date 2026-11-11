#!/usr/bin/env bash
# Shared paths for already provisioned environments; sourcing this installs nothing.
# download.txt keeps the shared model/data directory at the repository root,
# outside the rebuttal_cored code folder. Explicit overrides still take priority.
export REBUTTAL_ASSETS="${REBUTTAL_ASSETS:-$(cd "$PWD/.." && pwd)/rebuttal/assets}"
ENV_ROOT="${REBUTTAL_ENV_ROOT:-/mnt/local/uvenvs}"
export REBUTTAL_VENV="${REBUTTAL_VENV:-$ENV_ROOT/rebuttal}"
export REBUTTAL_PYTHON="${REBUTTAL_PYTHON:-$REBUTTAL_VENV/bin/python}"
export REBUTTAL_PYTHON_LONGCOT="${REBUTTAL_PYTHON_LONGCOT:-$ENV_ROOT/sft/bin/python}"
export REBUTTAL_PYTHON_PALIGN="${REBUTTAL_PYTHON_PALIGN:-$ENV_ROOT/palign/bin/python}"
export REBUTTAL_PYTHON_SGL="${REBUTTAL_PYTHON_SGL:-$ENV_ROOT/sgl/bin/python}"
export REBUTTAL_PYTHON_SSFT="${REBUTTAL_PYTHON_SSFT:-$ENV_ROOT/SegmentSelectiveSFT/bin/python}"
export REBUTTAL_PYTHON_CORED="${REBUTTAL_PYTHON_CORED:-$ENV_ROOT/cored/bin/python}"
EVAL_ENV="${REBUTTAL_EVAL_VENV:-$ENV_ROOT/rebuttal_eval}"
export REBUTTAL_EVAL_PYTHON="${REBUTTAL_EVAL_PYTHON:-$EVAL_ENV/bin/python}"

rebuttal_use_coordinator() {
  if [[ ! -f "$REBUTTAL_PYTHON" || ! -x "$REBUTTAL_PYTHON" ]]; then
    echo "Missing coordinator Python: $REBUTTAL_PYTHON (server environment from rebuttal.txt)" >&2
    return 1
  fi
  local target_bin target_root old_venv old_conda entry clean_path clean_libraries
  local -a entries
  target_bin="$(dirname "$REBUTTAL_PYTHON")"
  target_root="$(dirname "$target_bin")"
  old_venv="${VIRTUAL_ENV:-}"
  old_conda="${CONDA_PREFIX:-}"
  clean_path="$target_bin"
  IFS=: read -r -a entries <<< "${PATH:-}"
  for entry in "${entries[@]}"; do
    [[ -z "$entry" || "$entry" == "$target_bin" ]] && continue
    [[ -n "$old_venv" && "$entry" == "$old_venv/bin" ]] && continue
    [[ -n "$old_conda" && "$entry" == "$old_conda/bin" ]] && continue
    [[ "$entry" == "$ENV_ROOT/"*"/bin" ]] && continue
    clean_path="$clean_path:$entry"
  done
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    clean_libraries=""
    IFS=: read -r -a entries <<< "$LD_LIBRARY_PATH"
    for entry in "${entries[@]}"; do
      [[ -z "$entry" ]] && continue
      if [[ "$entry" != "$target_root/"* ]]; then
        [[ -n "$old_venv" && "$entry" == "$old_venv/"* ]] && continue
        [[ -n "$old_conda" && "$entry" == "$old_conda/"* ]] && continue
        [[ "$entry" == "$ENV_ROOT/"* ]] && continue
      fi
      clean_libraries="${clean_libraries:+$clean_libraries:}$entry"
    done
    export LD_LIBRARY_PATH="$clean_libraries"
  fi
  export PATH="$clean_path" VIRTUAL_ENV="$target_root" PYTHONNOUSERSITE=1
  unset PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONEXECUTABLE __PYVENV_LAUNCHER__
  unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
  export PYTHONPATH="$PWD/sgl/src:$PWD:$PWD/cored"
  PY="$REBUTTAL_PYTHON"
}
