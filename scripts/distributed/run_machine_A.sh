#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_EXE="${ROOT_DIR}/.venv/Scripts/python.exe"
if [[ ! -x "${PYTHON_EXE}" ]]; then
    PYTHON_EXE="${ROOT_DIR}/.venv/bin/python"
fi
if [[ ! -x "${PYTHON_EXE}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_EXE="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_EXE="python"
    else
        echo "Error: no Python executable found (.venv or system)." >&2
        exit 1
    fi
fi

ARGS=("$@")
HAS_STAGE=0
for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[$i]}" == "--stage" ]]; then
        HAS_STAGE=1
        break
    fi
done
if [[ ${HAS_STAGE} -eq 0 ]]; then
    ARGS+=("--stage" "all")
fi

exec "${PYTHON_EXE}" "${ROOT_DIR}/scripts/distributed/run_split_training.py" --machine A "${ARGS[@]}"
