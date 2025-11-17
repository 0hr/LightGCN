#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Detect Python ---
PY_EXE="${PYTHON:-}"
if [[ -z "${PY_EXE}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PY_EXE="python3"
  elif command -v python >/dev/null 2>&1; then
    PY_EXE="python"
  else
    echo "Python not found. Please install Python 3.8+ and retry." >&2
    exit 1
  fi
fi

echo "Using Python: $(${PY_EXE} -V)"

VENV_DIR="${ROOT_DIR}/.venv"
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment at ${VENV_DIR}"
  "${PY_EXE}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
echo "Venv activated: ${VIRTUAL_ENV}"


python -m pip install --upgrade pip setuptools wheel
if [[ -f "${ROOT_DIR}/requirements.txt" ]]; then
  echo "Installing requirements.txt"
  python -m pip install -r "${ROOT_DIR}/requirements.txt"
else
  echo "requirements.txt not found; installing minimal deps"
  python -m pip install numpy scipy matplotlib tqdm requests
fi

# --- Download datasets via data.py (preprocessed) ---
echo "Downloading amazon via data.py"
python "${ROOT_DIR}/data.py" --amazon || echo "amazon download skipped/failed"

echo "Downloading gowalla via data.py"
python "${ROOT_DIR}/data.py" --gowalla || echo "gowalla download skipped/failed"

echo "Downloading yelp via data.py"
python "${ROOT_DIR}/data.py" --yelp || echo "yelp download skipped/failed"

# --- Download & process ML-1M via preprocessing.py ---
echo "Processing ml-1m via preprocessing.py (may be large)"
python "${ROOT_DIR}/preprocessing.py" --dataset ml-1m --split ratio || echo "ml-1m processing skipped/failed"

echo "All done. Datasets under ${ROOT_DIR}/data/."
