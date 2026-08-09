#!/usr/bin/env bash
#
# download_data.sh: fetch IEEE-CIS Fraud Detection from Kaggle.
#
# Usage: ./scripts/download_data.sh
#
# Requirements:
#   - The Kaggle CLI:  uv pip install kaggle   (or pipx install kaggle)
#   - Credentials, in any of their four forms: the access token in
#     ~/.kaggle/access_token (chmod 600), the KAGGLE_API_TOKEN variable, the
#     long-standing ~/.kaggle/kaggle.json, or KAGGLE_USERNAME and KAGGLE_KEY.
#   - Having accepted the competition rules once on the web:
#     https://www.kaggle.com/c/ieee-fraud-detection/rules
#
# Idempotent: if the CSV files are already there, it does not download again.
# It unpacks the train tables only. The test files carry NO labels (see
# design.md §3.3) and are ignored on purpose.

set -euo pipefail

COMPETITION="ieee-fraud-detection"
RAW_DIR="${1:-data/raw}"

# --- Environment checks --------------------------------------------------------
if ! command -v kaggle >/dev/null 2>&1; then
  echo "ERROR: the kaggle CLI was not found. Install it with:" >&2
  echo "  uv pip install kaggle" >&2
  exit 1
fi

# Kaggle accepts two methods, and the CLI tries the access-token one first
# (kagglesdk/kaggle_env.py, get_access_token_from_env). Checking kaggle.json
# alone rejected perfectly valid credentials.
if [[ ! -f "${HOME}/.kaggle/access_token" \
   && ! -f "${HOME}/.kaggle/access_token.txt" \
   && -z "${KAGGLE_API_TOKEN:-}" \
   && ! -f "${HOME}/.kaggle/kaggle.json" \
   && ( -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ) ]]; then
  echo "ERROR: no Kaggle credentials. Any one of these will do:" >&2
  echo "  - An access token in ~/.kaggle/access_token (chmod 600), which is what" >&2
  echo "    https://www.kaggle.com/settings/api hands out today" >&2
  echo "  - The KAGGLE_API_TOKEN variable, holding the token or a path to a file" >&2
  echo "  - The long-standing ~/.kaggle/kaggle.json (chmod 600)" >&2
  echo "  - The KAGGLE_USERNAME and KAGGLE_KEY variables" >&2
  exit 1
fi

mkdir -p "${RAW_DIR}"

# --- Download (idempotent) -----------------------------------------------------
if [[ -f "${RAW_DIR}/train_transaction.csv" && -f "${RAW_DIR}/train_identity.csv" ]]; then
  echo "The train CSV files are already in ${RAW_DIR}. Nothing to do."
  exit 0
fi

echo "Downloading '${COMPETITION}' into ${RAW_DIR} ..."
# --file pulls only the tables in use, instead of the full zip of about 1.2 GB.
kaggle competitions download -c "${COMPETITION}" -f train_transaction.csv -p "${RAW_DIR}"
kaggle competitions download -c "${COMPETITION}" -f train_identity.csv   -p "${RAW_DIR}"

# --- Unpacking -----------------------------------------------------------------
shopt -s nullglob
for z in "${RAW_DIR}"/*.zip; do
  echo "Unpacking $(basename "${z}") ..."
  unzip -o -q "${z}" -d "${RAW_DIR}"
  rm -f "${z}"
done

echo "Done. Files in ${RAW_DIR}:"
ls -lh "${RAW_DIR}"/train_*.csv
