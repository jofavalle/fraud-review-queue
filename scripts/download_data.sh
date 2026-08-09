#!/usr/bin/env bash
#
# download_data.sh — descarga IEEE-CIS Fraud Detection desde Kaggle.
#
# Va en: fraud-review-queue/scripts/download_data.sh
# Uso:   ./scripts/download_data.sh
#
# Requisitos:
#   - Kaggle CLI:  uv pip install kaggle   (o pipx install kaggle)
#   - Credenciales, en cualquiera de sus cuatro formas: el token de acceso en
#     ~/.kaggle/access_token (chmod 600), la variable KAGGLE_API_TOKEN, el
#     ~/.kaggle/kaggle.json de siempre, o KAGGLE_USERNAME y KAGGLE_KEY.
#   - Haber aceptado las reglas de la competencia una vez en la web:
#     https://www.kaggle.com/c/ieee-fraud-detection/rules
#
# Idempotente: si los CSV ya existen, no vuelve a descargar.
# Solo desempaqueta las tablas de train. Los archivos de test NO tienen
# etiquetas (ver design.md §3.3) y se ignoran a propósito.

set -euo pipefail

COMPETITION="ieee-fraud-detection"
RAW_DIR="${1:-data/raw}"

# --- Comprobaciones de entorno -------------------------------------------------
if ! command -v kaggle >/dev/null 2>&1; then
  echo "ERROR: no encuentro la CLI de kaggle. Instálala con:" >&2
  echo "  uv pip install kaggle" >&2
  exit 1
fi

# Kaggle admite dos métodos, y la CLI prueba primero el del token de acceso
# (kagglesdk/kaggle_env.py, get_access_token_from_env). Comprobar solo el
# kaggle.json rechazaba credenciales perfectamente válidas.
if [[ ! -f "${HOME}/.kaggle/access_token" \
   && ! -f "${HOME}/.kaggle/access_token.txt" \
   && -z "${KAGGLE_API_TOKEN:-}" \
   && ! -f "${HOME}/.kaggle/kaggle.json" \
   && ( -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ) ]]; then
  echo "ERROR: sin credenciales de Kaggle. Vale cualquiera de estas:" >&2
  echo "  - Token de acceso en ~/.kaggle/access_token (chmod 600), que es lo que" >&2
  echo "    entrega hoy https://www.kaggle.com/settings/api" >&2
  echo "  - La variable KAGGLE_API_TOKEN, con el token o con la ruta a un archivo" >&2
  echo "  - El clásico ~/.kaggle/kaggle.json (chmod 600)" >&2
  echo "  - Las variables KAGGLE_USERNAME y KAGGLE_KEY" >&2
  exit 1
fi

mkdir -p "${RAW_DIR}"

# --- Descarga (idempotente) ----------------------------------------------------
if [[ -f "${RAW_DIR}/train_transaction.csv" && -f "${RAW_DIR}/train_identity.csv" ]]; then
  echo "Los CSV de train ya existen en ${RAW_DIR}. Nada que hacer."
  exit 0
fi

echo "Descargando '${COMPETITION}' en ${RAW_DIR} ..."
# --file baja solo las tablas que usamos, en lugar del zip completo (~1.2 GB).
kaggle competitions download -c "${COMPETITION}" -f train_transaction.csv -p "${RAW_DIR}"
kaggle competitions download -c "${COMPETITION}" -f train_identity.csv   -p "${RAW_DIR}"

# --- Descompresión -------------------------------------------------------------
shopt -s nullglob
for z in "${RAW_DIR}"/*.zip; do
  echo "Descomprimiendo $(basename "${z}") ..."
  unzip -o -q "${z}" -d "${RAW_DIR}"
  rm -f "${z}"
done

echo "Listo. Archivos en ${RAW_DIR}:"
ls -lh "${RAW_DIR}"/train_*.csv
