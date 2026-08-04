"""Ingesta CSV -> Parquet con DuckDB.

Va en: fraud-review-queue/src/fraudq/data/ingest.py

Convierte las dos tablas de train de IEEE-CIS a Parquet y deriva las columnas
de tiempo relativo (`day`, `hour`) desde `TransactionDT`. Nada de fechas de
calendario: `TransactionDT` es un offset en segundos (design.md §3.3).

Por qué DuckDB y no pandas:
  - No carga los ~590k x 430 columnas en memoria; hace streaming CSV->Parquet.
  - `sample_size=-1` fuerza a escanear todas las filas para inferir tipos.
    Muchas columnas son dispersas y una muestra corta las tiparía mal.
  - Parquet queda columnar y comprimido: el EDA y los joins posteriores en
    DuckDB vuelan.

NO se tocan los archivos de test: `test_transaction.csv` no tiene `isFraud`
(design.md §3.3). El split train/calib/test se construye particionando
`train_transaction` por día, no usando los archivos de la competencia.

Uso:
    python -m fraudq.data.ingest
    python -m fraudq.data.ingest --raw-dir data/raw --out-dir data/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

SECONDS_PER_DAY = 86_400
SECONDS_PER_HOUR = 3_600


def _csv_to_parquet_transactions(con: duckdb.DuckDBPyConnection, src: Path, dst: Path) -> None:
    """train_transaction.csv -> transactions.parquet, con day y hour derivados."""
    con.execute(
        f"""
        COPY (
            SELECT
                *,
                CAST(TransactionDT // {SECONDS_PER_DAY} AS INTEGER)              AS day,
                CAST((TransactionDT // {SECONDS_PER_HOUR}) % 24 AS INTEGER)      AS hour
            FROM read_csv_auto('{src.as_posix()}', sample_size=-1)
        )
        TO '{dst.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )


def _csv_to_parquet_identity(con: duckdb.DuckDBPyConnection, src: Path, dst: Path) -> None:
    """train_identity.csv -> identity.parquet (sin transformar)."""
    con.execute(
        f"""
        COPY (
            SELECT * FROM read_csv_auto('{src.as_posix()}', sample_size=-1)
        )
        TO '{dst.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )


def ingest(raw_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    txn_csv = raw_dir / "train_transaction.csv"
    idy_csv = raw_dir / "train_identity.csv"
    for f in (txn_csv, idy_csv):
        if not f.exists():
            raise FileNotFoundError(
                f"No encuentro {f}. Corre scripts/download_data.sh primero."
            )

    txn_pq = out_dir / "transactions.parquet"
    idy_pq = out_dir / "identity.parquet"

    con = duckdb.connect()  # en memoria; DuckDB hace el streaming a disco
    try:
        print(f"transactions: {txn_csv}  ->  {txn_pq}")
        _csv_to_parquet_transactions(con, txn_csv, txn_pq)

        print(f"identity:     {idy_csv}  ->  {idy_pq}")
        _csv_to_parquet_identity(con, idy_csv, idy_pq)

        # Resumen mínimo para confirmar que la conversión no rompió nada.
        # (No es EDA: eso va en el notebook, timeboxed.)
        n_txn, n_days, fraud_rate = con.execute(
            f"""
            SELECT COUNT(*), MAX(day) - MIN(day) + 1, AVG(isFraud)
            FROM read_parquet('{txn_pq.as_posix()}')
            """
        ).fetchone()
        n_idy = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{idy_pq.as_posix()}')"
        ).fetchone()[0]
    finally:
        con.close()

    print(
        f"\nOK  transacciones={n_txn:,}  días={n_days}  "
        f"tasa_fraude={fraud_rate:.4f}  filas_identity={n_idy:,} "
        f"(cobertura {n_idy / n_txn:.1%})"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="CSV -> Parquet (IEEE-CIS) con DuckDB.")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = p.parse_args()
    ingest(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
