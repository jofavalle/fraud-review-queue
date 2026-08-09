"""CSV -> Parquet ingestion with DuckDB.

Converts the two IEEE-CIS train tables to Parquet and derives the relative time
columns (`day`, `hour`) from `TransactionDT`. No calendar dates:
`TransactionDT` is an offset in seconds (design.md §3.3).

Why DuckDB and not pandas:
  - It never loads the ~590k x 430 columns into memory; it streams CSV->Parquet.
  - `sample_size=-1` forces a scan of every row to infer types. Many columns
    are sparse, and a short sample would type them wrongly.
  - Parquet comes out columnar and compressed, so the EDA and the later joins
    in DuckDB fly.

The competition test files are NOT touched: `test_transaction.csv` has no
`isFraud` (design.md §3.3). The train/calib/test split is built by partitioning
`train_transaction` by day, not by using those files.

Usage:
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
    """train_transaction.csv -> transactions.parquet, with day and hour derived."""
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
    """train_identity.csv -> identity.parquet, untransformed."""
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
            raise FileNotFoundError(f"{f} not found. Run scripts/download_data.sh first.")

    txn_pq = out_dir / "transactions.parquet"
    idy_pq = out_dir / "identity.parquet"

    con = duckdb.connect()  # in memory; DuckDB does the streaming to disk
    try:
        print(f"transactions: {txn_csv}  ->  {txn_pq}")
        _csv_to_parquet_transactions(con, txn_csv, txn_pq)

        print(f"identity:     {idy_csv}  ->  {idy_pq}")
        _csv_to_parquet_identity(con, idy_csv, idy_pq)

        # A minimal summary, to confirm the conversion broke nothing. This is
        # not EDA: that belongs in the notebook.
        n_txn, n_days, fraud_rate = con.execute(
            f"""
            SELECT COUNT(*), MAX(day) - MIN(day) + 1, AVG(isFraud)
            FROM read_parquet('{txn_pq.as_posix()}')
            """
        ).fetchone()
        n_idy = con.execute(f"SELECT COUNT(*) FROM read_parquet('{idy_pq.as_posix()}')").fetchone()[
            0
        ]
    finally:
        con.close()

    print(
        f"\nOK  transactions={n_txn:,}  days={n_days}  "
        f"fraud_rate={fraud_rate:.4f}  identity_rows={n_idy:,} "
        f"(coverage {n_idy / n_txn:.1%})"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="CSV -> Parquet (IEEE-CIS) with DuckDB.")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = p.parse_args()
    ingest(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
