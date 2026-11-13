"""Stage 1: pull vote accounts from BigQuery into data/raw/vote_accounts.parquet.

Usage:
    python -m src.extract --project my-gcp-project          # live pull (dry-run checked)
    python -m src.extract --from-snapshot data/snapshots/raw_vote_accounts.pkl
    python -m src.extract --project my-gcp-project --refresh

Auth: `gcloud auth application-default login` once. The project is billed for
bytes scanned; it can also be set via the GOOGLE_CLOUD_PROJECT env var.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from . import config
from .data_io import write_raw

GB = 1024**3


def load_query(path: Path = config.QUERY_FILE) -> str:
    return path.read_text()


def dry_run_bytes(client, query: str) -> int:
    from google.cloud import bigquery

    job = client.query(
        query,
        job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False),
    )
    return job.total_bytes_processed


def pull_from_bigquery(project: str, max_gb: float, assume_yes: bool = False) -> pd.DataFrame:
    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    query = load_query()

    scanned = dry_run_bytes(client, query)
    print(f"Dry run: query would scan {scanned / GB:,.2f} GB "
          f"(limit {max_gb:,.2f} GB, billed to project '{project}').")
    if scanned > max_gb * GB:
        sys.exit(f"Aborting: scan exceeds --max-gb {max_gb}. Trim columns in "
                 f"{config.QUERY_FILE.name} or raise the limit.")
    if not assume_yes:
        if input("Run the query? [y/N] ").strip().lower() != "y":
            sys.exit("Cancelled.")

    job = client.query(
        query,
        job_config=bigquery.QueryJobConfig(maximum_bytes_billed=int(max_gb * GB)),
    )
    df = job.to_dataframe()
    print(f"Pulled {len(df):,} rows x {df.shape[1]} cols "
          f"({(job.total_bytes_billed or 0) / GB:,.2f} GB billed).")
    return df


def load_snapshot(path: Path) -> pd.DataFrame:
    # Only load pickles you created yourself (pickle can execute code).
    df = pd.read_pickle(path)
    print(f"Loaded snapshot {path} : {len(df):,} rows x {df.shape[1]} cols.")
    return df


def run(project: str | None = None, from_snapshot: Path | None = None,
        refresh: bool = False, max_gb: float = config.DEFAULT_MAX_GB,
        assume_yes: bool = False) -> Path:
    out = config.raw_path()
    if out.exists() and not refresh:
        print(f"[extract] {out} exists, skipping (use --refresh to re-pull).")
        return out

    if from_snapshot:
        df = load_snapshot(Path(from_snapshot))
    else:
        project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            sys.exit("Pass --project <gcp-project-id> or set GOOGLE_CLOUD_PROJECT.")
        df = pull_from_bigquery(project, max_gb, assume_yes)

    write_raw(df, out)
    print(f"[extract] wrote {out}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", help="GCP project to bill (personal project).")
    p.add_argument("--from-snapshot", type=Path, help="Load a pickled DataFrame instead of querying.")
    p.add_argument("--refresh", action="store_true", help="Overwrite existing raw output.")
    p.add_argument("--max-gb", type=float, default=config.DEFAULT_MAX_GB, help="Abort if scan exceeds this.")
    p.add_argument("-y", "--yes", action="store_true", help="Skip the confirmation prompt after the dry run.")
    a = p.parse_args(argv)
    run(a.project, a.from_snapshot, a.refresh, a.max_gb, a.yes)


if __name__ == "__main__":
    main()
