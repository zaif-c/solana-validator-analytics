"""Read/write helpers for the raw vote-account table.

Nested BigQuery columns (arrays of structs) come back as numpy arrays of dicts.
They are serialized to JSON strings on write and decoded back to lists of dicts
on read, so downstream code sees the same shape the notebook did.
"""

from __future__ import annotations

import datetime as dt
import decimal
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, decimal.Decimal):
        return int(o) if o == o.to_integral_value() else float(o)
    if isinstance(o, (dt.datetime, dt.date, pd.Timestamp)):
        return o.isoformat()
    raise TypeError(f"Not JSON serializable: {type(o)!r}")


def _encode(x) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return json.dumps(x, default=_json_default)


def _decode(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    return json.loads(s)


def normalize_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Apply early type fixes to a freshly pulled frame."""
    df = df.copy()
    # BigQuery NUMERIC -> Decimal objects; cast once here.
    df["lamports"] = pd.to_numeric(df["lamports"], errors="coerce").astype("float64")
    return df


def write_raw(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or config.raw_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    out = normalize_raw(df)
    for col in config.NESTED_COLUMNS:
        if col in out.columns:
            out[col] = out[col].map(_encode)
    out.to_parquet(path, index=False)
    return path


def read_raw(path: Path | None = None) -> pd.DataFrame:
    path = path or config.raw_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.extract` "
            "(or `--from-snapshot <file.pkl>`) first."
        )
    df = pd.read_parquet(path)
    for col in config.NESTED_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(_decode)
    return df


def read_features(path: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(path or config.features_path())


def read_clusters(path: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(path or config.clusters_path())
