"""Stage 2: raw vote accounts -> one row of model features per validator.

Writes data/processed/features.parquet with the identifiers, the final FEATURES,
and intermediate columns (mean_efficiency_pct, *_raw before imputation) for EDA,
plus features.stats.json with row counts at each cleaning step.

Cleaning behaviour comes from the active profile in config.PROFILES:
  v2 (default) : dedup on vote account `pubkey`, latest by block_slot, rows
                 without timestamps dropped, era-aware efficiency (TVC),
                 out-of-range epoch records dropped, null placeholders not
                 counted as epochs, inactive (never-credited) validators dropped.
  colab        : reproduces the original notebook (cells 7, 27-42).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import config
from .data_io import read_raw

TIMESTAMP_COLUMNS = ["block_timestamp", "retrieval_timestamp"]


# --- Per-validator helpers ---------------------------------------------------

def get_epoch_deltas(epoch_credit_records) -> list[int]:
    """Credits earned in each recorded epoch (credits - previous_credits)."""
    return [d for _, d in get_epoch_records(epoch_credit_records)]


def get_epoch_records(epoch_credit_records) -> list[tuple[int | None, int]]:
    """(epoch, credits earned) for each well-formed record."""
    if not isinstance(epoch_credit_records, (list, np.ndarray)):
        return []
    out = []
    for record in epoch_credit_records:
        try:
            delta = int(record["credits"]) - int(record["previous_credits"])
        except (KeyError, TypeError, ValueError):
            continue
        epoch = record.get("epoch")
        out.append((int(epoch) if epoch is not None else None, delta))
    return out


def is_valid_record(epoch: int | None, delta: int) -> bool:
    """Gain must be within [0, max attainable in that epoch's era]."""
    return epoch is not None and 0 <= delta <= config.max_epoch_credits(epoch, era_aware=True)


def valid_epoch_records(epoch_credit_records, drop_invalid: bool) -> list[tuple[int | None, int]]:
    recs = get_epoch_records(epoch_credit_records)
    return [(e, d) for e, d in recs if is_valid_record(e, d)] if drop_invalid else recs


def get_efficiencies(epoch_credit_records, era_aware: bool, drop_invalid: bool = False) -> np.ndarray:
    """Per-epoch efficiency in % of the max credits attainable in that epoch."""
    return np.array([
        d / config.max_epoch_credits(e, era_aware) * 100
        for e, d in valid_epoch_records(epoch_credit_records, drop_invalid)
    ], dtype=float)


def mean_efficiency(eff) -> float:
    return float(np.mean(eff)) if len(eff) > 0 else np.nan


def median_efficiency(eff) -> float:
    return float(np.median(eff)) if len(eff) > 0 else np.nan


def efficiency_cv(eff) -> float:
    if len(eff) < 2:
        return np.nan
    mean = np.mean(eff)
    if mean == 0:
        return np.nan
    return float(np.std(eff) / mean)


def performance_trend(eff) -> float:
    """Mean efficiency of the second half of epochs minus the first half."""
    if len(eff) < 4:
        return np.nan
    eff = np.asarray(eff)
    mid = len(eff) // 2
    return float(np.mean(eff[mid:]) - np.mean(eff[:mid]))


def n_records(x) -> int:
    return len(x) if isinstance(x, (list, np.ndarray)) else 0


# --- Table-level steps -------------------------------------------------------

def drop_missing_timestamps(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.dropna(subset=TIMESTAMP_COLUMNS)


def dedup_latest(raw: pd.DataFrame, settings: dict | None = None) -> pd.DataFrame:
    """Keep the latest snapshot per identity key.

    v2: key = vote account `pubkey`, ordered by block_slot (never null) then
        retrieval_timestamp, stable sort -> deterministic.
    colab: key = node_pubkey, ordered by retrieval_timestamp with the default
        quicksort; NaT timestamps sort last and so "win" (original quirk).
    """
    s = settings or config.settings()
    kind = "stable" if s["stable_sort"] else "quicksort"
    return (
        raw.sort_values(s["order_by"], kind=kind)
        .drop_duplicates(subset=s["dedup_key"], keep="last")
        .reset_index(drop=True)
    )


def engineer(v: pd.DataFrame, s: dict) -> pd.DataFrame:
    v = v.copy()
    drop_invalid = s["drop_invalid_epoch_records"]
    eff = v["epoch_credits"].map(lambda x: get_efficiencies(x, s["era_aware_efficiency"], drop_invalid))
    v["mean_efficiency_pct"] = eff.map(mean_efficiency)
    v["median_efficiency_pct"] = eff.map(median_efficiency)
    v["efficiency_cv"] = eff.map(efficiency_cv)
    v["performance_trend"] = eff.map(performance_trend)
    if s["count_real_epochs_only"]:
        v["num_epochs_recorded"] = v["epoch_credits"].map(
            lambda x: len(valid_epoch_records(x, drop_invalid)))
    else:
        v["num_epochs_recorded"] = v["epoch_credits"].map(n_records)
    v["num_invalid_epoch_records"] = v["epoch_credits"].map(
        lambda x: sum(not is_valid_record(e, d) for e, d in get_epoch_records(x)))
    v["log_lamports"] = np.log10(v["lamports"].astype(float) + 1)
    v["num_authorized_voters"] = v["authorized_voters"].map(n_records)
    v["num_prior_voters"] = v["prior_voters"].map(n_records)
    v["last_epoch"] = v["epoch_credits"].map(
        lambda x: max((e for e, _ in get_epoch_records(x) if e is not None), default=np.nan)
    )
    return v


def drop_invalid(v: pd.DataFrame) -> pd.DataFrame:
    # Commission is a percentage; values > 100 are invalid.
    return v[v["commission"] <= config.MAX_COMMISSION].copy()


def drop_inactive(v: pd.DataFrame) -> pd.DataFrame:
    """Drop validators that never earned a vote credit (no real epoch records)."""
    return v[v["num_epochs_recorded"] > 0].copy()


def impute(v: pd.DataFrame) -> pd.DataFrame:
    v = v.copy()
    v["has_history"] = (v["num_epochs_recorded"] > 1).astype(int)
    for col in ["median_efficiency_pct", "efficiency_cv", "performance_trend"]:
        v[f"{col}_raw"] = v[col]
    cv_median = v.loc[v["has_history"] == 1, "efficiency_cv"].median()
    v["median_efficiency_pct"] = v["median_efficiency_pct"].fillna(0)
    v["performance_trend"] = v["performance_trend"].fillna(0)
    v["efficiency_cv"] = v["efficiency_cv"].fillna(cv_median)
    return v


PRE_IMPUTE_FEATURES = [
    "commission", "log_lamports", "mean_efficiency_pct", "median_efficiency_pct",
    "efficiency_cv", "performance_trend", "num_epochs_recorded",
]

KEEP_COLUMNS = (
    ["node_pubkey", "pubkey", "block_slot", "block_timestamp", "last_epoch"]
    + config.FEATURES
    + ["mean_efficiency_pct", "median_efficiency_pct_raw", "efficiency_cv_raw",
       "performance_trend_raw", "num_authorized_voters", "num_prior_voters", "lamports",
       "num_invalid_epoch_records"]
)


def build_features(raw: pd.DataFrame, settings: dict | None = None) -> tuple[pd.DataFrame, dict]:
    s = settings or config.settings()
    stats = {"profile": config.profile_name() if settings is None else "custom",
             "settings": s, "raw_rows": len(raw)}

    if s["require_timestamps"]:
        n = len(raw)
        raw = drop_missing_timestamps(raw)
        stats["dropped_missing_timestamps"] = n - len(raw)

    v = dedup_latest(raw, s)
    stats["unique_validators"] = len(v)

    v = engineer(v, s)
    stats["validators_with_invalid_epoch_records"] = int((v["num_invalid_epoch_records"] > 0).sum())
    stats["invalid_epoch_records"] = int(v["num_invalid_epoch_records"].sum())
    n_before = len(v)
    v = drop_invalid(v)
    stats["dropped_invalid_commission"] = n_before - len(v)
    if s.get("exclude_inactive"):
        n_before = len(v)
        v = drop_inactive(v)
        stats["dropped_inactive"] = n_before - len(v)
    stats["complete_rows_pre_impute"] = int(v[PRE_IMPUTE_FEATURES].dropna().shape[0])
    v = impute(v)
    stats["rows_out"] = len(v)
    v["commission"] = v["commission"].astype("float64")
    return v[KEEP_COLUMNS].reset_index(drop=True), stats


def run(refresh: bool = False) -> None:
    out = config.features_path()
    if out.exists() and not refresh:
        print(f"[features] {out} exists, skipping (use --refresh to rebuild).")
        return
    feats, stats = build_features(read_raw())
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out, index=False)
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(f"[features] wrote {out}  {json.dumps({k: v for k, v in stats.items() if k != 'settings'})}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refresh", action="store_true")
    run(p.parse_args(argv).refresh)


if __name__ == "__main__":
    main()
