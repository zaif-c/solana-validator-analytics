"""Shared paths and constants for the pipeline.

Set SVA_DATA_DIR to point the pipeline at a different data directory
(e.g. a scratch copy for experiments) without touching the code.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = ROOT / "sql"


def data_dir() -> Path:
    return Path(os.environ.get("SVA_DATA_DIR", ROOT / "data"))


def raw_path() -> Path:
    return data_dir() / "raw" / "vote_accounts.parquet"


def features_path() -> Path:
    return data_dir() / "processed" / "features.parquet"


def clusters_path() -> Path:
    return data_dir() / "processed" / "clusters.parquet"


def metrics_path() -> Path:
    return data_dir() / "processed" / "metrics.json"


def checks_path() -> Path:
    return data_dir() / "processed" / "checks.json"


# --- Extract -----------------------------------------------------------------
QUERY_FILE = SQL_DIR / "vote_accounts.sql"
# Refuse to run a query that would scan more than this unless --max-gb is raised.
DEFAULT_MAX_GB = 25.0

# Columns that hold arrays of structs. Stored as JSON strings in the raw parquet
# so they round-trip exactly regardless of pyarrow's type inference.
NESTED_COLUMNS = [
    "epoch_credits",
    "votes",
    "last_timestamp",
    "authorized_voters",
    "prior_voters",
]

# --- Features ----------------------------------------------------------------
SLOTS_PER_EPOCH = 432_000
# Timely Vote Credits (SIMD-0033) raised the max from 1 to 16 credits per slot.
# Rated reports mainnet activation at epoch 703 (2024-11-26); `python -m src.checks`
# prints the first epoch where observed credits exceed 432,000 to confirm it.
TVC_ACTIVATION_EPOCH = 703
TVC_CREDITS_PER_SLOT = 16
# Post-TVC max per epoch; the original notebook used this for every epoch.
MAX_EPOCH_CREDITS = SLOTS_PER_EPOCH * TVC_CREDITS_PER_SLOT  # 6,912,000
MAX_COMMISSION = 100


def max_epoch_credits(epoch: int | None, era_aware: bool) -> int:
    if not era_aware:
        return MAX_EPOCH_CREDITS
    if epoch is not None and epoch >= TVC_ACTIVATION_EPOCH:
        return MAX_EPOCH_CREDITS
    return SLOTS_PER_EPOCH


# --- Cleaning profiles -------------------------------------------------------
# "v2"    : current methodology (default).
# "colab" : reproduces the original Colab notebook exactly, for validation.
# Select with SVA_PROFILE=colab. Use a separate SVA_DATA_DIR per profile, since
# stages skip themselves when outputs already exist.
PROFILES = {
    "v2": {
        "dedup_key": "pubkey",                 # vote account = stable validator identity
        "order_by": ["block_slot", "retrieval_timestamp"],  # block_slot is never null
        "stable_sort": True,
        "require_timestamps": True,            # drop rows missing block/retrieval timestamp
        "era_aware_efficiency": True,          # 432k max pre-TVC, 6.912M after
        # Drop epoch records with gain < 0 or > that era's max (e.g. a lone record
        # with previous_credits=0 carrying a lifetime total).
        "drop_invalid_epoch_records": True,
        # Count only real epoch records; no-history accounts carry one all-null
        # placeholder {credits: None, epoch: None, previous_credits: None}.
        "count_real_epochs_only": True,
        # Exclude validators that never earned a vote credit (0 real epoch
        # records). Overlaps ~99.9% with "last vote slot == 0" (never voted).
        "exclude_inactive": True,
    },
    "colab": {
        "dedup_key": "node_pubkey",
        "order_by": ["retrieval_timestamp"],
        "stable_sort": False,
        "require_timestamps": False,
        "era_aware_efficiency": False,
        "drop_invalid_epoch_records": False,
        "count_real_epochs_only": False,
        "exclude_inactive": False,
    },
}


def profile_name() -> str:
    name = os.environ.get("SVA_PROFILE", "v2")
    if name not in PROFILES:
        raise ValueError(f"SVA_PROFILE must be one of {list(PROFILES)}, got {name!r}")
    return name


def settings() -> dict:
    return PROFILES[profile_name()]

FEATURES = [
    "commission",
    "log_lamports",
    "median_efficiency_pct",
    "efficiency_cv",
    "performance_trend",
    "num_epochs_recorded",
    "has_history",
]

# --- Model -------------------------------------------------------------------
N_COMPONENTS = 4
N_CLUSTERS = 6
RANDOM_STATE = 42
N_INIT = 10
