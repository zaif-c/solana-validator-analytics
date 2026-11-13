"""Data-quality report on the raw vote-account pull.

    python -m src.checks            # print report, write data/processed/checks.json
    python -m src.checks --refresh  # recompute even if checks.json exists

Read-only: it never changes the data. Run it after every new pull and before
deciding on cleaning rules. Sections:
  rows / missingness, timestamps, duplicates & identity keys,
  dormant accounts, epoch-credit integrity, TVC boundary, snapshot age, misc.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import config
from .data_io import read_raw
from .features import TIMESTAMP_COLUMNS, get_epoch_records, is_valid_record, n_records

# Solana rent: 3,480 lamports per byte-year, 2 years for exemption, +128 bytes overhead.
RENT_LAMPORTS_PER_BYTE_YEAR = 3_480
RENT_EXEMPTION_YEARS = 2
ACCOUNT_STORAGE_OVERHEAD = 128


def rent_exempt_minimum(space: pd.Series) -> pd.Series:
    return (space.astype(float) + ACCOUNT_STORAGE_OVERHEAD) * RENT_LAMPORTS_PER_BYTE_YEAR * RENT_EXEMPTION_YEARS


def _pct(n, d):
    return round(100 * n / d, 2) if d else None


def _vc(s: pd.Series, top: int = 10) -> dict:
    return {str(k): int(v) for k, v in s.value_counts(dropna=False).head(top).items()}


def timestamps(raw: pd.DataFrame) -> dict:
    n = len(raw)
    miss = raw[TIMESTAMP_COLUMNS].isna()
    any_missing = miss.any(axis=1)
    has_ts = raw[~any_missing]
    lost_pubkeys = set(raw["pubkey"]) - set(has_ts["pubkey"])
    return {
        "missing_block_timestamp": int(miss["block_timestamp"].sum()),
        "missing_retrieval_timestamp": int(miss["retrieval_timestamp"].sum()),
        "missing_both": int(miss.all(axis=1).sum()),
        "missing_either": int(any_missing.sum()),
        "missing_either_pct": _pct(any_missing.sum(), n),
        "rows_with_both": int((~any_missing).sum()),
        "vote_accounts_total": int(raw["pubkey"].nunique()),
        "vote_accounts_with_timestamped_row": int(has_ts["pubkey"].nunique()),
        "vote_accounts_lost_if_required": len(lost_pubkeys),
        # Is a missing timestamp ever the *latest* row (by block_slot) for its account?
        "accounts_whose_latest_row_lacks_timestamp": int(
            raw.sort_values("block_slot", kind="stable")
            .drop_duplicates("pubkey", keep="last")[TIMESTAMP_COLUMNS].isna().any(axis=1).sum()
        ),
        "block_vs_retrieval_lag_days": (
            (has_ts["retrieval_timestamp"] - has_ts["block_timestamp"]).dt.total_seconds() / 86400
        ).describe().round(2).to_dict(),
    }


def duplicates(raw: pd.DataFrame) -> dict:
    nested = config.NESTED_COLUMNS
    flat = raw.drop(columns=[c for c in nested if c in raw.columns])
    nodes_per_pubkey = raw.groupby("pubkey")["node_pubkey"].nunique()
    pubkeys_per_node = raw.groupby("node_pubkey")["pubkey"].nunique()
    rows_per_pubkey = raw["pubkey"].value_counts()
    return {
        "rows": len(raw),
        "unique_pubkey": int(raw["pubkey"].nunique()),
        "unique_node_pubkey": int(raw["node_pubkey"].nunique()),
        "unique_pubkey_node_pairs": int(raw[["pubkey", "node_pubkey"]].drop_duplicates().shape[0]),
        "exact_duplicate_rows_flat_cols": int(flat.duplicated().sum()),
        "duplicate_tx_signature": int(raw["tx_signature"].duplicated().sum()),
        "duplicate_pubkey_block_slot": int(raw.duplicated(["pubkey", "block_slot"]).sum()),
        "rows_per_pubkey": _vc(rows_per_pubkey, 10),
        "pubkeys_with_multiple_nodes": int((nodes_per_pubkey > 1).sum()),
        "nodes_with_multiple_pubkeys": int((pubkeys_per_node > 1).sum()),
        "max_pubkeys_per_node": int(pubkeys_per_node.max()),
        "top_nodes_by_pubkey_count": {k: int(v) for k, v in pubkeys_per_node.nlargest(5).items()},
        # Same (pubkey, block_slot) but different content -> conflicting records.
        "conflicting_pubkey_slot_rows": int(
            flat[flat.duplicated(["pubkey", "block_slot"], keep=False)]
            .drop_duplicates().duplicated(["pubkey", "block_slot"]).sum()
        ),
    }


def _last_vote(x):
    if not x:
        return (np.nan, pd.NaT)
    r = x[0]
    return (r.get("slot"), pd.to_datetime(r.get("timestamp"), errors="coerce", utc=True))


def dormancy(latest: pd.DataFrame) -> dict:
    n = len(latest)
    lv = latest["last_timestamp"].map(_last_vote)
    slot = pd.to_numeric(lv.map(lambda t: t[0]), errors="coerce")
    ts = lv.map(lambda t: t[1])
    never_voted = (slot.fillna(0) == 0) | (ts <= pd.Timestamp("1970-01-02", tz="UTC"))
    at_rent_min = latest["lamports"] <= rent_exempt_minimum(latest["space"]) + 1
    n_ep = latest["epoch_credits"].map(n_records)
    return {
        "basis": "latest row per pubkey (by block_slot)",
        "validators": n,
        "never_voted": int(never_voted.sum()), "never_voted_pct": _pct(never_voted.sum(), n),
        "balance_at_rent_exempt_min": int(at_rent_min.sum()), "balance_at_rent_exempt_min_pct": _pct(at_rent_min.sum(), n),
        "epoch_records_0": int((n_ep == 0).sum()),
        "epoch_records_1": int((n_ep == 1).sum()),
        "epoch_records_ge_2": int((n_ep >= 2).sum()),
        "never_voted_and_rent_min": int((never_voted & at_rent_min).sum()),
        "commission_of_never_voted": _vc(latest.loc[never_voted, "commission"], 5),
    }


def epoch_credit_integrity(latest: pd.DataFrame) -> dict:
    malformed = placeholders = out_of_range = negative = zero = over_tvc_max = over_pre_tvc_max_before_boundary = 0
    unsorted = with_gaps = gap_epochs = missing_epoch = 0
    first_over_pre = None
    for recs in latest["epoch_credits"]:
        if not isinstance(recs, list):
            continue
        parsed = get_epoch_records(recs)
        malformed += len(recs) - len(parsed)
        placeholders += sum(all(r.get(k) is None for k in ("credits", "epoch", "previous_credits")) for r in recs)
        out_of_range += sum(not is_valid_record(e, d) for e, d in parsed)
        epochs = [e for e, _ in parsed if e is not None]
        missing_epoch += len(parsed) - len(epochs)
        if epochs != sorted(epochs):
            unsorted += 1
        es = sorted(epochs)
        g = sum(b - a - 1 for a, b in zip(es, es[1:]) if b - a > 1)
        with_gaps += g > 0
        gap_epochs += g
        for e, d in parsed:
            negative += d < 0
            zero += d == 0
            over_tvc_max += d > config.MAX_EPOCH_CREDITS
            if d > config.SLOTS_PER_EPOCH and e is not None:
                first_over_pre = e if first_over_pre is None else min(first_over_pre, e)
                if e < config.TVC_ACTIVATION_EPOCH:
                    over_pre_tvc_max_before_boundary += 1
    return {
        "unparseable_records": malformed,
        "of_which_all_null_placeholders": placeholders,
        "records_out_of_era_range": out_of_range,
        "records_missing_epoch": missing_epoch,
        "validators_unsorted_epochs": unsorted,
        "validators_with_epoch_gaps": int(with_gaps),
        "total_missing_epochs_in_gaps": int(gap_epochs),
        "negative_deltas": int(negative),
        "zero_deltas": int(zero),
        "deltas_over_6912000": int(over_tvc_max),
        # TVC boundary check: credits > 432k are only possible after TVC.
        "configured_tvc_epoch": config.TVC_ACTIVATION_EPOCH,
        "first_epoch_with_credits_over_432000": first_over_pre,
        "deltas_over_432000_before_configured_tvc_epoch": int(over_pre_tvc_max_before_boundary),
    }


def snapshot_age(latest: pd.DataFrame) -> dict:
    ts = pd.to_datetime(latest["block_timestamp"], utc=True)
    last_epoch = latest["epoch_credits"].map(
        lambda x: max((e for e, _ in get_epoch_records(x) if e is not None), default=np.nan)
    )
    era = np.where(last_epoch.isna(), "no_history",
                   np.where(last_epoch >= config.TVC_ACTIVATION_EPOCH, "post_tvc", "pre_tvc"))
    return {
        "latest_snapshot_by_year": {str(k): int(v) for k, v in ts.dt.year.astype("Int64").value_counts(dropna=False).sort_index().items()},
        "last_recorded_epoch": last_epoch.describe().round(1).to_dict(),
        "last_epoch_era": _vc(pd.Series(era)),
    }


def misc(raw: pd.DataFrame, latest: pd.DataFrame) -> dict:
    return {
        "commission_over_100": int((latest["commission"] > config.MAX_COMMISSION).sum()),
        "commission_eq_100": int((latest["commission"] == 100).sum()),
        "space_values": _vc(latest["space"]),
        "rent_epoch_values": _vc(latest["rent_epoch"]),
        "owner_values": _vc(raw["owner"]),
        "account_type_values": _vc(raw["account_type"]),
        "lamports_nonpositive": int((latest["lamports"] <= 0).sum()),
    }


def run_checks(raw: pd.DataFrame) -> dict:
    latest = raw.sort_values("block_slot", kind="stable").drop_duplicates("pubkey", keep="last")
    missing = raw.isna().sum()
    return {
        "rows": {"raw_rows": len(raw), "columns": raw.shape[1],
                 "missing_by_column": {k: int(v) for k, v in missing[missing > 0].items()}},
        "timestamps": timestamps(raw),
        "duplicates": duplicates(raw),
        "dormancy": dormancy(latest),
        "epoch_credits": epoch_credit_integrity(latest),
        "snapshot_age": snapshot_age(latest),
        "misc": misc(raw, latest),
    }


def print_report(rep: dict) -> None:
    for section, body in rep.items():
        print(f"\n== {section} " + "=" * (60 - len(section)))
        for k, v in body.items():
            print(f"  {k:<48} {v}")


def run(refresh: bool = False) -> dict:
    out = config.checks_path()
    if out.exists() and not refresh:
        print(f"[checks] {out} exists, skipping (use --refresh to recompute).")
        return json.loads(out.read_text())
    rep = run_checks(read_raw())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str))
    print_report(rep)
    print(f"\n[checks] wrote {out}")
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refresh", action="store_true")
    run(p.parse_args(argv).refresh)


if __name__ == "__main__":
    main()
