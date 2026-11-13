"""Unit tests for cleaning/feature helpers and a synthetic end-to-end run (no BigQuery)."""

import numpy as np
import pandas as pd
import pytest

from src import checks, config, features, model
from src.data_io import read_raw, write_raw

PRE = config.SLOTS_PER_EPOCH          # 432,000
POST = config.MAX_EPOCH_CREDITS       # 6,912,000
TVC = config.TVC_ACTIVATION_EPOCH
V2 = config.PROFILES["v2"]
COLAB = config.PROFILES["colab"]


def ec(*gains, first_epoch=600, start=0):
    """Build epoch_credits records from per-epoch credit gains."""
    recs, prev = [], start
    for i, g in enumerate(gains):
        recs.append({"epoch": first_epoch + i, "credits": prev + g, "previous_credits": prev})
        prev += g
    return recs


# --- helpers -----------------------------------------------------------------

def test_epoch_deltas_and_metrics():
    recs = ec(POST, POST // 2, POST, POST // 2, first_epoch=TVC)
    assert features.get_epoch_deltas(recs) == [POST, POST // 2, POST, POST // 2]
    eff = features.get_efficiencies(recs, era_aware=True)
    assert features.median_efficiency(eff) == pytest.approx(75.0)
    assert features.performance_trend(eff) == pytest.approx(0.0)
    assert np.isnan(features.performance_trend(eff[:3]))
    assert np.isnan(features.efficiency_cv(eff[:1]))
    assert features.get_epoch_deltas(None) == []
    assert features.get_epoch_deltas([{"credits": "x"}]) == []


def test_era_aware_efficiency_across_tvc_boundary():
    # 90% of max both before and after TVC -> flat 90%, no artificial trend.
    recs = ec(int(0.9 * PRE), int(0.9 * PRE), int(0.9 * POST), int(0.9 * POST), first_epoch=TVC - 2)
    eff = features.get_efficiencies(recs, era_aware=True)
    np.testing.assert_allclose(eff, [90, 90, 90, 90])
    assert features.performance_trend(eff) == pytest.approx(0.0)
    # Original (non-era-aware) formula shows a spurious jump.
    legacy = features.get_efficiencies(recs, era_aware=False)
    assert features.performance_trend(legacy) > 80


def _rows(**over):
    base = {"pubkey": "v1", "node_pubkey": "n1", "block_slot": 1,
            "block_timestamp": pd.Timestamp("2025-01-01", tz="UTC"),
            "retrieval_timestamp": pd.Timestamp("2025-01-02", tz="UTC")}
    return {**base, **over}


def test_dedup_v2_uses_pubkey_and_block_slot():
    raw = pd.DataFrame([
        _rows(block_slot=10, node_pubkey="old_node"),
        _rows(block_slot=20, node_pubkey="new_node"),                 # same vote account, re-keyed
        _rows(pubkey="v2", node_pubkey="new_node", block_slot=5),     # 2nd vote account on same node
    ])
    out = features.dedup_latest(raw, V2)
    assert sorted(out["pubkey"]) == ["v1", "v2"]
    assert out.set_index("pubkey").loc["v1", "block_slot"] == 20
    # Colab profile would have merged v1/v2 under new_node and kept old_node separately.
    assert len(features.dedup_latest(raw, COLAB)) == 2
    assert set(features.dedup_latest(raw, COLAB)["node_pubkey"]) == {"old_node", "new_node"}


def test_drop_missing_timestamps():
    raw = pd.DataFrame([_rows(), _rows(pubkey="v2", block_timestamp=pd.NaT),
                        _rows(pubkey="v3", retrieval_timestamp=pd.NaT)])
    assert list(features.drop_missing_timestamps(raw)["pubkey"]) == ["v1"]


# --- synthetic end-to-end ------------------------------------------------------

def synthetic_raw(n=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        k = int(rng.choice([0, 1, 3, 14, 64]))
        first = int(rng.integers(500, 760))
        gains = [int(rng.integers(0, config.max_epoch_credits(first + j, True))) for j in range(k)]
        never = k == 0
        for dup in range(1 + (i % 7 == 0)):
            rows.append({
                "node_pubkey": f"node{i // 2}",  # two vote accounts per node identity
                "pubkey": f"vote{i}",
                "commission": int(rng.choice([0, 5, 10, 100])),
                "lamports": str(int(rng.integers(3e7, 1e12))),
                "space": 3762, "rent_epoch": 361, "owner": "Vote111", "account_type": "vote",
                "tx_signature": f"sig{i}-{dup}",
                "block_slot": 1000 * i + dup,
                "epoch_credits": np.array(ec(*gains, first_epoch=first), dtype=object),
                "votes": np.array([], dtype=object),
                "last_timestamp": np.array([{"slot": 0 if never else 1,
                                             "timestamp": pd.Timestamp(0 if never else "2025-01-01", tz="UTC")}], dtype=object),
                "authorized_voters": np.array([{"epoch": 1, "pubkey": "a"}], dtype=object),
                "prior_voters": np.array([], dtype=object),
                "block_timestamp": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=dup),
                "retrieval_timestamp": pd.Timestamp("2025-01-02", tz="UTC") + pd.Timedelta(days=dup),
            })
    rows.append({**rows[0], "pubkey": "bad", "node_pubkey": "badnode", "commission": 255, "tx_signature": "sigbad"})
    rows.append({**rows[1], "pubkey": "nots", "node_pubkey": "notsnode", "block_timestamp": pd.NaT, "tx_signature": "signots"})
    return pd.DataFrame(rows)


def test_roundtrip_and_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("SVA_DATA_DIR", str(tmp_path))
    raw = synthetic_raw()
    write_raw(raw)
    back = read_raw()
    assert len(back) == len(raw)
    assert back["lamports"].dtype == "float64"
    assert back.loc[0, "epoch_credits"] == [dict(r) for r in raw.loc[0, "epoch_credits"]]

    feats, stats = features.build_features(back, V2)
    assert stats["dropped_missing_timestamps"] == 1
    assert stats["unique_validators"] == 301            # 300 + "bad"
    assert stats["dropped_invalid_commission"] == 1
    assert len(feats) == 300 and feats["pubkey"].is_unique
    assert not feats[config.FEATURES].isna().any().any()
    assert feats["median_efficiency_pct_raw"].max() <= 100

    clusters, metrics = model.fit(feats)
    assert clusters["cluster"].nunique() == config.N_CLUSTERS
    assert sum(metrics["cluster_sizes"].values()) == 300


def test_checks_report(tmp_path, monkeypatch):
    monkeypatch.setenv("SVA_DATA_DIR", str(tmp_path))
    write_raw(synthetic_raw())
    rep = checks.run_checks(read_raw())
    assert rep["timestamps"]["missing_either"] == 1
    assert rep["duplicates"]["nodes_with_multiple_pubkeys"] > 0
    assert rep["duplicates"]["duplicate_tx_signature"] == 0
    assert rep["epoch_credits"]["negative_deltas"] == 0
    assert rep["epoch_credits"]["deltas_over_432000_before_configured_tvc_epoch"] == 0
    assert rep["dormancy"]["never_voted"] > 0


# --- epoch-record validity -------------------------------------------------------

PLACEHOLDER = [{"credits": None, "epoch": None, "previous_credits": None}]


def test_invalid_records_dropped_in_v2():
    lifetime = [{"credits": "2049830130", "epoch": 1078, "previous_credits": "0"}]
    assert len(features.get_efficiencies(lifetime, era_aware=True, drop_invalid=True)) == 0
    assert features.get_efficiencies(lifetime, era_aware=True)[0] > 1000  # legacy keeps it
    # Pre-TVC gain above 432k is impossible -> dropped; post-TVC same gain is fine.
    assert not features.is_valid_record(TVC - 1, PRE + 1)
    assert features.is_valid_record(TVC, PRE + 1)
    assert not features.is_valid_record(TVC, -5)


def test_placeholder_not_counted_as_epoch():
    df = pd.DataFrame({
        "epoch_credits": [PLACEHOLDER, ec(PRE // 2, first_epoch=600), ec(PRE, PRE, first_epoch=600)],
        "lamports": [1e8] * 3, "authorized_voters": [[]] * 3, "prior_voters": [[]] * 3,
    })
    v2 = features.engineer(df, V2)
    assert v2["num_epochs_recorded"].tolist() == [0, 1, 2]
    assert features.engineer(df, COLAB)["num_epochs_recorded"].tolist() == [1, 1, 2]
