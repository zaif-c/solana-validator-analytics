"""Check the `colab` profile reproduces the original Colab results on the snapshot.

    export SVA_PROFILE=colab SVA_DATA_DIR=data/colab
    python -m src.pipeline --from-snapshot data/snapshots/raw_vote_accounts_cut_2026-06-15.pkl
    pytest tests/test_reproduce_colab.py

The cut snapshot keeps rows with block_timestamp <= 2026-06-15 21:45:47 (the
original run's max) plus null-timestamp rows, which gives back exactly the 9,346
original rows. Skips unless outputs in SVA_DATA_DIR were built with the colab profile.

Tolerance: the original dedup (unstable sort on retrieval_timestamp, NaT last)
has 28 node_pubkeys with tied candidate rows, so the winner depends on the row
order BigQuery returned. That flips ~1 validator, so counts allow +/-2.
"""

TOL = 2

import json
from pathlib import Path

import numpy as np
import pytest

from src import config
from src.data_io import read_clusters

REF = json.loads((Path(__file__).parent / "reference_colab.json").read_text())

def _built_with_colab_profile() -> bool:
    p = config.metrics_path()
    return p.exists() and json.loads(p.read_text()).get("profile") == "colab"


pytestmark = [
    pytest.mark.reference,
    pytest.mark.skipif(not _built_with_colab_profile(),
                       reason="needs outputs built with SVA_PROFILE=colab from the snapshot"),
]


@pytest.fixture(scope="module")
def stats():
    return json.loads(config.features_path().with_suffix(".stats.json").read_text())


@pytest.fixture(scope="module")
def metrics():
    return json.loads(config.metrics_path().read_text())


@pytest.mark.parametrize("key", ["raw_rows", "unique_validators", "dropped_invalid_commission",
                                 "complete_rows_pre_impute", "rows_out"])
def test_row_counts(stats, key):
    exact = key in ("raw_rows", "unique_validators", "dropped_invalid_commission", "rows_out")
    assert stats[key] == pytest.approx(REF[key], abs=0 if exact else TOL)


def test_pca_variance(metrics):
    np.testing.assert_allclose(metrics["explained_variance_ratio"], REF["explained_variance_ratio"], atol=1e-3)
    assert metrics["cumulative_variance_kept"] == pytest.approx(REF["cumulative_variance_kept"], abs=1e-3)


def test_cluster_sizes(metrics):
    got = sorted(metrics["cluster_sizes"].values())
    assert all(abs(g - r) <= TOL for g, r in zip(got, REF["cluster_sizes_sorted"])), got


def test_silhouette(metrics):
    assert metrics["silhouette"] == pytest.approx(REF["silhouette_approx"], abs=0.01)


def test_cluster_profiles():
    # Cluster ids are arbitrary; match each reference cluster to the nearest size.
    c = read_clusters()
    sizes = c["cluster"].value_counts()
    med = c.groupby("cluster")[config.FEATURES].median()
    for size, expected in REF["cluster_median_profiles_by_size"].items():
        cid = (sizes - int(size)).abs().idxmin()
        assert abs(sizes[cid] - int(size)) <= TOL
        for feat, val in expected.items():
            assert med.loc[cid, feat] == pytest.approx(val, abs=1e-3), (size, feat)
