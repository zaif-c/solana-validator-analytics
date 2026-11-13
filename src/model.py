"""Stage 3: features -> StandardScaler -> PCA -> K-Means.

Writes data/processed/clusters.parquet (node_pubkey, features, PC1..PCn, cluster)
and data/processed/metrics.json (explained variance, silhouette, cluster sizes).

Logic mirrors notebook cells 45-71.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config
from .data_io import read_features


def fit(features: pd.DataFrame, n_components: int = config.N_COMPONENTS,
        n_clusters: int = config.N_CLUSTERS) -> tuple[pd.DataFrame, dict]:
    X = features[config.FEATURES].to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X)

    full = PCA().fit(X_scaled)
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X_scaled)
    pcs = [f"PC{i + 1}" for i in range(n_components)]

    km = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_STATE, n_init=config.N_INIT)
    labels = km.fit_predict(Z)

    out = features.copy()
    out[pcs] = Z
    out["cluster"] = labels

    metrics = {
        "profile": config.profile_name(),
        "n_rows": int(len(out)),
        "features": config.FEATURES,
        "n_components": n_components,
        "explained_variance_ratio": full.explained_variance_ratio_.round(6).tolist(),
        "cumulative_variance_kept": float(np.cumsum(full.explained_variance_ratio_)[n_components - 1]),
        "loadings": pd.DataFrame(pca.components_.T, index=config.FEATURES, columns=pcs).round(4).to_dict(),
        "n_clusters": n_clusters,
        "inertia": float(km.inertia_),
        "silhouette": float(silhouette_score(Z, labels)),
        "cluster_sizes": {int(k): int(v) for k, v in pd.Series(labels).value_counts().sort_index().items()},
    }
    return out, metrics


def k_sweep(features: pd.DataFrame, ks=range(2, 11), n_components: int = config.N_COMPONENTS) -> pd.DataFrame:
    """Inertia and silhouette across k, for elbow / silhouette plots."""
    Z = PCA(n_components=n_components).fit_transform(
        StandardScaler().fit_transform(features[config.FEATURES].to_numpy(dtype=float))
    )
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=config.N_INIT)
        lab = km.fit_predict(Z)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(Z, lab) if k > 1 else np.nan})
    return pd.DataFrame(rows)


def cluster_profiles(clusters: pd.DataFrame, agg: str = "median") -> pd.DataFrame:
    return clusters.groupby("cluster")[config.FEATURES].agg(agg)


def profile_zscores(profiles: pd.DataFrame) -> pd.DataFrame:
    return ((profiles - profiles.mean()) / profiles.std()).astype(float)


def run(refresh: bool = False) -> None:
    out = config.clusters_path()
    if out.exists() and not refresh:
        print(f"[model] {out} exists, skipping (use --refresh to refit).")
        return
    clusters, metrics = fit(read_features())
    out.parent.mkdir(parents=True, exist_ok=True)
    clusters.to_parquet(out, index=False)
    config.metrics_path().write_text(json.dumps(metrics, indent=2))
    print(f"[model] wrote {out}")
    print(f"[model] cumvar({metrics['n_components']} PCs)={metrics['cumulative_variance_kept']:.4f} "
          f"silhouette={metrics['silhouette']:.4f} sizes={metrics['cluster_sizes']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refresh", action="store_true")
    run(p.parse_args(argv).refresh)


if __name__ == "__main__":
    main()
