# Learning Validator Archetypes from On-Chain Solana Data

This project clusters Solana validators into behavioral archetypes using public vote-account data. It started as a UCSB PSTAT 135 course project and now continues as a FinTech Labs research thread.

## Pipeline

```
sql/vote_accounts.sql ──extract──▶ data/raw/vote_accounts.parquet
                       ──checks───▶ data/processed/checks.json      (read-only data-quality report)
                       ──features─▶ data/processed/features.parquet (+ features.stats.json)
                       ──model────▶ data/processed/clusters.parquet (+ metrics.json)
```

| Stage | Module | What it does |
|---|---|---|
| extract | `src/extract.py` | Dry-runs the query, checks bytes scanned, pulls `account_type = 'vote'` rows, casts `lamports`, and stores nested columns as JSON. |
| checks | `src/checks.py` | Reports missing timestamps, duplicate and identity-key overlaps, dormant accounts, epoch-credit integrity (gaps, negative values), the TVC boundary and snapshot age. It never changes the data. |
| features | `src/features.py` | Drops rows missing timestamps, keeps the latest snapshot per vote account, computes era-aware efficiency features, drops commission > 100, and imputes missing values. |
| model | `src/model.py` | Applies StandardScaler, then PCA (4 components), then K-Means (k=6, `random_state=42`, `n_init=10`). |

If a stage's output already exists, that stage is skipped. Constants such as the TVC epoch, the feature list, k and the PC count are in `src/config.py`.

### Cleaning profiles

| | `v2` (default) | `colab` (original notebook) |
|---|---|---|
| Identity / dedup key | vote account `pubkey` | `node_pubkey` |
| "Latest" row | max `block_slot` (stable sort) | max `retrieval_timestamp` (rows with no timestamp sort last and win) |
| Rows missing `block_timestamp` / `retrieval_timestamp` | dropped | kept |
| Efficiency denominator | 432,000 before epoch 703 (timely vote credits went live), 6,912,000 from 703 on | 6,912,000 for every epoch |

Choose a profile with `SVA_PROFILE`. Give each profile its own `SVA_DATA_DIR`, because stages skip themselves when their outputs already exist.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
nbstripout --install            # strip notebook outputs on commit
gcloud auth application-default login   # only needed for live pulls
```

## Running

```bash
# Build from the frozen Colab snapshot (no BigQuery cost)
python -m src.pipeline --from-snapshot data/snapshots/raw_vote_accounts_cut_2026-06-15.pkl

# Validate the port: reproduce the original Colab numbers in a separate dir
SVA_PROFILE=colab SVA_DATA_DIR=data/colab python -m src.pipeline --from-snapshot data/snapshots/raw_vote_accounts_cut_2026-06-15.pkl
SVA_PROFILE=colab SVA_DATA_DIR=data/colab pytest tests/test_reproduce_colab.py

# Fresh pull, billed to your own GCP project (dry-run size check + prompt first)
python -m src.pipeline --project <your-gcp-project> --refresh extract

# After editing feature logic: rebuild features and everything downstream
python -m src.pipeline --refresh features

# After changing k / PCs only
python -m src.pipeline --refresh model
```

You can also run each stage on its own, for example `python -m src.features --refresh`. To experiment without overwriting the main outputs, set `SVA_DATA_DIR=/some/other/dir`.

The notebooks in `notebooks/` read only the cached files and never query BigQuery:
- `01_eda`: raw columns, missingness, deduplication, nested fields
- `02_features`: feature distributions, imputation, correlations
- `03_clustering`: scree plot and loadings, elbow and silhouette, cluster profiles

`notebooks/archive/` holds the original Colab notebook for reference.

## Data notes

- The source is `bigquery-public-data.crypto_solana_mainnet_us.Accounts`, which is about 125 GB. Never download it. BigQuery bills by the columns scanned, so trimming the `SELECT` list in `sql/vote_accounts.sql` reduces cost.
- The table is live, so a fresh pull will not match the Colab run. The frozen snapshot is kept for reproducibility.
- `epoch_credits`, `votes`, `last_timestamp`, `authorized_voters` and `prior_voters` are arrays of structs. They are stored as JSON strings in the raw parquet, and `src.data_io.read_raw()` decodes them.
- Known limitation: there is no true delegated-stake feature yet. `log_lamports`, the vote-account balance, is used as a proxy.

## Tests

```bash
pytest tests/test_features.py          # unit tests and a synthetic end-to-end run
pytest tests/test_reproduce_colab.py   # needs SVA_PROFILE=colab outputs (see above)
```

`tests/reference_colab.json` stores the original Colab results: 9,346 raw rows, 8,577 validators, 3,484 complete rows, about 85.4% variance in 4 PCs, silhouette about 0.77, and cluster sizes 35/348/1083/1625/2106/3379. Once you change the query or features, the reproduction test is expected to fail. At that point it serves only as a record of the original baseline.
