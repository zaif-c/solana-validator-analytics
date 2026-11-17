# Learning Validator Archetypes from On-Chain Solana Data — v1

> **This is the `v1` branch, an archived version.** The project has three versions, each on its own branch:
>
> | Branch | Version |
> |---|---|
> | **`v1`** (this branch) | The original PSTAT 135 course project (UCSB): a single Colab notebook run on GCP, plus the final paper. |
> | [`v2`](https://github.com/zaif-c/solana-validator-analytics/tree/v2) | v1 ported to a local, staged, re-runnable pipeline, with a corrected data-cleaning method. |
> | [`v3`](https://github.com/zaif-c/solana-validator-analytics/tree/v3) | A rebuild from scratch on a new data source. This is the current version and will become `main`. |

This project clusters Solana validators into behavioral archetypes. The data is public vote-account data from the BigQuery table `bigquery-public-data.crypto_solana_mainnet_us.Accounts`.

## Contents

| File | What it is |
|---|---|
| `Solana_Validator_Analysis.ipynb` | The full analysis. It was run in Colab Enterprise on GCP, and its cell outputs are kept. |
| `Learning Validator Archetypes from On-Chain Solana Data.pdf` | The final paper. |

## Method

1. **Load:** query the vote accounts (`account_type = 'vote'`, 20 columns) from BigQuery, giving 9,346 rows.
2. **Deduplicate:** keep the latest `retrieval_timestamp` per `node_pubkey`, giving 8,577 validators.
3. **Engineer features** from each validator's `epoch_credits` history:
   - median epoch efficiency, measured against 6,912,000 credits per epoch
   - efficiency coefficient of variation
   - performance trend (second half of epochs minus first half)
   - number of epochs recorded
   - log of the vote-account balance (lamports)
   - commission
   - a has-history flag
4. **Clean and impute:** drop the one row with commission above 100%, then fill in missing values for validators with no history.
5. **Cluster:** apply StandardScaler, then PCA keeping 4 components (85.4% of variance), then K-Means with k=6 (silhouette about 0.77).

**The six archetypes:** Low-Commission Mature, Inactive, High-Commission Mature, Elite Performance, Emerging Low-Commission, and Volatile.

## Running it

Open the notebook in Colab or any Jupyter environment that has `google-cloud-bigquery` installed and is signed in to a GCP project. The query in cell 2 is billed to that project for the bytes it scans. The source table is live, so a new run will return more rows than the original.

## Known issues (addressed in v2)

- **Deduplication:** it uses `node_pubkey` instead of the vote account, and rows with no timestamp sort last and win.
- **Efficiency:** every epoch is scored against the post-epoch-703 maximum (timely vote credits). This makes efficiency before epoch 703 look like about 6%, and the "Elite Performance" archetype is mostly an artifact of it.
- **Inactive accounts:** about half the validators never voted and are clustered along with the active ones.
