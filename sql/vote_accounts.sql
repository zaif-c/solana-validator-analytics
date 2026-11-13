-- Vote-account snapshot rows from the public Solana mainnet Accounts table.
-- Table is ~125 GB; BigQuery bills by columns scanned, so trimming the SELECT
-- list (e.g. dropping votes / prior_voters / authorized_withdrawer) cuts cost.
-- Columns kept identical to the original Colab notebook (cell 2) for now.
SELECT
    node_pubkey,
    pubkey,
    block_timestamp,
    commission,
    lamports,
    root_slot,
    epoch_credits,
    votes,
    last_timestamp,
    authorized_voters,
    prior_voters,
    authorized_withdrawer,
    tx_signature,
    retrieval_timestamp,
    block_slot,
    block_hash,
    owner,
    account_type,
    rent_epoch,
    space
FROM `bigquery-public-data.crypto_solana_mainnet_us.Accounts`
WHERE account_type = 'vote'
