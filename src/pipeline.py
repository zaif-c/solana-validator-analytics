"""Run all stages in order. Each stage skips itself if its output exists.

    python -m src.pipeline                       # run whatever is missing
    python -m src.pipeline --refresh features    # rebuild features + everything downstream
    python -m src.pipeline --from-snapshot data/snapshots/raw_vote_accounts.pkl
    python -m src.pipeline --project my-gcp-project --refresh extract
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import checks, config, extract, features, model

STAGES = ["extract", "checks", "features", "model"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refresh", choices=STAGES,
                   help="Re-run this stage and every stage after it.")
    p.add_argument("--project", help="GCP project for the extract stage.")
    p.add_argument("--from-snapshot", type=Path, help="Build raw data from a pickled snapshot.")
    p.add_argument("--max-gb", type=float, default=config.DEFAULT_MAX_GB)
    p.add_argument("-y", "--yes", action="store_true")
    a = p.parse_args(argv)

    start = STAGES.index(a.refresh) if a.refresh else len(STAGES)
    refresh = {s: i >= start for i, s in enumerate(STAGES)}

    print(f"[pipeline] profile={config.profile_name()} data_dir={config.data_dir()}")
    extract.run(a.project, a.from_snapshot, refresh["extract"], a.max_gb, a.yes)
    checks.run(refresh["checks"])
    features.run(refresh["features"])
    model.run(refresh["model"])


if __name__ == "__main__":
    main()
