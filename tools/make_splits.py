"""Generate the deterministic 40 / 5 / 6 participant split.

The split is computed off the *full* participant roster present in
WebGazerETRA2018Dataset_Release20180420/, sorted by numeric id ascending.
The last 6 ids become the test split, the next-to-last 5 become val, and
everything before that is train. Re-running on the same dataset gives the
same split — no random seeds, no shuffling.

Writes tools/artifacts/splits.json:

    {
      "train": ["P_01", "P_02", ...],
      "val":   ["P_45", ...],
      "test":  ["P_55", "P_56", ...],
      "generated_from": <count>
    }
"""

from __future__ import annotations

import argparse
import sys

from common import (
    ARTIFACTS_DIR,
    SPLITS_PATH,
    ensure_dir,
    list_participants,
    make_split,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-test", type=int, default=6)
    parser.add_argument("--n-val", type=int, default=5)
    parser.add_argument("--out", default=str(SPLITS_PATH),
                        help=f"Output path (default {SPLITS_PATH}).")
    args = parser.parse_args()

    pids = list_participants()
    if not pids:
        print("No participants found.", file=sys.stderr)
        return 1

    split = make_split(pids, n_test=args.n_test, n_val=args.n_val)
    split["generated_from"] = len(pids)

    ensure_dir(ARTIFACTS_DIR)
    write_json(SPLITS_PATH if args.out == str(SPLITS_PATH) else args.out, split)

    print(f"Wrote split for {len(pids)} participants to {args.out}")
    print(f"  train ({len(split['train'])}): {split['train']}")
    print(f"  val   ({len(split['val'])}): {split['val']}")
    print(f"  test  ({len(split['test'])}): {split['test']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
