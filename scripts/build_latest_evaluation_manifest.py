#!/usr/bin/env python3
"""Build the complete held-out case/paradigm/tCNR evaluation manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


PARADIGMS = ("block", "multi_step", "pseudo_random_binary")
TCNRS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/simulated/splits/train_val_test_split.json"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    split_data = json.loads(args.splits.read_text())
    rows: list[str] = []
    for split, key in (("validation", "val_cases"), ("test", "test_cases")):
        for case_id in split_data[key]:
            for paradigm in PARADIGMS:
                for tcnr in TCNRS:
                    rows.append(f"{split}\t{case_id}\t{paradigm}\t{tcnr:g}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("split\tcase_id\tparadigm\ttcnr\n" + "\n".join(rows) + "\n")
    print(f"Wrote {len(rows)} held-out conditions to {args.out}")


if __name__ == "__main__":
    main()
