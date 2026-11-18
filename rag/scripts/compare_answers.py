"""Paired comparison of two answer systems (A = baseline, B = variant).

Usage:
  python rag/scripts/compare_answers.py A.jsonl B.jsonl JUDGE_DIR_A JUDGE_DIR_B [--k 5]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["query_id"]] = row
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("answers_a")
    ap.add_argument("answers_b")
    ap.add_argument("judge_a")
    ap.add_argument("judge_b")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    a, b = load(args.answers_a), load(args.answers_b)
    sa = load(Path(args.judge_a) / "answer_judge_per_query.jsonl")
    sb = load(Path(args.judge_b) / "answer_judge_per_query.jsonl")
    qids = sorted(set(a) & set(b) & set(sa) & set(sb))
    if not qids:
        raise SystemExit("No queries are present in all four files")

    same_transcript = sum(a[q]["transcript"] == b[q]["transcript"] for q in qids)
    diffs, changed = [], []
    for q in qids:
        diffs.append(sb[q]["normalised_answer_score"] - sa[q]["normalised_answer_score"])
        changed.append(set(a[q]["retrieved_record_ids"][:args.k])
                       != set(b[q]["retrieved_record_ids"][:args.k]))
    diffs, changed = np.array(diffs), np.array(changed)

    rng = np.random.default_rng(0)
    boots = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])

    print(f"Queries compared: {len(qids)}  (same transcript in {same_transcript})")
    print(f"Mean answer score  A: {np.mean([sa[q]['normalised_answer_score'] for q in qids]):.4f}"
          f"   B: {np.mean([sb[q]['normalised_answer_score'] for q in qids]):.4f}")
    print(f"Mean B - A over all queries: {diffs.mean():+.4f}  (95% bootstrap CI {lo:+.4f} to {hi:+.4f})")
    print(f"Queries whose top-{args.k} evidence set differs: {int(changed.sum())}")
    if changed.any():
        d = diffs[changed]
        print(f"  among those: mean {d.mean():+.4f}, B better {(d > 0).sum()}, "
              f"worse {(d < 0).sum()}, equal {(d == 0).sum()}")
    if (~changed).any():
        print(f"  unchanged evidence: mean {diffs[~changed].mean():+.4f} "
              f"(generation/judge noise only)")


if __name__ == "__main__":
    main()