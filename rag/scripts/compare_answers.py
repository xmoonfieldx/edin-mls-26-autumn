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


def bootstrap_interval(score_differences):
    """Resample paired query differences with the original seed and draw order."""
    rng = np.random.default_rng(0)
    query_count = len(score_differences)
    bootstrap_means = [
        score_differences[rng.integers(0, query_count, query_count)].mean()
        for _ in range(10000)
    ]
    return np.percentile(bootstrap_means, [2.5, 97.5])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("answers_a")
    parser.add_argument("answers_b")
    parser.add_argument("judge_a")
    parser.add_argument("judge_b")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    answers_a, answers_b = load(args.answers_a), load(args.answers_b)
    judgements_a = load(Path(args.judge_a) / "answer_judge_per_query.jsonl")
    judgements_b = load(Path(args.judge_b) / "answer_judge_per_query.jsonl")
    query_ids = sorted(set(answers_a) & set(answers_b) & set(judgements_a) & set(judgements_b))
    if not query_ids:
        raise SystemExit("No queries are present in all four files")

    same_transcript = sum(
        answers_a[query_id]["transcript"] == answers_b[query_id]["transcript"]
        for query_id in query_ids
    )
    score_differences, evidence_changed = [], []
    for query_id in query_ids:
        score_differences.append(
            judgements_b[query_id]["normalised_answer_score"]
            - judgements_a[query_id]["normalised_answer_score"]
        )
        evidence_changed.append(
            set(answers_a[query_id]["retrieved_record_ids"][:args.k])
            != set(answers_b[query_id]["retrieved_record_ids"][:args.k])
        )
    score_differences, evidence_changed = np.array(score_differences), np.array(evidence_changed)

    lower_bound, upper_bound = bootstrap_interval(score_differences)

    print(f"Queries compared: {len(query_ids)}  (same transcript in {same_transcript})")
    print(f"Mean answer score  A: {np.mean([judgements_a[query_id]['normalised_answer_score'] for query_id in query_ids]):.4f}"
          f"   B: {np.mean([judgements_b[query_id]['normalised_answer_score'] for query_id in query_ids]):.4f}")
    print(f"Mean B - A over all queries: {score_differences.mean():+.4f}  (95% bootstrap CI {lower_bound:+.4f} to {upper_bound:+.4f})")
    print(f"Queries whose top-{args.k} evidence set differs: {int(evidence_changed.sum())}")
    if evidence_changed.any():
        changed_differences = score_differences[evidence_changed]
        print(f"  among those: mean {changed_differences.mean():+.4f}, B better {(changed_differences > 0).sum()}, "
              f"worse {(changed_differences < 0).sum()}, equal {(changed_differences == 0).sum()}")
    if (~evidence_changed).any():
        print(f"  unchanged evidence: mean {score_differences[~evidence_changed].mean():+.4f} "
              f"(generation/judge noise only)")


if __name__ == "__main__":
    main()
