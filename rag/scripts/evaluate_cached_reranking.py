"""Pointwise ELM reranking over saved retrieval shortlists."""
import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from src.data import CourseData, load_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", required=True, type=Path)
    parser.add_argument("--transcript-cache", required=True, type=Path)
    parser.add_argument("--pointwise-cache", required=True, type=Path)
    parser.add_argument("--split", choices=("practice", "dev"), default="practice")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    data = CourseData(ROOT / "data" / "rag_corpus")
    qrels = data.load_qrels(args.split)
    expected = {row["query_id"] for row in data.audio_items(args.split)}
    transcripts = {
        row["query_id"]: row["transcript"]
        for row in load_jsonl(args.transcript_cache)
    }
    rows = load_jsonl(args.rankings)
    if not rows:
        parser.error("No rankings found")
    method = rows[0]["method"]
    seen = set()
    for row in rows:
        qid = row["query_id"]
        ids = row["record_ids"]
        if qid in seen or qid not in expected or qid not in qrels:
            parser.error(f"{qid}: duplicate query or missing split/qrels entry")
        seen.add(qid)
        if row["split"] != args.split or row["method"] != method:
            parser.error(f"{qid}: inconsistent split or method")
        if qid not in transcripts or hashlib.sha256(
            transcripts[qid].encode()
        ).hexdigest() != row["transcript_sha256"]:
            parser.error(f"{qid}: missing or changed transcript")
        if (len(ids) > 20 or len(ids) != len(set(ids))
                or any(rid not in data.records for rid in ids)):
            parser.error(f"{qid}: invalid shortlist")
    if args.limit is None and seen != expected:
        parser.error(f"Incomplete rankings: {len(seen)}/{len(expected)} queries")
    if args.limit is not None:
        rows = rows[:args.limit]

    load_dotenv(ROOT / ".env")
    from extensions.cached_pointwise import CachedPointwise
    from extensions.llm_reranking import POINTWISE_PROMPT_VARIANT
    from src.generator import ELMGenerator

    model = os.environ.get("ELM_MODEL", "utter-project/EuroLLM-22B-Instruct-2512")
    endpoint = os.environ.get(
        "ELM_BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://elm.edina.ac.uk/api/v1")
    )
    scorer = CachedPointwise(data, args.pointwise_cache, model, endpoint,
                            ELMGenerator, cache_only=args.cache_only)
    cutoffs = (5, 10, 20)
    before = {k: [] for k in cutoffs}
    after = {k: [] for k in cutoffs}
    totals = dict.fromkeys(("requests", "cache_hits", "invalid_outputs"), 0)
    print(f"Method={method} split={args.split} queries={len(rows)} "
          f"model={model} prompt={POINTWISE_PROMPT_VARIANT}", flush=True)

    for i, row in enumerate(rows, 1):
        qid = row["query_id"]
        ids = row["record_ids"]
        ranked, stats, _, error = scorer.rerank(qid, transcripts[qid], ids)
        if len(ranked) != len(ids) or set(ranked) != set(ids):
            raise RuntimeError(f"{qid}: reranker changed the candidate set")
        for key in totals:
            totals[key] += stats[key]
        gold = set(qrels[qid])
        for k in cutoffs:
            before[k].append(len(gold.intersection(ids[:k])) / len(gold))
            after[k].append(len(gold.intersection(ranked[:k])) / len(gold))
        if error:
            print(f"{qid}: kept retrieval order after invalid response: {error}")
        if i == 1 or i % 10 == 0 or i == len(rows):
            print(f"[{i}/{len(rows)}] R@5={before[5][-1]:.3f} -> {after[5][-1]:.3f} "
                  f"requests={totals['requests']} cache_hits={totals['cache_hits']}", flush=True)

    for k in cutoffs:
        baseline, reranked = statistics.mean(before[k]), statistics.mean(after[k])
        print(f"Recall@{k}: {baseline:.6f} -> {reranked:.6f} "
              f"({100 * (reranked - baseline):+.2f} pp)")
    print(f"ELM requests: {totals['requests']} | Cache hits: {totals['cache_hits']} | "
          f"Invalid outputs: {totals['invalid_outputs']}")


if __name__ == "__main__":
    main()
