"""Save retrieval rankings and diagnose missed evidence without calling ELM."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import CourseData
from src.metrics import recall_at_k, mrr_at_k, ndcg_at_k


def diagnose(query_id, transcript, retrieval, relevance, data):
    """Compare a candidate ranking with public qrels; no model inference here."""
    ranked_ids = retrieval["record_ids"]
    ranks = {record_id: rank for rank, record_id in enumerate(ranked_ids, start=1)}
    relevant_hits = len(set(ranked_ids).intersection(relevance))
    relevant_count = len(relevance)
    if not relevant_count:
        raise ValueError(f"{query_id}: no positive relevance judgements")
    metrics = {f"Recall@{k}": recall_at_k(ranked_ids, relevance, k)
               for k in (5, 10, 20, 100)}
    metrics.update({"MRR@10": mrr_at_k(ranked_ids, relevance, 10),
                    "NDCG@10": ndcg_at_k(ranked_ids, relevance, 10)})
    candidate_oracle = min(5, relevant_hits) / relevant_count
    recall5_ceiling = min(5, relevant_count) / relevant_count
    # This is an upper bound on improvement, not a reranker's predicted score.
    reranking_gain = candidate_oracle - metrics["Recall@5"]
    candidate_gap = recall5_ceiling - candidate_oracle
    if reranking_gain > 1e-12 and candidate_gap > 1e-12:
        opportunity = "ranking_and_candidate_coverage"
    elif reranking_gain > 1e-12:
        opportunity = "ranking"
    elif candidate_gap > 1e-12:
        opportunity = "candidate_coverage"
    else:
        opportunity = "at_recall5_ceiling"
    return {
        "query_id": query_id,
        "transcript": transcript,
        "metrics": metrics,
        "r10_candidate": metrics["Recall@10"] < 1.0,
        "r5_candidate": metrics["Recall@5"] < 1.0,
        "relevant_count": relevant_count,
        "relevant_in_candidates": relevant_hits,
        "relevant_missing_from_candidates": relevant_count - relevant_hits,
        "candidate_count": len(ranked_ids),
        "max_possible_recall5": recall5_ceiling,
        "oracle_recall5_from_candidates": candidate_oracle,
        "potential_reranking_gain_at5": reranking_gain,
        "candidate_coverage_gap_at5": candidate_gap,
        "opportunity": opportunity,
        "retrieval_ms": retrieval["latency_ms"],
        "relevant_documents": [
            {"record_id": record_id, "relevance": grade, "rank": ranks.get(record_id),
             "text": data.record_text(record_id)}
            for record_id, grade in sorted(relevance.items(), key=lambda x: (-x[1], x[0]))
        ],
        "retrieved_documents": [
            {"rank": rank, "record_id": record_id, "score": score,
             "relevance": relevance.get(record_id, 0), "text": data.record_text(record_id)}
            for rank, (record_id, score) in enumerate(
                zip(ranked_ids, retrieval["scores"]), start=1)
        ],
    }


def save_reports(out_dir, rows, split, k_max, cache_path, cache_only, report_limit):
    metrics = list(rows[0]["metrics"])
    summary = {
        "split": split, "queries": len(rows), "k_max": k_max,
        "transcript_cache": str(cache_path.resolve()), "cache_only": cache_only,
        "r10_candidates": sum(r["r10_candidate"] for r in rows),
        "r5_candidates": sum(r["r5_candidate"] for r in rows),
        "r10_perfect_but_r5_imperfect": sum(
            not r["r10_candidate"] and r["r5_candidate"] for r in rows),
        "metrics": {key: sum(r["metrics"][key] for r in rows) / len(rows)
                    for key in metrics},
        "opportunities": {name: sum(r["opportunity"] == name for r in rows)
                          for name in ("ranking", "candidate_coverage",
                                       "ranking_and_candidate_coverage", "at_recall5_ceiling")},
    }
    for key in ("max_possible_recall5", "oracle_recall5_from_candidates",
                "potential_reranking_gain_at5", "candidate_coverage_gap_at5"):
        summary["mean_" + key] = sum(r[key] for r in rows) / len(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = ["query_id", "r10_candidate", "r5_candidate", "opportunity",
              "relevant_count", "relevant_in_candidates", "relevant_missing_from_candidates",
              "max_possible_recall5", "oracle_recall5_from_candidates",
              "potential_reranking_gain_at5", "candidate_coverage_gap_at5", "transcript"]
    with (out_dir / "queries.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields + metrics)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{key: row[key] for key in fields}, **row["metrics"]})

    selected = sorted((r for r in rows if r["r5_candidate"] or r["r10_candidate"]),
                      key=lambda r: (r["metrics"]["Recall@5"], r["query_id"]))
    lines = ["# Retrieval diagnosis", "", f"Split: {split}; queries: {len(rows)}.", "",
             "```json", json.dumps(summary, indent=2), "```", "",
             "R@10 < 1 flags incomplete top-ten coverage. R@5 < 1 is flagged separately.",
             "Some queries have more than five (or ten) relevant records, so perfect recall",
             "at that cutoff is impossible. Compare against max_possible_recall5.", "",
             "The candidate oracle is the best Recall@5 obtainable by reordering the saved",
             "candidate pool. It is an upper bound, not an expected reranker result.",
             "A null relevant-document rank means outside this candidate pool, not absent",
             "from the corpus. Qrels are the public judgements; inspect ASR errors manually.", "",
             f"Showing {min(report_limit, len(selected))} of {len(selected)} flagged queries,",
             "ordered by lowest Recall@5. Full texts and rankings are in queries.jsonl.", ""]
    for row in selected[:report_limit]:
        lines += [f"## {row['query_id']}", "", "Transcript:", "",
                  *["> " + line for line in row["transcript"].splitlines()], "",
                  f"Metrics: {json.dumps(row['metrics'])}", "",
                  f"Opportunity: {row['opportunity']}; candidate oracle Recall@5: "
                  f"{row['oracle_recall5_from_candidates']:.3f}.", "",
                  "Relevant records and ranks:", ""]
        for doc in row["relevant_documents"]:
            rank = doc["rank"] if doc["rank"] is not None else "outside candidate pool"
            lines.append(f"- {doc['record_id']}: {rank} (relevance {doc['relevance']})")
        lines += ["", "Top five records (text excerpts):", ""]
        for doc in row["retrieved_documents"][:5]:
            excerpt = " ".join(doc["text"].split())[:500]
            lines += [f"- Rank {doc['rank']}: {doc['record_id']} "
                      f"(score {doc['score']:.4f}, relevance {doc['relevance']})",
                      f"  {excerpt}"]
        lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["practice", "dev"], default="practice")
    parser.add_argument("--transcript-cache", type=Path, required=True)
    parser.add_argument("--cache-only", action="store_true",
                    help="Require all transcripts in cache; never start ASR (no GPU needed with --device cpu)")
    parser.add_argument("--device", default=None, help="Query encoder device")
    parser.add_argument("--k-max", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, required=True, help="New output directory")
    parser.add_argument("--report-limit", type=int, default=20, help="Queries shown in report.md; all are saved in JSONL/CSV")
    args = parser.parse_args()
    if args.k_max < 100 or args.report_limit < 1:
        parser.error("--k-max must be at least 100; --report-limit must be positive")
    if args.out_dir.exists():
        parser.error("Output directory exists; choose a new directory to preserve prior results")
    cache = {}
    if args.transcript_cache.exists():
        for line in args.transcript_cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["query_id"]] = row["transcript"]
    data_dir = ROOT / "data/rag_corpus"
    data = CourseData(data_dir)
    items = data.audio_items(args.split)
    qrels = data.load_qrels(args.split)
    if not items:
        parser.error("Selected split has no audio queries")
    missing_transcripts = [i["query_id"] for i in items if not isinstance(cache.get(i["query_id"]), str)
               or not cache[i["query_id"]].strip()]
    if args.cache_only and missing_transcripts:
        parser.error(f"Cache is missing {len(missing_transcripts)} nonempty transcripts; first: {missing_transcripts[0]}. "
                 "Use the original cache or omit --cache-only to transcribe missing queries.")
    from src.asr_adapter import ASRAdapter
    from src.encoder import LocalEncoder
    from src.retriever import ExactDenseRetriever

    encoder = LocalEncoder(ROOT / "models/enc_a", device=args.device)
    retriever = ExactDenseRetriever(data_dir / "embeddings/corpus_enc_a.npy",
                                    data_dir / "embeddings/corpus_enc_a_ids.json")
    args.out_dir.mkdir(parents=True)
    rows = []
    with ASRAdapter() as asr, \
            (args.out_dir / "queries.jsonl").open("x", encoding="utf-8") as all_file, \
            (args.out_dir / "r10_candidates.jsonl").open("x", encoding="utf-8") as candidates_file:
        for i, item in enumerate(items, start=1):
            query_id = item["query_id"]
            if query_id in missing_transcripts:
                transcript = asr.transcribe(data_dir / item["audio_file"])
                args.transcript_cache.parent.mkdir(parents=True, exist_ok=True)
                with args.transcript_cache.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"query_id": query_id, "transcript": transcript}, ensure_ascii=False) + "\n")
                cache[query_id] = transcript
            transcript = cache[query_id]
            retrieval = retriever.retrieve(encoder.encode(transcript), k=args.k_max)
            row = diagnose(query_id, transcript, retrieval, qrels[query_id], data)
            row.update({"split": args.split, "audio_file": item["audio_file"]})
            rows.append(row)
            encoded = json.dumps(row, ensure_ascii=False) + "\n"
            all_file.write(encoded)
            all_file.flush()
            if row["r10_candidate"]:
                candidates_file.write(encoded)
                candidates_file.flush()
            print(f"[{i:>3}/{len(items)}] {query_id} "
                  f"R@5={row['metrics']['Recall@5']:.3f} "
                  f"R@10={row['metrics']['Recall@10']:.3f} "
                  f"R@100={row['metrics']['Recall@100']:.3f} {row['opportunity']}", flush=True)
    summary = save_reports(args.out_dir, rows, args.split, args.k_max,
                           args.transcript_cache, args.cache_only, args.report_limit)
    print(json.dumps(summary, indent=2))
    print(f"Reports saved to {args.out_dir}")


if __name__ == "__main__":
    main()
