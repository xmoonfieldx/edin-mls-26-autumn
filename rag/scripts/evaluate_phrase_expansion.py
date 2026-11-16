"""Compare three 40-passage pools with cached transcripts and zero ELM calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extensions.lexical_retrieval import BM25Retriever
from extensions.phrase_graph import PhraseGraphIndex
from src.data import CourseData
from src.metrics import recall_at_k, mrr_at_k

METHODS = ("dense", "phrase_graph", "bm25")
BUDGET = 40
PREFIX = 20


def build_pool(dense_ids, expansion_hits, method):
    """Preserve the dense prefix, add up to 20 novel IDs, then dense backfill."""
    if len(dense_ids) < BUDGET or len(set(dense_ids)) != len(dense_ids):
        raise ValueError("dense ranking must contain at least 40 unique passage IDs")
    if method == "dense":
        candidates = [{"record_id": rid, "source": "dense_top40"}
                      for rid in dense_ids[:BUDGET]]
    else:
        candidates = [{"record_id": rid, "source": "dense_top20"}
                      for rid in dense_ids[:PREFIX]]
        seen = set(dense_ids[:PREFIX])
        for hit in expansion_hits:
            rid = hit["record_id"]
            if rid not in seen and len(candidates) < BUDGET:
                candidates.append({**hit, "source": method + "_expansion"})
                seen.add(rid)
        for rid in dense_ids:
            if len(candidates) == BUDGET:
                break
            if rid not in seen:
                candidates.append({"record_id": rid, "source": "dense_backfill"})
                seen.add(rid)
    ids = [c["record_id"] for c in candidates]
    assert len(ids) == len(set(ids)) == BUDGET, "pool must contain 40 unique IDs"
    if method != "dense":
        assert ids[:PREFIX] == dense_ids[:PREFIX], "expansion must retain dense top 20"
    return candidates


def measure_method(method, transcript, dense_ids, shared_dense_ms, phrase_index,
                   bm25, reranker, synchronize=lambda: None):
    """No qrels here. Score every pool separately, without reusing CE scores."""
    synchronize()
    started = perf_counter()
    if method == "phrase_graph":
        hits = phrase_index.expand(dense_ids[:5], set(dense_ids[:PREFIX]), k=20)
    elif method == "bm25":
        # At most 20 hits can overlap the prefix: 40 BM25 hits suffice for
        # 20 novel matches. If fewer matches exist, build_pool backfills.
        hits = [{"record_id": rid, "bm25_rank": rank}
                for rank, rid in enumerate(bm25.retrieve(transcript, k=40), start=1)]
    else:
        hits = []
    candidates = build_pool(dense_ids, hits, method)
    ids = [c["record_id"] for c in candidates]
    pool_ms = (perf_counter() - started) * 1000
    scoring_start = perf_counter()
    if reranker is None:
        scores, ranked = None, None
    else:
        scores = reranker.score_candidates(transcript, ids)
        assert set(scores) == set(ids), "cross-encoder must score all 40 passages"
        ranked = sorted(ids, key=lambda rid: (-scores[rid], rid))
    synchronize()
    finished = perf_counter()
    return {
        "candidates": candidates,
        "reranked_ids": ranked,
        "cross_encoder_scores": scores,
        "cross_encoder_score_count": 0 if scores is None else len(scores),
        "timing_ms": {
            "shared_dense": shared_dense_ms,
            "expansion_and_pool": pool_ms,
            "cross_encoder_and_sort": 0.0 if scores is None else (finished - scoring_start) * 1000,
            "total": shared_dense_ms + (finished - started) * 1000,
        },
    }


def evaluate_pool(result, relevance, dense_ids):
    """Attach labels after candidate construction and scoring are complete."""
    relevant = {rid for rid, grade in relevance.items() if grade > 0}
    pool = {c["record_id"] for c in result["candidates"]}
    baseline = set(dense_ids[:BUDGET])
    expansion = {c["record_id"] for c in result["candidates"]
                 if c["source"].endswith("_expansion")}
    gained = relevant & (pool - baseline)
    lost = relevant & (baseline - pool)
    result.update({
        "candidate_recall@40": recall_at_k(list(pool), relevant, BUDGET),
        "relevant_in_pool": len(pool & relevant),
        "gained_relevant_ids_vs_dense40": sorted(gained),
        "lost_relevant_ids_vs_dense40": sorted(lost),
        "gained_relevant_count_vs_dense40": len(gained),
        "lost_relevant_count_vs_dense40": len(lost),
        "rescued_empty_dense20": not bool(set(dense_ids[:PREFIX]) & relevant)
                                 and bool(pool & relevant),
        "expansion_ids": [c["record_id"] for c in result["candidates"]
                          if c["record_id"] in expansion],
        "dense_backfill_ids": [c["record_id"] for c in result["candidates"]
                               if c["source"] == "dense_backfill"],
        "reranked_metrics": None,
    })
    ranked = result["reranked_ids"]
    if ranked is not None:
        result["reranked_metrics"] = {
            "Recall@5": recall_at_k(ranked, relevant, 5),
            "Recall@10": recall_at_k(ranked, relevant, 10),
            "MRR@10": mrr_at_k(ranked, relevant, 10),
        }
        # Distinguish expansion-source hits (which may already be in dense40)
        # from genuinely new labelled passages relative to dense40.
        result["expansion_relevant_at_cutoff"] = {
            str(k): len(set(ranked[:k]) & expansion & relevant) for k in (5, 10)}
        result["gained_relevant_vs_dense40_at_cutoff"] = {
            str(k): len(set(ranked[:k]) & gained) for k in (5, 10)}
    return result


def summarize(rows, config, build_seconds):
    summary = {"config": config, "queries": len(rows), "index_build_seconds": build_seconds,
               "methods": {}}
    for method in METHODS:
        results = [row["methods"][method] for row in rows]
        item = {
            "candidate_recall@40": mean(r["candidate_recall@40"] for r in results),
            "gained_relevant_count_vs_dense40": sum(r["gained_relevant_count_vs_dense40"] for r in results),
            "lost_relevant_count_vs_dense40": sum(r["lost_relevant_count_vs_dense40"] for r in results),
            "queries_rescued_from_empty_dense20": sum(r["rescued_empty_dense20"] for r in results),
            "expansion_candidates": sum(len(r["expansion_ids"]) for r in results),
            "dense_backfill_candidates": sum(len(r["dense_backfill_ids"]) for r in results),
            "median_query_latency_ms": median(r["timing_ms"]["total"] for r in results),
            "median_stage_latency_ms": {
                key: median(r["timing_ms"][key] for r in results)
                for key in ("shared_dense", "expansion_and_pool", "cross_encoder_and_sort")},
            "cross_encoder_scores": sum(r["cross_encoder_score_count"] for r in results),
        }
        if not config["candidate_only"]:
            item["reranked_metrics"] = {
                key: mean(r["reranked_metrics"][key] for r in results)
                for key in ("Recall@5", "Recall@10", "MRR@10")}
            for field in ("expansion_relevant_at_cutoff", "gained_relevant_vs_dense40_at_cutoff"):
                item[field] = {str(k): sum(r[field][str(k)] for r in results) for k in (5, 10)}
        summary["methods"][method] = item
    return summary


def load_transcripts(path, items):
    if not path.is_file():
        raise ValueError(f"Transcript cache not found: {path}; ASR fallback is disabled")
    cache = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        qid, transcript = row["query_id"], row["transcript"]
        if qid in cache and cache[qid] != transcript:
            raise ValueError(f"Conflicting cached transcripts for {qid} at line {number}")
        cache[qid] = transcript
    for item in items:
        qid = item["query_id"]
        if not isinstance(cache.get(qid), str) or not cache[qid].strip():
            raise ValueError(f"Missing nonempty cached transcript for {qid}; ASR fallback is disabled")
    return cache


def device_synchronizer(device):
    if device.startswith("cuda"):
        import torch
        return lambda: torch.cuda.synchronize(device)
    if device == "mps":
        import torch
        return torch.mps.synchronize
    return lambda: None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["practice", "dev"], default="practice")
    ap.add_argument("--transcript-cache", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="cpu", help="Device for both local models (default: cpu)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-phrase-df", type=int, default=100,
                    help="Maximum phrase document frequency, inclusive (default: 100 passages)")
    ap.add_argument("--candidate-only", action="store_true")
    ap.add_argument("--output", type=Path, required=True, help="New per-query JSONL output path")
    args = ap.parse_args()
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")
    if args.batch_size < 1 or args.max_phrase_df < 2:
        ap.error("--batch-size must be positive and --max-phrase-df must be at least 2")
    summary_path = args.output.with_suffix(".summary.json")
    if args.output.exists() or summary_path.exists() or summary_path == args.output:
        ap.error("Choose a new output path; existing results will not be overwritten")
    try:
        data = CourseData(ROOT / "data/rag_corpus")
        if len(data.records) < BUDGET:
            raise ValueError("Corpus must contain at least 40 passages")
        items = data.audio_items(args.split)
        if args.limit is not None:
            items = items[:args.limit]
        if not items:
            raise ValueError("No queries in selected split")
        cache = load_transcripts(args.transcript_cache, items)
        qrels = data.load_qrels(args.split)
        if any(not qrels.get(item["query_id"]) for item in items):
            raise ValueError("Every selected query needs positive public qrels for evaluation")
    except (ValueError, OSError, KeyError) as exc:
        ap.error(str(exc))

    # No ASR, generator or judge imports. Candidate-only never loads a CE model.
    from src.encoder import LocalEncoder
    from src.retriever import ExactDenseRetriever
    encoder = LocalEncoder(ROOT / "models/enc_a", device=args.device)
    embeddings = ROOT / "data/rag_corpus/embeddings"
    retriever = ExactDenseRetriever(embeddings / "corpus_enc_a.npy", embeddings / "corpus_enc_a_ids.json")
    if set(retriever.record_ids) != set(data.records):
        ap.error("Dense embedding IDs do not match corpus IDs")
    reranker = None
    if not args.candidate_only:
        from extensions.cross_encoder_reranking import CrossEncoderReranker
        reranker = CrossEncoderReranker(data, device=args.device, batch_size=args.batch_size)
    started = perf_counter()
    phrase_index = PhraseGraphIndex(data.records, max_document_frequency=args.max_phrase_df)
    phrase_seconds = perf_counter() - started
    started = perf_counter()
    bm25 = BM25Retriever(data.records)
    bm25_seconds = perf_counter() - started
    build_seconds = {"phrase_index": phrase_seconds, "bm25_index": bm25_seconds}
    print(f"Index builds: phrase={phrase_seconds:.3f}s BM25={bm25_seconds:.3f}s", flush=True)
    config = {
        "split": args.split, "limit": args.limit, "device": args.device,
        "batch_size": args.batch_size, "candidate_only": args.candidate_only,
        "budget": BUDGET, "dense_prefix": PREFIX, "phrase_seeds": 5,
        "max_phrase_df": args.max_phrase_df, "phrase_lengths": [2, 3],
        "transcript_cache": str(args.transcript_cache.resolve()),
        "selected_transcripts_sha256": hashlib.sha256(json.dumps(
            [(item["query_id"], cache[item["query_id"]]) for item in items],
            ensure_ascii=False).encode()).hexdigest(),
        "cross_encoder": None if reranker is None else "cross-encoder/ms-marco-MiniLM-L6-v2",
        "timing": "Includes query encoding, exact dense retrieval with ID tie breaks, pool construction, "
                  "and independent 40-pair CE scoring/sorting. Dense time shared and included in every method. "
                  "Excludes model loading, index construction, qrels evaluation and output I/O. "
                  "Method order rotates; no scoring warm-up and no shared CE score cache.",
    }
    synchronize = device_synchronizer(args.device)
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        for index, item in enumerate(items):
            qid = item["query_id"]
            transcript = cache[qid]
            synchronize()
            started = perf_counter()
            vector = encoder.encode(transcript)
            # Retrieve all scores through the existing API so ties at the
            # top-40 boundary can be resolved by ID, not argpartition order.
            dense = retriever.retrieve(vector, k=len(retriever.record_ids))
            dense_ids = [rid for rid, score in sorted(
                zip(dense["record_ids"], dense["scores"], strict=True),
                key=lambda pair: (-pair[1], pair[0]))]
            synchronize()
            shared_ms = (perf_counter() - started) * 1000
            offset = index % len(METHODS)
            order = METHODS[offset:] + METHODS[:offset]
            results = {}
            for method in order:
                results[method] = measure_method(method, transcript, dense_ids, shared_ms,
                                                  phrase_index, bm25, reranker, synchronize)
            for method in METHODS:
                evaluate_pool(results[method], qrels[qid], dense_ids)
            row = {"query_id": qid, "transcript": transcript, "split": args.split,
                   "dense_top40_ids": dense_ids[:BUDGET], "scoring_order": order,
                   "methods": results}
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            rows.append(row)
            print(f"[{index + 1}/{len(items)}] {qid} candidate R@40: " + " ".join(
                f"{m}={results[m]['candidate_recall@40']:.3f}" for m in METHODS), flush=True)
    summary = summarize(rows, config, build_seconds)
    with summary_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Per-query output: {args.output}; summary: {summary_path}")


if __name__ == "__main__":
    main()
