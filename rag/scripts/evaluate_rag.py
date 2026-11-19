import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extensions.cross_encoder_reranking import CrossEncoderReranker
from extensions.lexical_retrieval import BM25Retriever, reciprocal_rank_fusion
from src.asr_adapter import ASRAdapter
from src.data import CourseData
from src.encoder import LocalEncoder
from src.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from src.retriever import ExactDenseRetriever


def load_cache(path: Path):
    cache = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    cache[row["query_id"]] = row["transcript"]
    return cache


def append_cache(path: Path, qid: str, transcript: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"query_id": qid, "transcript": transcript}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["practice", "dev"], default="practice")
    ap.add_argument("--transcript-cache", type=Path, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--k-max", type=int, default=100)
    ap.add_argument("--rerank-depth", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)

    methods = ap.add_mutually_exclusive_group()
    methods.add_argument("--cross-encoder", action="store_true")
    methods.add_argument("--bm25-rerank", action="store_true")
    methods.add_argument("--hybrid-rrf", action="store_true")
    methods.add_argument(
        "--bm25-slots", type=int, choices=(2, 5),
        help="Reserve up to this many of the first 20 ranks for BM25 hits.",
    )
    methods.add_argument(
        "--hybrid-llm-rerank",
        action="store_true",
        help="Fuse dense and BM25 rankings, then rerank the shortlist with pointwise ELM.",
    )
    methods.add_argument("--llm-rerank", choices=["pointwise", "pairwise", "listwise"])
    methods.add_argument("--export-features", type=Path)
    methods.add_argument("--dense-feedback", choices=("only", "protect-top10", "rrf"))
    ap.add_argument("--feedback-k", type=int, default=3)
    ap.add_argument("--feedback-weight", type=float, default=0.2)

    args = ap.parse_args()

    if args.feedback_k < 1:
        ap.error("--feedback-k must be positive")
    if not 0.0 <= args.feedback_weight <= 1.0:
        ap.error("--feedback-weight must be between 0 and 1")

    if args.k_max < 100:
        ap.error("--k-max must be at least 100 to report Recall@100")

    uses_hybrid = args.hybrid_rrf or args.hybrid_llm_rerank or args.bm25_slots is not None
    llm_strategy = "pointwise" if args.hybrid_llm_rerank else args.llm_rerank

    uses_shortlist = (
        args.cross_encoder or args.bm25_rerank or llm_strategy
        or args.export_features is not None
    )
    if uses_shortlist and not 10 <= args.rerank_depth <= args.k_max:
        ap.error("--rerank-depth must be between 10 and --k-max")

    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")

    data_dir = ROOT / "data" / "rag_corpus"
    data = CourseData(data_dir)
    items = data.audio_items(args.split)
    if args.limit is not None:
        items = items[:args.limit]
    if not items:
        ap.error(f"No queries found for the {args.split} split")

    qrels = data.load_qrels(args.split)
    cache = load_cache(args.transcript_cache) if args.transcript_cache else {}

    asr = ASRAdapter()
    encoder = LocalEncoder(ROOT / "models" / "enc_a", device=args.device)
    retriever = ExactDenseRetriever(
        data_dir / "embeddings" / "corpus_enc_a.npy",
        data_dir / "embeddings" / "corpus_enc_a_ids.json",
    )

    cross_encoder = None
    if args.cross_encoder or args.export_features is not None:
        cross_encoder = CrossEncoderReranker(data, device=args.device or "cpu")

    bm25 = None
    if args.bm25_rerank or uses_hybrid or args.export_features is not None:
        index_started = perf_counter()
        bm25 = BM25Retriever(data.records)
        index_seconds = perf_counter() - index_started
        print(f"BM25 index build time: {index_seconds:.2f} s")

    llm_reranker = None
    llm_totals = {"requests": 0, "prompt_chars": 0, "invalid_outputs": 0}
    if llm_strategy is not None:
        from extensions.llm_reranking import LLMReranker
        from src.generator import ELMGenerator

        llm_reranker = LLMReranker(data, ELMGenerator())

    depth = args.rerank_depth
    rows = []
    feature_rows = []

    if args.dense_feedback:
        row_by_id = {
            record_id: row for row, record_id in enumerate(retriever.record_ids)
        }
    gained_at20 = 0
    lost_at20 = 0

    for i, item in enumerate(items, start=1):
        qid = item["query_id"]
        if qid in cache:
            transcript = cache[qid]
        else:
            audio = data_dir / item["audio_file"]
            transcript = asr.transcribe(audio)
            if args.transcript_cache:
                append_cache(args.transcript_cache, qid, transcript)

        started = perf_counter()
        qvec = encoder.encode(transcript)
        ret = retriever.retrieve(qvec, k=args.k_max)
        ranked = ret["record_ids"]

        if args.dense_feedback:
            dense_ids = ranked
            seed_ids = dense_ids[:args.feedback_k]
            seed_rows = [row_by_id[record_id] for record_id in seed_ids]
            centroid = retriever.embeddings[seed_rows].mean(axis=0)
            query = np.asarray(qvec, dtype=np.float32).reshape(-1)
            query = query / np.linalg.norm(query)
            weight = args.feedback_weight
            refined = (1 - weight) * query + weight * centroid
            feedback_ids = retriever.retrieve(refined, k=args.k_max)["record_ids"]
            if args.dense_feedback == "only":
                ranked = feedback_ids
            elif args.dense_feedback == "protect-top10":
                ranked = list(dict.fromkeys(dense_ids[:10] + feedback_ids))[:args.k_max]
            else:
                fused = reciprocal_rank_fusion(dense_ids[:50], feedback_ids[:50], k=20)
                ranked = list(dict.fromkeys(fused + dense_ids))[:args.k_max]

        if uses_hybrid:
            lexical_ids = bm25.retrieve(transcript, k=args.k_max)
            if args.bm25_slots is None:
                ranked = reciprocal_rank_fusion(ranked, lexical_ids, k=args.k_max)
            else:
                prefix = ranked[:20 - args.bm25_slots]
                bm25_additions = [rid for rid in lexical_ids if rid not in prefix]
                # Continue in dense order to backfill and retain ranks beyond 20.
                ranked = list(dict.fromkeys(
                    prefix + bm25_additions[:args.bm25_slots] + ranked
                ))[:args.k_max]

        llm_stats = None
        if args.export_features is not None:
            # Collect all three scores for the same dense shortlist.
            candidate_ids = ranked[:depth]
            lexical_scores = bm25.score_candidates(transcript, candidate_ids)
            cross_scores = cross_encoder.score_candidates(transcript, candidate_ids)

            candidates = []
            for record_id, dense_score in zip(
                candidate_ids, ret["scores"][:depth], strict=True
            ):
                candidates.append({
                    "record_id": record_id,
                    "dense": dense_score,
                    "bm25": lexical_scores[record_id],
                    "cross": cross_scores[record_id],
                })
            feature_rows.append({"query_id": qid, "candidates": candidates})
        elif cross_encoder is not None:
            reranked = cross_encoder.rerank(transcript, ranked[:depth])
            ranked = reranked + ranked[depth:]
        elif args.bm25_rerank:
            reranked = bm25.rerank(transcript, ranked[:depth])
            ranked = reranked + ranked[depth:]
        elif llm_reranker is not None:
            reranked, llm_stats = llm_reranker.rerank(
                transcript, ranked[:depth], llm_strategy
            )
            ranked = reranked + ranked[depth:]

        query_ms = (perf_counter() - started) * 1000
        rel = qrels[qid]
        if args.dense_feedback:
            dense_hits = set(dense_ids[:20]).intersection(rel)
            feedback_hits = set(ranked[:20]).intersection(rel)
            gained_at20 += len(feedback_hits - dense_hits)
            lost_at20 += len(dense_hits - feedback_hits)
        row = {
            "query_id": qid,
            "query_ms": query_ms,
            "Recall@5": recall_at_k(ranked, rel, 5),
            "Recall@10": recall_at_k(ranked, rel, 10),
            "Recall@20": recall_at_k(ranked, rel, 20),
            "Recall@100": recall_at_k(ranked, rel, 100),
            "MRR@10": mrr_at_k(ranked, rel, 10),
            "NDCG@10": ndcg_at_k(ranked, rel, 10),
        }
        rows.append(row)

        progress = (
            f"[{i:>3}/{len(items)}] {qid} "
            f"R@5={row['Recall@5']:.3f} R@10={row['Recall@10']:.3f}"
        )
        if llm_stats is not None:
            for key in llm_totals:
                llm_totals[key] += llm_stats[key]
            progress += f" ELM calls={llm_stats['requests']}"
            if llm_stats["invalid_outputs"]:
                progress += " (invalid output, used pre-rerank order)"
        print(progress)

    if args.export_features is not None:
        args.export_features.parent.mkdir(parents=True, exist_ok=True)
        with args.export_features.open("w", encoding="utf-8") as output:
            for feature_row in feature_rows:
                output.write(json.dumps(feature_row, allow_nan=False) + "\n")
        print(f"\nSaved {len(feature_rows)} query rows to {args.export_features}")
        print("Recall below uses dense order. Latency includes feature collection.")

    keys = ["Recall@5", "Recall@10", "Recall@20", "Recall@100", "MRR@10", "NDCG@10"]
    print("\nAggregate:")
    for key in keys:
        values = []
        for row in rows:
            values.append(row[key])
        print(f"  {key:<10} {np.mean(values):.6f}")

    query_times = []
    for row in rows:
        query_times.append(row["query_ms"])
    print(f"  Median query latency: {np.median(query_times):.2f} ms")

    if args.dense_feedback:
        print(f"  Labelled hits gained/lost vs dense at 20: {gained_at20}/{lost_at20}")
    if llm_reranker is not None:
        print(f"  ELM calls attempted: {llm_totals['requests']}")
        print(f"  Prompt characters sent: {llm_totals['prompt_chars']}")
        print(f"  Queries with invalid output: {llm_totals['invalid_outputs']}")


if __name__ == "__main__":
    main()
