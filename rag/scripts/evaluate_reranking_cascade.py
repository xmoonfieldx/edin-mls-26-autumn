"""Compare dense retrieval, cross-encoder reranking, and a pointwise ELM cascade."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extensions.cached_pointwise import CachedPointwise
from extensions.cross_encoder_reranking import CrossEncoderReranker
from src.data import CourseData, load_jsonl
from src.encoder import LocalEncoder
from src.generator import ELMGenerator
from src.metrics import recall_at_k
from src.retriever import ExactDenseRetriever

POOL_SIZE = 100
SHORTLIST_SIZE = 30
CUTOFFS = (5, 10, 20, 30, 100)


def keep_tail(reranked, original, depth):
    prefix = original[:depth]
    if len(reranked) != len(prefix) or set(reranked) != set(prefix):
        raise RuntimeError("Reranker changed its input candidate set")
    return list(reranked) + original[depth:]


def main():
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript-cache", type=Path, required=True)
    parser.add_argument("--pointwise-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Batch size and limit must be positive")
    if args.output.exists() or args.output.resolve() in (
        args.transcript_cache.resolve(), args.pointwise_cache.resolve()
    ):
        parser.error("Choose a fresh output path distinct from the cache paths")

    load_dotenv(ROOT / ".env")
    data_dir = ROOT / "data" / "rag_corpus"
    data = CourseData(data_dir)
    qrels = data.load_qrels("practice")
    transcripts = {
        row["query_id"]: row["transcript"] for row in load_jsonl(args.transcript_cache)
    }
    qids = [row["query_id"] for row in data.audio_items("practice")][:args.limit]
    missing = [qid for qid in qids if not transcripts.get(qid) or not qrels.get(qid)]
    if not qids or missing:
        parser.error(f"Need cached transcripts and positive qrels. Missing: {missing[:5]}")

    encoder = LocalEncoder(ROOT / "models" / "enc_a", device=args.device)
    embeddings = data_dir / "embeddings"
    retriever = ExactDenseRetriever(
        embeddings / "corpus_enc_a.npy", embeddings / "corpus_enc_a_ids.json"
    )
    cross_encoder = CrossEncoderReranker(data, device=args.device, batch_size=args.batch_size)
    model = os.environ.get("ELM_MODEL", "utter-project/EuroLLM-22B-Instruct-2512")
    endpoint = os.environ.get(
        "ELM_BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://elm.edina.ac.uk/api/v1")
    )
    pointwise = CachedPointwise(
        data, args.pointwise_cache, model, endpoint, ELMGenerator, cache_only=args.cache_only
    )

    totals = {name: [] for name in ("dense", "cross_encoder", "cascade")}
    counts = dict.fromkeys(("requests", "cache_hits", "invalid_outputs"), 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        for index, qid in enumerate(qids, start=1):
            transcript = transcripts[qid]
            start = perf_counter()
            dense = retriever.retrieve(encoder.encode(transcript), k=POOL_SIZE)["record_ids"]
            retrieved = perf_counter()
            cross = keep_tail(cross_encoder.rerank(transcript, dense), dense, POOL_SIZE)
            scored = perf_counter()
            head, stats, details, error = pointwise.rerank(qid, transcript, cross[:SHORTLIST_SIZE])
            cascade = keep_tail(head, cross, SHORTLIST_SIZE)
            finished = perf_counter()
            for key in counts:
                counts[key] += stats[key]
            rankings = {"dense": dense, "cross_encoder": cross, "cascade": cascade}
            metrics = {}
            for method, ids in rankings.items():
                values = {f"R@{k}": float(recall_at_k(ids, qrels[qid], k)) for k in CUTOFFS}
                totals[method].append(values)
                metrics[method] = values
            output.write(json.dumps({
                "query_id": qid,
                "transcript_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
                "rankings": rankings,
                "metrics": metrics,
                "pointwise": stats,
                "pointwise_score_keys": [row["cache_key"] for row in details],
                "pointwise_error": error,
                "stage_ms": {
                    "dense": (retrieved - start) * 1000,
                    "cross_encoder": (scored - retrieved) * 1000,
                    "pointwise": (finished - scored) * 1000,
                },
            }) + "\n")
            output.flush()
            print(
                f"[{index}/{len(qids)}] {qid} "
                + " ".join(f"{name} R@10={values['R@10']:.3f}" for name, values in metrics.items())
                + f" | ELM calls={stats['requests']} cache hits={stats['cache_hits']}",
                flush=True,
            )
            if error:
                print(f"{qid}: kept cross-encoder order after invalid response: {error}", flush=True)

    print(f"{'Method':<15} " + " ".join(f"{'R@' + str(k):>8}" for k in CUTOFFS))
    for method, rows in totals.items():
        print(f"{method:<15} " + " ".join(f"{mean(row[f'R@{k}'] for row in rows):8.6f}" for k in CUTOFFS))
    print(f"ELM requests: {counts['requests']} | Cache hits: {counts['cache_hits']} | "
          f"Queries with invalid output: {counts['invalid_outputs']}")
    print(f"Per-query rankings and timings: {args.output}")


if __name__ == "__main__":
    main()
