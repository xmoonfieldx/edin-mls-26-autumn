"""Compare candidate depths for local and pointwise reranking."""
import argparse
import json
import os
import random
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from evaluate_retrieval_followups import (
    ROOT,
    CourseData,
    LocalFeatures,
    rank_report,
    read_transcripts,
)


def positive_ids(labels, ranked=None):
    if hasattr(labels, "items"):
        return {rid for rid, grade in labels.items() if grade > 0}
    return set(labels)


def oracle_r5(pool, gold):
    return min(5, len(set(pool) & gold)) / len(gold) if gold else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local", "cascade", "elm100"),
                        default="local")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/rag_corpus")
    parser.add_argument("--transcript-cache", type=Path, required=True)
    parser.add_argument("--pointwise-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--set-encoder", action="store_true")
    parser.add_argument("--passage-field", default="text")
    args = parser.parse_args()
    if args.limit < 1 or args.batch_size < 1:
        parser.error("--limit and --batch-size must be positive")
    if args.mode != "local" and args.pointwise_cache is None:
        parser.error("--pointwise-cache is required for ELM modes")
    summary_path = args.output.with_suffix(".summary.json")
    if args.output.suffix != ".jsonl" or args.output.exists() or summary_path.exists():
        parser.error("Choose a fresh .jsonl output path")
    if args.pointwise_cache and args.pointwise_cache.resolve() in {
        args.output.resolve(), summary_path.resolve()
    }:
        parser.error("The score cache must be separate from the outputs")

    data = CourseData(args.data_dir)
    items = sorted(data.audio_items("practice"), key=lambda item: item["query_id"])
    random.Random(args.seed).shuffle(items)
    items = items[:args.limit]
    if not items:
        parser.error("No practice queries found")
    transcripts = read_transcripts(
        args.transcript_cache, [item["query_id"] for item in items]
    )
    qrels = data.load_qrels("practice")
    runtime = LocalFeatures(
        data, args.data_dir, args.device, args.batch_size,
        need_cross=args.mode != "elm100", need_bm25=False,
    )
    corpus_ids = set(runtime.retriever.record_ids)
    runtime.collect(items[0]["query_id"], transcripts[items[0]["query_id"]], 100)

    set_encoder = None
    if args.set_encoder:
        if args.mode != "local":
            parser.error("--set-encoder is supported with --mode local")
        from lightning_ir import CrossEncoderModule
        import torch

        set_encoder = CrossEncoderModule("webis/set-encoder-large")
        set_encoder = set_encoder.to(device=args.device, dtype=torch.float32).eval()

    def score_set(query, ids):
        docs = []
        for rid in ids:
            record = data.records[rid]
            text = record if isinstance(record, str) else record.get(args.passage_field)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"{rid}: no passage text at field {args.passage_field!r}. "
                    "Set --passage-field to the field your MiniLM scorer reads."
                )
            docs.append(text)
        runtime.sync()
        start = perf_counter()
        with torch.inference_mode():
            scores = set_encoder.score(query, docs).scores.flatten()
        scores = scores.cpu().tolist()
        runtime.sync()
        elapsed_ms = (perf_counter() - start) * 1000
        if len(scores) != len(ids):
            raise ValueError(f"Set-Encoder returned {len(scores)} scores for {len(ids)} passages")
        ranked = [rid for rid, _ in sorted(zip(ids, scores), key=lambda pair: -pair[1])]
        return ranked, elapsed_ms

    if set_encoder is not None:
        first_id = items[0]["query_id"]
        warmup = runtime.collect(first_id, transcripts[first_id], 20)
        warmup_ids = [candidate["record_id"] for candidate in warmup["candidates"]]
        score_set(transcripts[first_id], warmup_ids)

    scorer = None
    if args.mode != "local":
        from dotenv import load_dotenv
        from extensions.cached_pointwise import CachedPointwise
        from src.generator import ELMGenerator

        load_dotenv(ROOT / ".env")
        model = os.getenv("ELM_MODEL", "utter-project/EuroLLM-22B-Instruct-2512")
        endpoint = os.getenv("ELM_BASE_URL", os.getenv(
            "OPENAI_BASE_URL", "https://elm.edina.ac.uk/api/v1"
        ))
        scorer = CachedPointwise(
            data, args.pointwise_cache, model, endpoint, ELMGenerator,
            cache_only=False,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.output.open("x", encoding="utf-8", buffering=1) as output:
        for index, item in enumerate(items, 1):
            qid = item["query_id"]
            transcript = transcripts[qid]
            labels = qrels[qid]
            gold = positive_ids(labels)
            features = runtime.collect(qid, transcript, 100)
            candidates = features["candidates"]
            dense100 = [candidate["record_id"] for candidate in candidates]
            dense20 = dense100[:20]
            methods = {}

            def add(name, ranked, pool, **extra):
                methods[name] = {
                    **rank_report(ranked, labels, pool),
                    "ranked_ids": ranked,
                    "pool_ids": pool,
                    "oracle_r5": oracle_r5(pool, gold),
                    "deep_gold_in_top5": len(set(ranked[:5]) & gold - set(dense20)),
                    **extra,
                }

            dense_ms = features["timing_ms"]["dense"]
            add("dense", dense100, dense100, path_ms=dense_ms)

            if runtime.cross is not None:
                local100 = [
                    candidate["record_id"]
                    for candidate in sorted(candidates, key=lambda c: -c["cross"])
                ]
                original20 = set(dense20)
                local20 = [rid for rid in local100 if rid in original20]
                add("local20", local20, dense20)
                add("local100", local100, dense100,
                    path_ms=dense_ms + features["timing_ms"]["cross"])
                if args.mode == "local":
                    # Measure 20-pair inference separately from the shared scores.
                    runtime.sync()
                    started = perf_counter()
                    runtime.cross.score_candidates(transcript, dense20)
                    runtime.sync()
                    methods["local20"]["path_ms"] = (
                        dense_ms + (perf_counter() - started) * 1000
                    )

            if set_encoder is not None:
                set_ranked, set_ms = score_set(transcript, dense20)
                add("set_encoder20", set_ranked, dense20, path_ms=dense_ms + set_ms)

            if scorer is not None:
                def run_elm(name, pool):
                    rank, stats, details, error = scorer.rerank(qid, transcript, pool)
                    if error:
                        raise RuntimeError(f"{qid} {name}: {error}")
                    if len(rank) != len(pool) or set(rank) != set(pool):
                        raise RuntimeError(f"{qid} {name}: ranking changed pool membership")
                    add(name, rank, pool, api=stats)

                run_elm("elm20", dense20)
                if args.mode == "cascade":
                    # Keep dense top five eligible and fill with local favourites.
                    selected20 = list(dict.fromkeys(dense20[:5] + local100))[:20]
                    run_elm("cascade20", selected20)
                else:
                    run_elm("elm100", dense100)

            row = {
                "query_id": qid,
                "oracle_r5": {
                    "dense20": oracle_r5(dense20, gold),
                    "dense100": oracle_r5(dense100, gold),
                    "corpus": oracle_r5(corpus_ids, gold),
                },
                "methods": methods,
            }
            output.write(json.dumps(row, allow_nan=False) + "\n")
            rows.append(row)
            print(f"[{index}/{len(items)}] {qid}", flush=True)

    summary = {
        "queries": len(rows),
        "mode": args.mode,
        "device": args.device,
        "seed": args.seed,
        "oracle_r5": {
            pool: mean(row["oracle_r5"][pool] for row in rows)
            for pool in ("dense20", "dense100", "corpus")
        },
        "methods": {},
    }
    for name in rows[0]["methods"]:
        results = [row["methods"][name] for row in rows]
        baseline = [row["methods"]["dense"]["metrics"]["Recall@5"] for row in rows]
        deltas = [result["metrics"]["Recall@5"] - base
                  for result, base in zip(results, baseline)]
        method = {
            "Recall@5": mean(r["metrics"]["Recall@5"] for r in results),
            "Recall@10": mean(r["metrics"]["Recall@10"] for r in results),
            "oracle_r5": mean(r["oracle_r5"] for r in results),
            "R5_delta_vs_dense": mean(deltas),
            "R5_wins_vs_dense": sum(delta > 0 for delta in deltas),
            "R5_losses_vs_dense": sum(delta < 0 for delta in deltas),
            "queries_with_deep_gold_in_top5": sum(
                r["deep_gold_in_top5"] > 0 for r in results
            ),
        }
        if "path_ms" in results[0]:
            method["median_path_ms"] = median(r["path_ms"] for r in results)
        if "api" in results[0]:
            method["api"] = {
                field: sum(r["api"].get(field, 0) for r in results)
                for field in ("requests", "cache_hits", "logical_pair_scores",
                              "invalid_outputs")
            }
        summary["methods"][name] = method

    with summary_path.open("x", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()