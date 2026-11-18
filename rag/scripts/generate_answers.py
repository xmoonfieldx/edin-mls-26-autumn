"""Generate answer candidates using the audio-to-answer pipeline.

Default: the dense baseline (exact retrieval, top-k). With --rerank pointwise,
the dense top --rerank-depth is rescored by ELM one passage at a time and the
top-k of the reranked list is given to the answer model.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json
from extensions.rerank_retriever import (
    CachedASR, CachingGenerator, EncoderTap, RerankingRetriever, read_jsonl,
)
from src.asr_adapter import ASRAdapter
from src.context_builder import SimpleContextBuilder
from src.data import CourseData
from src.encoder import LocalEncoder
from src.generator import ELMGenerator
from src.rag import RAGPipeline
from src.retriever import ExactDenseRetriever


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["practice", "dev"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Use the first N queries only")
    ap.add_argument("--resume", action="store_true",
                    help="Add missing queries to an existing --output file instead of refusing")
    ap.add_argument("--rerank", choices=["pointwise"], default=None)
    ap.add_argument("--rerank-depth", type=int, default=20)
    ap.add_argument("--transcript-cache", type=Path, default=None,
                    help="Default: rag/.cache/<split>_transcripts.jsonl (shared with evaluate_rag.py)")
    ap.add_argument("--rerank-cache", type=Path, default=None,
                    help="Default: rag/.cache/pointwise_calls.jsonl")
    args = ap.parse_args()

    if args.k < 1 or (args.limit is not None and args.limit < 1):
        ap.error("--k and --limit must be positive")
    if args.rerank and args.rerank_depth < args.k:
        ap.error("--rerank-depth must be at least --k")
    if args.output.exists() and not args.resume:
        ap.error("Output exists; choose a new filename, or pass --resume to add queries to it")

    data_dir = ROOT / "data/rag_corpus"
    data = CourseData(data_dir)
    items = data.audio_items(args.split)
    if args.limit:
        items = items[:args.limit]

    # Resume: keep only complete rows, then skip queries already done.
    done = set()
    if args.output.exists():
        old_rows = read_jsonl(args.output)
        with args.output.open("w", encoding="utf-8") as f:
            for row in old_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        done = {row["query_id"] for row in old_rows}
    pending = [item for item in items if item["query_id"] not in done]
    print(f"{len(done)} queries already in {args.output}; {len(pending)} to generate")
    if not pending:
        return

    transcript_cache = args.transcript_cache or (ROOT / ".cache" / f"{args.split}_transcripts.jsonl")
    path_to_qid = {str(data_dir / it["audio_file"]): it["query_id"] for it in items}
    asr = CachedASR(ASRAdapter(), path_to_qid, transcript_cache)

    encoder = LocalEncoder(ROOT / "models/enc_a", device=args.device)
    retriever = ExactDenseRetriever(
        data_dir / "embeddings/corpus_enc_a.npy",
        data_dir / "embeddings/corpus_enc_a_ids.json",
    )

    prompt_variant = None
    caching_generator = None
    if args.rerank:
        from extensions.llm_reranking import LLMReranker, POINTWISE_PROMPT_VARIANT
        prompt_variant = POINTWISE_PROMPT_VARIANT
        caching_generator = CachingGenerator(
            ELMGenerator(), args.rerank_cache or (ROOT / ".cache" / "pointwise_calls.jsonl")
        )
        encoder = EncoderTap(encoder)
        retriever = RerankingRetriever(
            retriever, encoder, LLMReranker(data, caching_generator),
            args.rerank_depth, caching_generator,
        )
        print(f"Pointwise reranking: depth {args.rerank_depth}, prompt variant '{prompt_variant}'")

    pipeline = RAGPipeline(
        asr, encoder, retriever, data, SimpleContextBuilder(), ELMGenerator()
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    invalid_total = 0
    with args.output.open("a", encoding="utf-8") as f:
        for item in pending:
            result = pipeline.answer_audio(data_dir / item["audio_file"], k=args.k)
            row = {
                "query_id": item["query_id"], "split": args.split,
                "audio_file": item["audio_file"],
                "transcript": result["transcript"], "answer": result["answer"],
                "retrieved_record_ids": result["retrieval"]["record_ids"],
                "timing": result["timing"],
                "system": args.rerank or "dense", "k": args.k,
                "rerank_depth": args.rerank_depth if args.rerank else None,
                "prompt_variant": prompt_variant,
            }
            line = item["query_id"]
            if args.rerank:
                retr = result["retrieval"]
                row["rerank"] = {key: retr[key] for key in (
                    "dense_latency_ms", "rerank_ms", "rerank_stats",
                    "rerank_cache_hits", "rerank_new_calls")}
                invalid_total += retr["rerank_stats"]["invalid_outputs"]
                line += (f"  new ELM calls={retr['rerank_new_calls']}"
                         f" cached={retr['rerank_cache_hits']}")
                if retr["rerank_stats"]["invalid_outputs"]:
                    line += "  (invalid output, dense order used)"
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(line)

    if caching_generator is not None:
        print(f"\nReranking calls: {caching_generator.new_calls} new, "
              f"{caching_generator.hits} from cache; queries with invalid output: {invalid_total}")
    if asr.misses:
        print(f"ASR ran for {asr.misses} queries not in the transcript cache")


if __name__ == "__main__":
    main()