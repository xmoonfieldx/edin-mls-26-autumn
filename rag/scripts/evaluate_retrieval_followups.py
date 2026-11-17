"""Offline practice pilots and explicit GPU/ELM pilots for retrieval experiments 1–3."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extensions.retrieval_followups import (
    candidate_report, call_pointwise, fingerprint, folds, load_jsonl, metrics,
    pointwise_log_outcomes, rank_fusion, rank_report, result_summary,
    select_fusion, select_gate, shortlist, unique_rows, validate_features,
)
from src.data import CourseData


def read_transcripts(path, ids):
    if path is None or not path.is_file():
        raise ValueError("A complete --transcript-cache is required; ASR fallback is disabled")
    transcripts = {}
    for row in load_jsonl(path):
        qid, text = row["query_id"], row["transcript"]
        if qid in transcripts and transcripts[qid] != text:
            raise ValueError(f"Conflicting cached transcripts for {qid}")
        transcripts[qid] = text
    for qid in ids:
        if not isinstance(transcripts.get(qid), str) or not transcripts[qid].strip():
            raise ValueError(f"Missing nonempty cached transcript for {qid}; no ASR fallback")
    return transcripts


def load_feature_rows(args, data):
    rows = load_jsonl(args.features)
    validate_features(rows, 40 if args.command == "shortlist" else args.depth)
    allowed = {r["query_id"] for r in data.audio_items("practice")}
    if any(r["query_id"] not in allowed for r in rows):
        raise ValueError("Offline tuning accepts practice queries only; dev must remain held out")
    if args.limit:
        rows = rows[:args.limit]
    if args.diagnostics:
        diagnostic = unique_rows(load_jsonl(args.diagnostics))
        for row in rows:
            old = diagnostic.get(row["query_id"])
            ids = [c["record_id"] for c in row["candidates"]]
            if old is None or ids != [d["record_id"] for d in old["retrieved_documents"][:len(ids)]]:
                raise ValueError(f"{row['query_id']}: features do not match diagnostic dense IDs")
            if "transcript" in row and row["transcript"] != old["transcript"]:
                raise ValueError(f"{row['query_id']}: diagnostic transcript mismatch")
            row["transcript"] = old["transcript"]
    for row in rows:
        if any(c["record_id"] not in data.records for c in row["candidates"]):
            raise ValueError(f"{row['query_id']}: unknown corpus passage ID")
    return rows


def identity(row):
    return {"query_id": row["query_id"],
            "transcript_sha256": fingerprint(row["transcript"]) if "transcript" in row else None,
            "dense_candidate_ids": [c["record_id"] for c in row["candidates"]]}


def offline(args, data):
    rows = load_feature_rows(args, data)
    qrels = data.load_qrels("practice")
    if not rows or any(not qrels.get(r["query_id"]) for r in rows):
        raise ValueError("All selected queries need nonempty practice qrels")
    config = {"experiment": args.command, "mode": "offline", "split": "practice",
              "features": str(args.features), "features_sha256": fingerprint(rows),
              "device": "cpu (cached features; no inference)", "new_api_calls": 0,
              "depth": args.depth, "seed": args.seed,
              "legacy_inputs_without_transcript_hash": any("transcript" not in r for r in rows)}
    if args.command == "shortlist":
        outputs = []
        for row in rows:
            dense = [c["record_id"] for c in row["candidates"][:20]]
            selected = shortlist(row["candidates"])
            rel = qrels[row["query_id"]]
            outputs.append({**identity(row), "selected_ids": selected,
                            "selection_policy": "dense10_plus_cross_top10_from_dense11_40",
                            "local_pair_score_budget": 40, "pointwise_pair_budget_per_method": 20,
                            "methods": {"dense20": candidate_report(dense, rel, dense),
                                        "selected20": candidate_report(selected, rel, dense)}})
        return outputs, config, None

    training_started = perf_counter()
    assignment = folds(rows, args.folds, args.seed)
    config["outer_folds"] = args.folds
    config["inner_folds"] = "up to 3 query folds within each outer training set"
    config["evaluation"] = "out-of-fold; hyperparameters selected on inner training folds only"
    outcomes = None
    if args.command == "gate":
        outcomes = pointwise_log_outcomes(args.pointwise_log, rows, qrels)
        config.update({"pointwise_log": str(args.pointwise_log), "feature_set": args.feature_set,
                       "inner_cv_max_call_fraction": args.max_call_fraction,
                       "legacy_log_assumption": "The metric-only log must come from the same transcripts/candidates. "
                       "It has no hashes or passage rankings; MRR and exact hit movements cannot be recovered. "
                       "The call-fraction target is selected on inner folds, not enforced using outer labels."})
    policies = {}
    for fold in range(args.folds):
        train = [r for r in rows if assignment[r["query_id"]] != fold]
        train_rel = {r["query_id"]: qrels[r["query_id"]] for r in train}
        if args.command == "fusion":
            policies[fold] = select_fusion(train, train_rel, args.depth, args.seed)
        else:
            policies[fold] = select_gate(train, {r["query_id"]: outcomes[r["query_id"]] for r in train},
                                          args.feature_set, args.max_call_fraction, args.seed)
    outputs = []
    for row in rows:
        qid = row["query_id"]
        policy = policies[assignment[qid]]
        dense = [c["record_id"] for c in row["candidates"][:args.depth]]
        rel = qrels[qid]
        started = perf_counter()
        if args.command == "fusion":
            ranked, scores = rank_fusion(row["candidates"], policy)
            policy_ms = (perf_counter() - started) * 1000
            cross = sorted(row["candidates"][:args.depth], key=lambda c: -c["cross"])
            methods = {"dense": rank_report(dense, rel, dense),
                       "cross": rank_report([c["record_id"] for c in cross], rel, dense),
                       "fusion": {**rank_report(ranked, rel, dense), "scores": scores}}
            extra = {"features": row["candidates"][:args.depth], "offline_policy_ms": policy_ms}
        else:
            use = call_pointwise(row["candidates"], policy)
            policy_ms = (perf_counter() - started) * 1000
            methods = {name: {"metrics": outcomes[qid][name]} for name in ("dense", "pointwise")}
            methods["gated"] = {"metrics": outcomes[qid]["pointwise" if use else "dense"]}
            extra = {"call_pointwise": use, "offline_policy_ms": policy_ms,
                     "query_features_input": row["candidates"][:20], "rankings_available": False}
        outputs.append({**identity(row), "fold": assignment[qid], "policy": policy,
                        **extra, "methods": methods})
    if args.command == "fusion":
        final = select_fusion(rows, {r["query_id"]: qrels[r["query_id"]] for r in rows}, args.depth, args.seed)
    else:
        final = select_gate(rows, outcomes, args.feature_set, args.max_call_fraction, args.seed)
    config["offline_cv_and_final_fit_seconds"] = perf_counter() - training_started
    artifact = {"experiment": args.command, "policy": final,
                "status": "fit on all selected practice queries; NOT an independently evaluated model",
                "training_query_ids": [r["query_id"] for r in rows], "config": config}
    return outputs, config, artifact


class LocalFeatures:
    def __init__(self, data, data_dir, device, batch_size, need_cross=True, need_bm25=True):
        from src.encoder import LocalEncoder
        from src.retriever import ExactDenseRetriever
        from extensions.lexical_retrieval import BM25Retriever
        self.encoder = LocalEncoder(ROOT / "models/enc_a", device=device)
        self.retriever = ExactDenseRetriever(data_dir / "embeddings/corpus_enc_a.npy",
                                             data_dir / "embeddings/corpus_enc_a_ids.json")
        if set(self.retriever.record_ids) != set(data.records):
            raise ValueError("Embedding IDs do not match corpus")
        self.cross = None
        if need_cross:
            from extensions.cross_encoder_reranking import CrossEncoderReranker
            self.cross = CrossEncoderReranker(data, device=device, batch_size=batch_size)
        started = perf_counter()
        self.bm25 = BM25Retriever(data.records) if need_bm25 else None
        self.index_seconds = perf_counter() - started if need_bm25 else 0
        self.sync = lambda: None
        if device.startswith("cuda"):
            import torch
            self.sync = lambda: torch.cuda.synchronize(device)
        elif device == "mps":
            import torch
            self.sync = torch.mps.synchronize

    def collect(self, qid, transcript, depth):
        self.sync()
        started = perf_counter()
        vector = self.encoder.encode(transcript)
        dense = self.retriever.retrieve(vector, k=len(self.retriever.record_ids))
        ordered = sorted(zip(dense["record_ids"], dense["scores"], strict=True),
                         key=lambda pair: (-pair[1], pair[0]))[:depth]
        ids = [rid for rid, _ in ordered]
        if len(ids) != depth:
            raise ValueError(f"Need at least {depth} corpus passages")
        self.sync()
        dense_finished = perf_counter()
        lexical = self.bm25.score_candidates(transcript, ids) if self.bm25 else dict.fromkeys(ids, 0.)
        lexical_finished = perf_counter()
        cross = self.cross.score_candidates(transcript, ids) if self.cross else dict.fromkeys(ids, 0.)
        self.sync()
        finished = perf_counter()
        return {"query_id": qid, "transcript": transcript,
                "candidates": [{"record_id": rid, "dense": score, "bm25": lexical[rid], "cross": cross[rid]}
                               for rid, score in ordered],
                "timing_ms": {"dense": (dense_finished - started) * 1000,
                              "bm25": (lexical_finished - dense_finished) * 1000,
                              "cross": (finished - lexical_finished) * 1000,
                              "feature_collection": (finished - started) * 1000},
                "cross_pair_scores": depth if self.cross else 0}


def feature_export(args, data):
    items = data.audio_items("practice")
    if args.limit:
        items = items[:args.limit]
    transcripts = read_transcripts(args.transcript_cache, [r["query_id"] for r in items])
    runtime = LocalFeatures(data, args.data_dir, args.device, args.batch_size)
    outputs = []
    for i, item in enumerate(items):
        qid = item["query_id"]
        row = runtime.collect(qid, transcripts[qid], 50)
        outputs.append(row)
        # Save completed queries immediately; interrupted exports remain inspectable.
        append_output(args.output, row)
        print(f"[{i + 1}/{len(items)}] {qid} exported 50 local score triples", flush=True)
    return outputs, {"experiment": "feature_export", "device": args.device,
                     "batch_size": args.batch_size, "cross_encoder": "cross-encoder/ms-marco-MiniLM-L6-v2",
                     "bm25_index_seconds": runtime.index_seconds, "new_api_calls": 0}, None


def empty_api():
    return dict.fromkeys(("requests", "prompt_chars", "invalid_outputs", "cache_hits", "logical_pair_scores"), 0)


def score_pointwise(scorer, qid, transcript, ids, features):
    ranked, stats, details, error = scorer.rerank(qid, transcript, ids)
    source = {c["record_id"]: (i, c) for i, c in enumerate(features["candidates"], start=1)}
    final_rank = {rid: rank for rank, rid in enumerate(ranked, start=1)}
    details = [{**entry, "dense_rank": source[entry["record_id"]][0],
                "dense_score": source[entry["record_id"]][1]["dense"],
                "cross_score": source[entry["record_id"]][1]["cross"],
                "bm25_score": source[entry["record_id"]][1]["bm25"],
                "final_rank": final_rank[entry["record_id"]]} for entry in details]
    return ranked, stats, details, error


def fusion_gpu(args, data):
    source = unique_rows(load_jsonl(args.policies))
    items = [r for r in data.audio_items("practice") if r["query_id"] in source]
    if len(items) != len(source):
        raise ValueError("Online fusion requires unique practice query IDs")
    if args.limit:
        items = items[:args.limit]
    if any("fold" not in source[r["query_id"]] or "weights" not in source[r["query_id"]].get("policy", {})
           for r in items):
        raise ValueError("--policies must be the per-query out-of-fold fusion JSONL")
    transcripts = read_transcripts(args.transcript_cache, [r["query_id"] for r in items])
    runtime = LocalFeatures(data, args.data_dir, args.device, args.batch_size)
    qrels = data.load_qrels("practice")
    outputs = []
    for i, item in enumerate(items):
        qid = item["query_id"]
        old = source[qid]
        transcript = transcripts[qid]
        if old.get("transcript_sha256") and old["transcript_sha256"] != fingerprint(transcript):
            raise ValueError(f"{qid}: transcript changed since offline fusion")
        row = runtime.collect(qid, transcript, 20)
        ids = [c["record_id"] for c in row["candidates"]]
        if ids != old["dense_candidate_ids"][:20]:
            raise ValueError(f"{qid}: dense IDs changed; regenerate matched GPU features and policies")
        started = perf_counter()
        ranked, scores = rank_fusion(row["candidates"], old["policy"])
        policy_ms = (perf_counter() - started) * 1000
        cross = sorted(row["candidates"], key=lambda c: -c["cross"])
        times = row["timing_ms"]
        methods = {"dense": {**rank_report(ids, qrels[qid], ids), "path_ms": times["dense"], "api": empty_api()},
                   "cross": {**rank_report([c["record_id"] for c in cross], qrels[qid], ids),
                             "path_ms": times["dense"] + times["cross"], "api": empty_api()},
                   "fusion": {**rank_report(ranked, qrels[qid], ids), "scores": scores,
                              "path_ms": times["feature_collection"] + policy_ms, "api": empty_api()}}
        output = {**identity(row), "fold": old["fold"], "policy": old["policy"],
                  "local_features": row, "methods": methods}
        outputs.append(output)
        append_output(args.output, output)
        print(f"[{i + 1}/{len(items)}] {qid} fusion R5={methods['fusion']['metrics']['Recall@5']:.3f}", flush=True)
    return outputs, {"experiment": "fusion-online", "device": args.device, "batch_size": args.batch_size,
                     "bm25_index_seconds": runtime.index_seconds, "new_api_calls": 0,
                     "timing": "Shared measured GPU stages: dense only for dense control; dense+CE for cross; "
                     "dense+BM25+CE+policy for fusion. Excludes model/index loading, labels and output."}, None


def online(args, data):
    from dotenv import load_dotenv
    from extensions.cached_pointwise import CachedPointwise
    source = unique_rows(load_jsonl(args.policies if args.command == "gate-online" else args.shortlists))
    items = [r for r in data.audio_items("practice") if r["query_id"] in source]
    if len(items) != len(source):
        raise ValueError("Online pilots accept complete, unique practice query IDs only")
    if args.limit:
        items = items[:args.limit]
    transcripts = read_transcripts(args.transcript_cache, [r["query_id"] for r in items])
    is_gate = args.command == "gate-online"
    if is_gate and any("policy" not in source[r["query_id"]] or "feature_set" not in source[r["query_id"]]["policy"]
                       for r in items):
        raise ValueError("--policies must be the per-query out-of-fold gate JSONL")
    need_all = not is_gate or any(source[r["query_id"]]["policy"]["feature_set"] == "all" for r in items)
    runtime = LocalFeatures(data, args.data_dir, args.device, args.batch_size,
                            need_cross=need_all, need_bm25=is_gate and need_all)
    load_dotenv(ROOT / ".env")
    model = os.getenv("ELM_MODEL", "utter-project/EuroLLM-22B-Instruct-2512")
    endpoint = os.getenv("ELM_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://elm.edina.ac.uk/api/v1"))

    def generator_factory():
        from src.generator import ELMGenerator
        return ELMGenerator()

    scorer = CachedPointwise(data, args.pointwise_cache, model, endpoint, generator_factory,
                             cache_only=args.cache_only)
    qrels = data.load_qrels("practice")
    outputs = []
    actual_requests = 0
    for i, item in enumerate(items):
        qid = item["query_id"]
        old = source[qid]
        transcript = transcripts[qid]
        if old.get("transcript_sha256") and old["transcript_sha256"] != fingerprint(transcript):
            raise ValueError(f"{qid}: transcript changed since offline policy/selection")
        features = runtime.collect(qid, transcript, 20 if is_gate else 40)
        ids = [c["record_id"] for c in features["candidates"]]
        compare_depth = 20 if is_gate else 40
        if ids != old["dense_candidate_ids"][:compare_depth]:
            raise ValueError(f"{qid}: dense IDs changed; regenerate matched GPU features and offline outputs")
        shared_ms = features["timing_ms"]["feature_collection"]
        dense20 = ids[:20]
        methods = {"dense": {**rank_report(dense20, qrels[qid], ids),
                              "api": empty_api(), "path_ms": shared_ms}}
        extra = {}
        if is_gate:
            started = perf_counter()
            use = call_pointwise(features["candidates"], old["policy"])
            if use:
                ranked, stats, details, error = score_pointwise(scorer, qid, transcript, dense20, features)
                actual_requests += stats["requests"]
            else:
                ranked, stats, details, error = dense20, empty_api(), [], None
            methods["gated"] = {**rank_report(ranked, qrels[qid], ids), "api": stats,
                                "path_ms": shared_ms + (perf_counter() - started) * 1000,
                                "pointwise_details": details, "fallback_error": error}
            extra = {"call_pointwise": use, "policy": old["policy"], "fold": old["fold"]}
            if args.compare_pointwise:
                if use:
                    methods["pointwise"] = {**methods["gated"], "shared_with_gated": True}
                else:
                    started = perf_counter()
                    rank, stats, details, error = score_pointwise(scorer, qid, transcript, dense20, features)
                    actual_requests += stats["requests"]
                    methods["pointwise"] = {**rank_report(rank, qrels[qid], ids), "api": stats,
                                             "path_ms": shared_ms + (perf_counter() - started) * 1000,
                                             "pointwise_details": details, "fallback_error": error}
        else:
            selected = shortlist(features["candidates"])
            extra = {"selected_ids": selected, "saved_selection_matches": selected == old["selected_ids"],
                     "local_pair_score_budget": 40, "pointwise_pair_budget_per_method": 20}
            # Both methods pay the measured complete 40-pair local scoring path.
            # Rotate API order; exact shared prompts can reuse persistent scores.
            order = ("pointwise_dense20", "pointwise_selected20")
            if i % 2:
                order = order[::-1]
            extra["scoring_order"] = order
            for method in order:
                candidates = dense20 if method == "pointwise_dense20" else selected
                started = perf_counter()
                ranked, stats, details, error = score_pointwise(scorer, qid, transcript, candidates, features)
                actual_requests += stats["requests"]
                methods[method] = {**rank_report(ranked, qrels[qid], ids), "api": stats,
                                   "path_ms": shared_ms + (perf_counter() - started) * 1000,
                                   "candidate_report": candidate_report(candidates, qrels[qid], dense20),
                                   "pointwise_details": details, "fallback_error": error}
                methods[method].update(candidate_report(candidates, qrels[qid], dense20))
        row = {**identity(features), "transcript": transcript, **extra, "methods": methods,
               "local_features": features, "cross_pair_scores": features["cross_pair_scores"]}
        append_output(args.output, row)
        outputs.append(row)
        print(f"[{i + 1}/{len(items)}] {qid} " + " ".join(
            f"{name}:R5={r['metrics']['Recall@5']:.3f},calls={r['api']['requests']}"
            for name, r in methods.items()), flush=True)
    config = {"experiment": args.command, "mode": "online", "device": args.device,
              "batch_size": args.batch_size, "pointwise_model": model, "endpoint": endpoint,
              "pointwise_cache": str(args.pointwise_cache), "cache_only": args.cache_only,
              "bm25_index_seconds": runtime.index_seconds,
              "timing": "Measured feature path plus method scoring. Model/index loading, evaluation and output excluded. "
              "Cache hits are included: do not interpret cached paths as comparable uncached API latency. "
              "requests counts logical calls; the SDK may retry internally. Shared control results are not additional requests.",
              "pointwise_budget_per_selected_query": 20,
              "actual_requests_across_unique_scoring_invocations": actual_requests}
    return outputs, config, None


def append_output(path, row):
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "fusion", "gate", "shortlist", "fusion-online", "gate-online", "shortlist-online"):
        sub = commands.add_parser(name)
        sub.add_argument("--data-dir", type=Path, default=ROOT / "data/rag_corpus")
        sub.add_argument("--output", type=Path, required=True, help="Fresh JSONL path; adjacent summary/policy files are saved")
        sub.add_argument("--limit", type=int)
        if name in ("fusion", "gate", "shortlist"):
            sub.add_argument("--features", type=Path, required=True)
            sub.add_argument("--diagnostics", type=Path, help="Verify dense IDs and attach transcript hashes from diagnostics")
            sub.add_argument("--depth", type=int, default=20, choices=[20], help="Keep the fusion/gate pool fixed at 20")
            sub.add_argument("--seed", type=int, default=17)
        if name in ("fusion", "gate"):
            sub.add_argument("--folds", type=int, default=5)
        if name == "gate":
            sub.add_argument("--pointwise-log", type=Path, required=True)
            sub.add_argument("--feature-set", choices=["dense", "all"], default="dense")
            sub.add_argument("--max-call-fraction", type=float, default=.5)
        if name in ("export", "fusion-online", "gate-online", "shortlist-online"):
            sub.add_argument("--transcript-cache", type=Path, required=True)
            sub.add_argument("--device", default="cpu")
            sub.add_argument("--batch-size", type=int, default=16)
        if name in ("gate-online", "shortlist-online"):
            sub.add_argument("--pointwise-cache", type=Path, required=True)
            sub.add_argument("--cache-only", action="store_true", help="Fail on missing scores instead of calling ELM")
        if name == "gate-online":
            sub.add_argument("--compare-pointwise", action="store_true", help="Also call pointwise on skipped queries for a matched control")
        if name in ("gate-online", "fusion-online"):
            sub.add_argument("--policies", type=Path, required=True, help="Per-query out-of-fold JSONL, not the all-practice policy")
        if name == "shortlist-online":
            sub.add_argument("--shortlists", type=Path, required=True)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if getattr(args, "batch_size", 1) < 1:
        parser.error("--batch-size must be positive")
    if not 0 <= getattr(args, "max_call_fraction", .5) <= 1:
        parser.error("--max-call-fraction must be between 0 and 1")
    summary_path = args.output.with_suffix(".summary.json")
    policy_path = args.output.with_suffix(".policy.json")
    if len({args.output, summary_path, policy_path}) != 3 or any(p.exists() for p in (args.output, summary_path, policy_path)):
        parser.error("Choose a fresh output path; existing outputs are preserved")
    if hasattr(args, "pointwise_cache") and args.pointwise_cache.resolve() in {
            p.resolve() for p in (args.output, summary_path, policy_path)}:
        parser.error("Pointwise score cache must be separate from evaluation output files")
    try:
        data = CourseData(args.data_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.command == "export":
            rows, config, policy = feature_export(args, data)
            summary = {"queries": len(rows), "config": config,
                       "cross_pair_scores": sum(r["cross_pair_scores"] for r in rows)}
        elif args.command.endswith("online"):
            rows, config, policy = fusion_gpu(args, data) if args.command == "fusion-online" else online(args, data)
            summary = result_summary(rows, config)
        else:
            rows, config, policy = offline(args, data)
            for row in rows:
                append_output(args.output, row)
            summary = result_summary(rows, config)
        with summary_path.open("x", encoding="utf-8") as f:
            f.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        if policy is not None:
            with policy_path.open("x", encoding="utf-8") as f:
                f.write(json.dumps(policy, indent=2, allow_nan=False) + "\n")
        print(json.dumps(summary, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
