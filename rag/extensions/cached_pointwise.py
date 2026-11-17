"""Persistent pointwise scores, using the existing ELM prompt and parser."""
from pathlib import Path
import json
from time import perf_counter

from extensions.llm_reranking import LLMReranker, POINTWISE_INSTRUCTION
from extensions.retrieval_followups import fingerprint, load_jsonl


class CachedPointwise:
    def __init__(self, data, cache_path, model, endpoint, generator_factory,
                 max_passage_chars=1600, cache_only=False):
        self.data = data
        self.path = Path(cache_path)
        self.model = model
        self.endpoint = endpoint
        self.factory = generator_factory
        self.backend = None
        self.max_chars = max_passage_chars
        self.cache_only = cache_only
        self.cache = {}
        if self.path.exists():
            for row in load_jsonl(self.path):
                if type(row.get("score")) is not int or not 0 <= row["score"] <= 4:
                    raise ValueError("Pointwise cache contains an invalid ordinal score")
                key = row["cache_key"]
                if key in self.cache and self.cache[key]["score"] != row["score"]:
                    raise ValueError("Conflicting pointwise cache entries")
                self.cache[key] = row

    def rerank(self, qid, transcript, candidate_ids):
        stats = {"requests": 0, "prompt_chars": 0, "invalid_outputs": 0,
                 "cache_hits": 0, "logical_pair_scores": 0, "logical_pair_budget": len(candidate_ids)}
        details = []
        try:
            for rank, rid in enumerate(candidate_ids, start=1):
                label = f"P{rank}"
                text = self.data.record_text(rid)[:self.max_chars]
                identity = {
                    "version": "existing-pointwise-v1", "query_id": qid,
                    "transcript_sha256": fingerprint(transcript), "record_id": rid,
                    "passage_sha256": fingerprint(text), "label": label,
                    "max_passage_chars": self.max_chars, "model": self.model,
                    "endpoint": self.endpoint, "temperature": .1,
                    "instruction_sha256": fingerprint(POINTWISE_INSTRUCTION),
                }
                key = fingerprint(identity)
                stats["logical_pair_scores"] += 1
                hit = key in self.cache
                if hit:
                    entry = self.cache[key]
                    stats["cache_hits"] += 1
                else:
                    if self.cache_only:
                        raise RuntimeError(f"Pointwise cache missing {qid}/{rid}; cache-only mode forbids API calls")
                    if self.backend is None:
                        self.backend = LLMReranker(self.data, self.factory(), self.max_chars)
                    started = perf_counter()
                    score = self.backend._ask(POINTWISE_INSTRUCTION, transcript, {label: text}, stats,
                                              strategy="pointwise")
                    if type(score) is not int or not 0 <= score <= 4:
                        raise ValueError(f"Invalid pointwise score for {rid}: {score!r}")
                    entry = {"cache_key": key, **identity, "score": score,
                             "request_ms": (perf_counter() - started) * 1000}
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
                        f.flush()
                    self.cache[key] = entry
                details.append({**entry, "cache_hit": hit, "input_rank": rank})
        except ValueError as exc:
            stats["invalid_outputs"] += 1
            return list(candidate_ids), stats, details, str(exc)
        scores = {entry["record_id"]: entry["score"] for entry in details}
        dense_rank = {rid: i for i, rid in enumerate(candidate_ids)}
        ranked = sorted(candidate_ids, key=lambda rid: (-scores[rid], dense_rank[rid], rid))
        return ranked, stats, details, None
