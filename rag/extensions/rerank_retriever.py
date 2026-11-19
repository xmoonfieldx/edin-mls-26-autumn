"""Wrappers that let RAGPipeline use cached transcripts and pointwise ELM reranking.

Nothing in src/ is modified. The wrappers sit between RAGPipeline and its parts:

  CachedASR           serve transcripts from the evaluate_rag.py transcript cache
  EncoderTap          remember the last transcript so the reranker can see it
  CachingGenerator    cache ELM reranking calls on disk, count hits and new calls
  RerankingRetriever  dense top-N -> pointwise rerank -> top-k, same output shape
"""
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    rows = []
    if path is not None and Path(path).exists():
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # half-written last line after a crash
    return rows


class CachedASR:
    """Return cached transcripts; load and run ASR only for queries not in the cache.

    The cache format matches evaluate_rag.py: {"query_id": ..., "transcript": ...}.
    Using one cache for every system keeps the transcripts identical.
    """

    def __init__(self, asr, path_to_qid, cache_path):
        self.asr = asr
        self.path_to_qid = path_to_qid
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = {r["query_id"]: r["transcript"] for r in read_jsonl(self.cache_path)}
        self.prepared = False
        self.misses = 0

    def transcribe(self, audio_path):
        qid = self.path_to_qid.get(str(audio_path))
        if qid in self.cache:
            return self.cache[qid]
        if not self.prepared:
            self.asr.prepare()
            self.prepared = True
        transcript = self.asr.transcribe(audio_path)
        self.misses += 1
        if qid is not None:
            self.cache[qid] = transcript
            if self.cache_path:
                append_jsonl(self.cache_path, {"query_id": qid, "transcript": transcript})
        return transcript


class EncoderTap:
    """Wrap the query encoder and remember the transcript it was last given."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.last = None

    def encode(self, text, *args, **kwargs):
        self.last = text
        return self.encoder.encode(text, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.encoder, name)


class CachingGenerator:
    """Cache ELM calls on disk, keyed by model name + full prompt text.

    The prompt contains the instruction, query and passage, so different prompt
    variants never share entries. Delete the cache file to force fresh calls.
    """

    def __init__(self, generator, cache_path):
        self.generator = generator
        self.cache_path = Path(cache_path)
        self.cache = {r["key"]: r["answer"] for r in read_jsonl(self.cache_path)}
        self.hits = 0
        self.new_calls = 0

    def generate(self, prompt):
        model = os.getenv("ELM_MODEL", "")
        key = hashlib.sha256((model + "\n" + prompt).encode("utf-8")).hexdigest()
        if key in self.cache:
            self.hits += 1
            return {"answer": self.cache[key], "latency_ms": 0.0}
        result = self.generator.generate(prompt)
        self.new_calls += 1
        self.cache[key] = result["answer"]
        append_jsonl(self.cache_path, {"key": key, "answer": result["answer"]})
        return result


class RerankingRetriever:
    """Dense top-`depth`, pointwise ELM rerank, then keep the top-k.

    Returns the same dict shape as ExactDenseRetriever.retrieve, with per-hit
    lists (record_ids, scores, ...) reordered together. Note: "scores" stay the
    dense scores of the passages; they are not the ELM scores. latency_ms covers
    dense search plus reranking; the parts are also returned separately.
    """

    def __init__(self, base, encoder_tap, reranker, depth, caching_generator):
        self.base = base
        self.tap = encoder_tap
        self.reranker = reranker
        self.depth = depth
        self.gen = caching_generator

    def retrieve(self, query_vector, k=5):
        if self.tap.last is None:
            raise RuntimeError("EncoderTap has not seen a transcript yet")
        ret = self.base.retrieve(query_vector, k=max(k, self.depth))
        ids = list(ret["record_ids"])

        hits0, calls0 = self.gen.hits, self.gen.new_calls
        started = perf_counter()
        reranked, stats = self.reranker.rerank(
            self.tap.last, ids[: self.depth], "pointwise"
        )
        rerank_ms = (perf_counter() - started) * 1000.0

        final = list(reranked) + ids[self.depth:]
        pos = {rid: i for i, rid in enumerate(ids)}
        order = [pos[rid] for rid in final[:k]]

        out = {}
        for key, val in ret.items():
            if isinstance(val, np.ndarray) and val.ndim >= 1 and len(val) == len(ids):
                out[key] = val[order]
            elif isinstance(val, (list, tuple)) and len(val) == len(ids):
                out[key] = [val[i] for i in order]
            else:
                out[key] = val
        out["record_ids"] = [ids[i] for i in order]

        dense_ms = float(ret.get("latency_ms", 0.0))
        out["dense_latency_ms"] = dense_ms
        out["rerank_ms"] = rerank_ms
        out["latency_ms"] = dense_ms + rerank_ms
        out["rerank_stats"] = stats
        out["rerank_cache_hits"] = self.gen.hits - hits0
        out["rerank_new_calls"] = self.gen.new_calls - calls0
        return out