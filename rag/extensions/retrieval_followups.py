"""Cached-feature policies. Ranking functions never receive relevance labels."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from src.metrics import recall_at_k, mrr_at_k


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def unique_rows(rows):
    result = {}
    for row in rows:
        qid = row["query_id"]
        if qid in result:
            raise ValueError(f"Duplicate query ID: {qid}")
        result[qid] = row
    return result


def validate_features(rows, minimum=20):
    unique_rows(rows)
    for row in rows:
        candidates = row["candidates"]
        ids = [c["record_id"] for c in candidates]
        if len(ids) < minimum or len(set(ids)) != len(ids):
            raise ValueError(f"{row['query_id']}: need at least {minimum} unique candidates")
        for c in candidates:
            if not all(np.isfinite(c[k]) for k in ("dense", "cross", "bm25")):
                raise ValueError(f"{row['query_id']}: nonfinite feature")
        if any(a["dense"] < b["dense"] for a, b in zip(candidates, candidates[1:])):
            raise ValueError(f"{row['query_id']}: candidates must be in dense order")


def folds(rows, count=5, seed=17):
    if not 2 <= count <= len(rows):
        raise ValueError("Need at least two folds and at least one query per fold")
    ordered = sorted((r["query_id"] for r in rows), key=lambda q: fingerprint([seed, q]))
    return {qid: i % count for i, qid in enumerate(ordered)}


def design(candidates):
    """Within-query min/max scaling needs no fitted statistics or labels."""
    raw = np.asarray([[c[k] for k in ("dense", "cross", "bm25")] for c in candidates])
    span = np.ptp(raw, axis=0)
    scaled = (raw - raw.min(axis=0)) / np.where(span > 0, span, 1)
    prior = np.linspace(1, 0, len(candidates))
    return np.column_stack([scaled, prior])


def fit_fusion(rows, training_qrels, depth, ridge, protect):
    prior = np.asarray([0., 0., 0., 1.])
    lhs, rhs = ridge * np.eye(4), ridge * prior
    informative = 0
    for row in rows:
        cs = row["candidates"][:depth]
        x = design(cs)
        # Unjudged records are training contrasts, not verified negatives.
        positive = [i for i, c in enumerate(cs) if c["record_id"] in training_qrels[row["query_id"]]]
        other = [i for i, c in enumerate(cs) if c["record_id"] not in training_qrels[row["query_id"]]]
        if positive and other:
            differences = (x[positive, None, :] - x[None, other, :]).reshape(-1, 4)
            lhs += differences.T @ differences / len(differences)
            rhs += differences.mean(axis=0)
            informative += 1
    return {"weights": np.linalg.solve(lhs, rhs).tolist(), "ridge": ridge,
            "protect": protect, "depth": depth, "informative_training_queries": informative,
            "training_queries_sha256": fingerprint(sorted(r["query_id"] for r in rows))}


def rank_fusion(candidates, policy):
    candidates = candidates[:policy["depth"]]
    scores = design(candidates) @ np.asarray(policy["weights"])
    protect = policy["protect"]
    order = list(range(protect)) + sorted(range(protect, len(candidates)),
                                          key=lambda i: (-scores[i], i, candidates[i]["record_id"]))
    return [candidates[i]["record_id"] for i in order], {
        c["record_id"]: float(s) for c, s in zip(candidates, scores, strict=True)}


def metrics(ids, rel):
    return {"Recall@5": recall_at_k(ids, rel, 5), "Recall@10": recall_at_k(ids, rel, 10),
            "MRR@10": mrr_at_k(ids, rel, 10)}


def rank_report(ids, rel, dense_ids):
    result = {"ranked_ids": ids, "metrics": metrics(ids, rel),
              "candidate_recall": recall_at_k(ids, rel, len(ids))}
    for k in (5, 10):
        promoted = (set(ids[:k]) - set(dense_ids[:k])) & set(rel)
        dropped = (set(dense_ids[:k]) - set(ids[:k])) & set(rel)
        result[f"promoted_relevant@{k}"] = sorted(promoted)
        result[f"dropped_relevant@{k}"] = sorted(dropped)
        result[f"promoted_dense_rank@{k}"] = {
            rid: dense_ids.index(rid) + 1 if rid in dense_ids else None for rid in sorted(promoted)}
    return result


def select_fusion(rows, training_qrels, depth, seed):
    """Inner query CV selects strength/anchors; outer labels are not supplied."""
    options = [(ridge, protect) for ridge in (.1, 1., 10.) for protect in (0, 2)]
    if len(rows) < 2:
        return fit_fusion(rows, training_qrels, depth, 10., 0)
    assignment = folds(rows, min(3, len(rows)), seed)
    best = None
    for ridge, protect in options:
        scores = []
        for fold in sorted(set(assignment.values())):
            train = [r for r in rows if assignment[r["query_id"]] != fold]
            validation = [r for r in rows if assignment[r["query_id"]] == fold]
            policy = fit_fusion(train, {r["query_id"]: training_qrels[r["query_id"]] for r in train},
                                depth, ridge, protect)
            scores += [metrics(rank_fusion(r["candidates"], policy)[0], training_qrels[r["query_id"]])
                       for r in validation]
        key = (np.mean([s["Recall@5"] for s in scores]),
               np.mean([s["Recall@10"] for s in scores]), ridge, -protect)
        if best is None or key > best[0]:
            best = (key, ridge, protect)
    result = fit_fusion(rows, training_qrels, depth, best[1], best[2])
    result["inner_cv_recall5"] = float(best[0][0])
    return result


def query_features(candidates, feature_set="dense"):
    cs = candidates[:20]
    values = {"gap5_6": cs[4]["dense"] - cs[5]["dense"],
              "gap1_2": cs[0]["dense"] - cs[1]["dense"]}
    if feature_set == "all":
        dense5 = {c["record_id"] for c in cs[:5]}
        for name in ("cross", "bm25"):
            top = sorted(range(len(cs)), key=lambda i: (-cs[i][name], i))[:5]
            values[name + "_disagreement"] = 1 - len(dense5 & {cs[i]["record_id"] for i in top}) / 5
    return values


def gate_options(feature_set):
    names = ["gap5_6", "gap1_2"]
    if feature_set == "all":
        names += ["cross_disagreement", "bm25_disagreement"]
    return [{"kind": "never"}, {"kind": "always"}] + [
        {"kind": "threshold", "feature": name, "quantile": q, "direction": direction}
        for name in names for q in (.25, .5, .75) for direction in ("le", "ge")]


def fit_gate(rows, option, feature_set):
    policy = {**option, "feature_set": feature_set,
              "training_queries_sha256": fingerprint(sorted(r["query_id"] for r in rows))}
    if option["kind"] == "threshold":
        values = [query_features(r["candidates"], feature_set)[option["feature"]] for r in rows]
        policy["threshold"] = float(np.quantile(values, option["quantile"]))
    return policy


def call_pointwise(candidates, policy):
    if policy["kind"] != "threshold":
        return policy["kind"] == "always"
    value = query_features(candidates, policy["feature_set"])[policy["feature"]]
    return value <= policy["threshold"] if policy["direction"] == "le" else value >= policy["threshold"]


def select_gate(rows, outcomes, feature_set, call_fraction, seed):
    if len(rows) < 2:
        return fit_gate(rows, {"kind": "never"}, feature_set)
    assignment = folds(rows, min(3, len(rows)), seed)
    best = None
    for option in gate_options(feature_set):
        selected = []
        scores = []
        for fold in sorted(set(assignment.values())):
            train = [r for r in rows if assignment[r["query_id"]] != fold]
            policy = fit_gate(train, option, feature_set)
            for row in rows:
                if assignment[row["query_id"]] != fold:
                    continue
                use = call_pointwise(row["candidates"], policy)
                selected.append(use)
                scores.append(outcomes[row["query_id"]]["pointwise" if use else "dense"])
        rate = float(np.mean(selected))
        if rate > call_fraction + 1e-12:
            continue
        key = (np.mean([s["Recall@5"] for s in scores]),
               np.mean([s["Recall@10"] for s in scores]), -rate)
        if best is None or key > best[0]:
            best = (key, option)
    policy = fit_gate(rows, best[1], feature_set)
    policy["inner_cv_recall5"] = float(best[0][0])
    return policy


def pointwise_log_outcomes(path, rows, qrels):
    text = Path(path).read_text(encoding="utf-8")
    matches = re.findall(r"(q_\w+) R@5=([\d.]+) R@10=([\d.]+)", text)
    by_id = unique_rows([{"query_id": q, "r5": float(a), "r10": float(b)} for q, a, b in matches])
    outcomes = {}
    for row in rows:
        qid = row["query_id"]
        if qid not in by_id:
            raise ValueError(f"Pointwise log missing {qid}")
        n = len(qrels[qid])
        pointwise = {"MRR@10": None}
        for k in (5, 10):
            logged = by_id[qid][f"r{k}"]
            hits = round(logged * n)
            if abs(logged - hits / n) > .000501 or hits > min(k, n):
                raise ValueError(f"Invalid rounded recall for {qid}")
            pointwise[f"Recall@{k}"] = hits / n
        outcomes[qid] = {"dense": metrics([c["record_id"] for c in row["candidates"][:20]], qrels[qid]),
                         "pointwise": pointwise}
    return outcomes


def shortlist(candidates):
    if len(candidates) < 40:
        raise ValueError("Deeper selection requires dense top 40")
    tail = sorted(enumerate(candidates[10:40], start=10), key=lambda pair: (-pair[1]["cross"], pair[0]))
    chosen = {c["record_id"] for c in candidates[:10]} | {c["record_id"] for _, c in tail[:10]}
    # Preserve original dense order, including for ordinal pointwise score ties.
    ids = [c["record_id"] for c in candidates[:40] if c["record_id"] in chosen]
    assert len(ids) == len(set(ids)) == 20
    return ids


def candidate_report(ids, rel, dense20):
    hits = len(set(ids) & set(rel))
    return {"candidate_ids": ids, "candidate_recall@20": hits / len(rel),
            "oracle_recall@5": min(5, hits) / len(rel),
            "gained_relevant_ids": sorted((set(ids) - set(dense20)) & set(rel)),
            "lost_relevant_ids": sorted((set(dense20) - set(ids)) & set(rel))}


def result_summary(rows, config):
    result = {"queries": len(rows), "config": config, "methods": {}}
    for method in rows[0]["methods"]:
        records = [r["methods"][method] for r in rows]
        item = {}
        for key in ("candidate_recall@20", "oracle_recall@5", "candidate_recall"):
            if key in records[0]:
                item[key] = float(np.mean([r[key] for r in records]))
        if "metrics" in records[0]:
            item["metrics"] = {}
            for key in ("Recall@5", "Recall@10", "MRR@10"):
                values = [r["metrics"].get(key) for r in records]
                item["metrics"][key] = None if any(v is None for v in values) else float(np.mean(values))
            for control in ("dense", "cross"):
                if control not in rows[0]["methods"]:
                    continue
                paired = {}
                for key in ("Recall@5", "Recall@10"):
                    deltas = np.asarray([r["methods"][method]["metrics"][key] -
                                         r["methods"][control]["metrics"][key] for r in rows])
                    rng = np.random.default_rng(17)
                    samples = rng.choice(deltas, (2000, len(deltas)), replace=True).mean(axis=1)
                    paired[key] = {"mean_delta": float(deltas.mean()), "wins": int((deltas > 1e-12).sum()),
                                   "losses": int((deltas < -1e-12).sum()), "ties": int((abs(deltas) <= 1e-12).sum()),
                                   "paired_bootstrap_95pct": np.quantile(samples, [.025, .975]).tolist()}
                item["paired_vs_" + control] = paired
        if "api" in records[0]:
            item["api"] = {key: sum(r["api"][key] for r in records)
                           for key in ("requests", "cache_hits", "logical_pair_scores", "invalid_outputs", "prompt_chars")}
            item["median_measured_path_ms"] = float(np.median([r["path_ms"] for r in records]))
        for key in ("gained_relevant_ids", "lost_relevant_ids"):
            if key in records[0]:
                item[key.replace("_ids", "_count")] = sum(len(r[key]) for r in records)
        result["methods"][method] = item
    if "call_pointwise" in rows[0]:
        count = sum(r["call_pointwise"] for r in rows)
        result.update({"selected_queries": count, "selected_fraction": count / len(rows),
                       "counterfactual_pointwise_calls": 20 * count,
                       "always_pointwise_calls": 20 * len(rows)})
    return result
