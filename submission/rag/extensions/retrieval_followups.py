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
        query_id = row["query_id"]
        if query_id in result:
            raise ValueError(f"Duplicate query ID: {query_id}")
        result[query_id] = row
    return result


def validate_features(rows, minimum=20):
    unique_rows(rows)
    for row in rows:
        candidates = row["candidates"]
        candidate_ids = [candidate["record_id"] for candidate in candidates]
        if len(candidate_ids) < minimum or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError(f"{row['query_id']}: need at least {minimum} unique candidates")
        for candidate in candidates:
            if not all(np.isfinite(candidate[k]) for k in ("dense", "cross", "bm25")):
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
    raw_scores = np.asarray([[c[k] for k in ("dense", "cross", "bm25")] for c in candidates])
    score_ranges = np.ptp(raw_scores, axis=0)
    normalised_scores = (raw_scores - raw_scores.min(axis=0)) / np.where(
        score_ranges > 0, score_ranges, 1
    )
    dense_rank_prior = np.linspace(1, 0, len(candidates))
    return np.column_stack([normalised_scores, dense_rank_prior])


def fit_fusion(rows, training_qrels, depth, ridge, protect):
    dense_rank_prior = np.asarray([0., 0., 0., 1.])
    normal_matrix, target_vector = ridge * np.eye(4), ridge * dense_rank_prior
    informative_queries = 0
    for row in rows:
        candidates = row["candidates"][:depth]
        features = design(candidates)
        # Unjudged records are training contrasts, not verified negatives.
        positive_indices = [
            i for i, candidate in enumerate(candidates)
            if candidate["record_id"] in training_qrels[row["query_id"]]
        ]
        unjudged_indices = [
            i for i, candidate in enumerate(candidates)
            if candidate["record_id"] not in training_qrels[row["query_id"]]
        ]
        if not positive_indices or not unjudged_indices:
            continue
        differences = (
            features[positive_indices, None, :] - features[None, unjudged_indices, :]
        ).reshape(-1, 4)
        normal_matrix += differences.T @ differences / len(differences)
        target_vector += differences.mean(axis=0)
        informative_queries += 1
    return {"weights": np.linalg.solve(normal_matrix, target_vector).tolist(), "ridge": ridge,
            "protect": protect, "depth": depth, "informative_training_queries": informative_queries,
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
    fold_by_query = folds(rows, min(3, len(rows)), seed)
    best_option = None
    for ridge, protect in options:
        validation_metrics = []
        for fold in sorted(set(fold_by_query.values())):
            training_rows = [r for r in rows if fold_by_query[r["query_id"]] != fold]
            validation_rows = [r for r in rows if fold_by_query[r["query_id"]] == fold]
            fold_qrels = {row["query_id"]: training_qrels[row["query_id"]] for row in training_rows}
            policy = fit_fusion(training_rows, fold_qrels, depth, ridge, protect)
            validation_metrics += [
                metrics(rank_fusion(row["candidates"], policy)[0], training_qrels[row["query_id"]])
                for row in validation_rows
            ]
        selection_key = (
            np.mean([score["Recall@5"] for score in validation_metrics]),
            np.mean([score["Recall@10"] for score in validation_metrics]),
            ridge, -protect,
        )
        if best_option is None or selection_key > best_option[0]:
            best_option = (selection_key, ridge, protect)
    result = fit_fusion(rows, training_qrels, depth, best_option[1], best_option[2])
    result["inner_cv_recall5"] = float(best_option[0][0])
    return result


def query_features(candidates, feature_set="dense"):
    candidates_top20 = candidates[:20]
    values = {"gap5_6": candidates_top20[4]["dense"] - candidates_top20[5]["dense"],
              "gap1_2": candidates_top20[0]["dense"] - candidates_top20[1]["dense"]}
    if feature_set == "all":
        dense_top5 = {c["record_id"] for c in candidates_top20[:5]}
        for name in ("cross", "bm25"):
            top_indices = sorted(range(len(candidates_top20)), key=lambda i: (-candidates_top20[i][name], i))[:5]
            values[name + "_disagreement"] = 1 - len(dense_top5 & {candidates_top20[i]["record_id"] for i in top_indices}) / 5
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
    if policy["direction"] == "le":
        return value <= policy["threshold"]
    return value >= policy["threshold"]


def select_gate(rows, outcomes, feature_set, call_fraction, seed):
    if len(rows) < 2:
        return fit_gate(rows, {"kind": "never"}, feature_set)
    fold_by_query = folds(rows, min(3, len(rows)), seed)
    best_option = None
    for option in gate_options(feature_set):
        selection_flags = []
        validation_outcomes = []
        for fold in sorted(set(fold_by_query.values())):
            training_rows = [r for r in rows if fold_by_query[r["query_id"]] != fold]
            policy = fit_gate(training_rows, option, feature_set)
            validation_rows = [row for row in rows if fold_by_query[row["query_id"]] == fold]
            for row in validation_rows:
                use_pointwise = call_pointwise(row["candidates"], policy)
                selection_flags.append(use_pointwise)
                validation_outcomes.append(outcomes[row["query_id"]]["pointwise" if use_pointwise else "dense"])
        rate = float(np.mean(selection_flags))
        if rate > call_fraction + 1e-12:
            continue
        selection_key = (
            np.mean([score["Recall@5"] for score in validation_outcomes]),
            np.mean([score["Recall@10"] for score in validation_outcomes]),
            -rate,
        )
        if best_option is None or selection_key > best_option[0]:
            best_option = (selection_key, option)
    policy = fit_gate(rows, best_option[1], feature_set)
    policy["inner_cv_recall5"] = float(best_option[0][0])
    return policy


def pointwise_log_outcomes(path, rows, qrels):
    text = Path(path).read_text(encoding="utf-8")
    matches = re.findall(r"(q_\w+) R@5=([\d.]+) R@10=([\d.]+)", text)
    by_id = unique_rows([{"query_id": q, "r5": float(a), "r10": float(b)} for q, a, b in matches])
    outcomes = {}
    for row in rows:
        query_id = row["query_id"]
        if query_id not in by_id:
            raise ValueError(f"Pointwise log missing {query_id}")
        relevant_count = len(qrels[query_id])
        pointwise = {"MRR@10": None}
        for k in (5, 10):
            logged_recall = by_id[query_id][f"r{k}"]
            relevant_hits = round(logged_recall * relevant_count)
            if abs(logged_recall - relevant_hits / relevant_count) > .000501 or relevant_hits > min(k, relevant_count):
                raise ValueError(f"Invalid rounded recall for {query_id}")
            pointwise[f"Recall@{k}"] = relevant_hits / relevant_count
        outcomes[query_id] = {"dense": metrics([c["record_id"] for c in row["candidates"][:20]], qrels[query_id]),
                         "pointwise": pointwise}
    return outcomes


def shortlist(candidates):
    if len(candidates) < 40:
        raise ValueError("Deeper selection requires dense top 40")
    ranked_tail = sorted(enumerate(candidates[10:40], start=10), key=lambda pair: (-pair[1]["cross"], pair[0]))
    selected_ids = {c["record_id"] for c in candidates[:10]} | {c["record_id"] for _, c in ranked_tail[:10]}
    # Preserve original dense order, including for ordinal pointwise score ties.
    shortlist_ids = [c["record_id"] for c in candidates[:40] if c["record_id"] in selected_ids]
    assert len(shortlist_ids) == len(set(shortlist_ids)) == 20
    return shortlist_ids


def candidate_report(ids, rel, dense20):
    hits = len(set(ids) & set(rel))
    return {"candidate_ids": ids, "candidate_recall@20": hits / len(rel),
            "oracle_recall@5": min(5, hits) / len(rel),
            "gained_relevant_ids": sorted((set(ids) - set(dense20)) & set(rel)),
            "lost_relevant_ids": sorted((set(dense20) - set(ids)) & set(rel))}


def paired_comparison(deltas):
    """Summarise paired query differences with the original bootstrap draws."""
    rng = np.random.default_rng(17)
    samples = rng.choice(deltas, (2000, len(deltas)), replace=True).mean(axis=1)
    return {
        "mean_delta": float(deltas.mean()),
        "wins": int((deltas > 1e-12).sum()),
        "losses": int((deltas < -1e-12).sum()),
        "ties": int((abs(deltas) <= 1e-12).sum()),
        "paired_bootstrap_95pct": np.quantile(samples, [.025, .975]).tolist(),
    }


def result_summary(rows, config):
    result = {"queries": len(rows), "config": config, "methods": {}}
    for method in rows[0]["methods"]:
        method_rows = [r["methods"][method] for r in rows]
        method_summary = {}
        for key in ("candidate_recall@20", "oracle_recall@5", "candidate_recall"):
            if key in method_rows[0]:
                method_summary[key] = float(np.mean([r[key] for r in method_rows]))
        if "metrics" in method_rows[0]:
            method_summary["metrics"] = {}
            for key in ("Recall@5", "Recall@10", "MRR@10"):
                values = [r["metrics"].get(key) for r in method_rows]
                method_summary["metrics"][key] = None if any(v is None for v in values) else float(np.mean(values))
            for control in ("dense", "cross"):
                if control not in rows[0]["methods"]:
                    continue
                paired = {}
                for key in ("Recall@5", "Recall@10"):
                    deltas = np.asarray([r["methods"][method]["metrics"][key] -
                                         r["methods"][control]["metrics"][key] for r in rows])
                    paired[key] = paired_comparison(deltas)
                method_summary["paired_vs_" + control] = paired
        if "api" in method_rows[0]:
            method_summary["api"] = {key: sum(r["api"][key] for r in method_rows)
                           for key in ("requests", "cache_hits", "logical_pair_scores", "invalid_outputs", "prompt_chars")}
            method_summary["median_measured_path_ms"] = float(np.median([r["path_ms"] for r in method_rows]))
        for key in ("gained_relevant_ids", "lost_relevant_ids"):
            if key in method_rows[0]:
                method_summary[key.replace("_ids", "_count")] = sum(len(r[key]) for r in method_rows)
        result["methods"][method] = method_summary
    if "call_pointwise" in rows[0]:
        count = sum(r["call_pointwise"] for r in rows)
        result.update({"selected_queries": count, "selected_fraction": count / len(rows),
                       "counterfactual_pointwise_calls": 20 * count,
                       "always_pointwise_calls": 20 * len(rows)})
    return result
