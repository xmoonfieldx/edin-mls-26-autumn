"""Policies, fold isolation, persistent scoring and online paths without real APIs."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from extensions import retrieval_followups as policies
from extensions.cached_pointwise import CachedPointwise
from extensions.llm_reranking import LLMReranker

spec = importlib.util.spec_from_file_location("followup_evaluator", ROOT / "scripts/evaluate_retrieval_followups.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def feature_rows(count=10):
    return [{"query_id": f"q_{i}", "transcript": f"Question {i}", "candidates": [
        {"record_id": f"r_{j:02}", "dense": 1 - j * (.002 + i * .0001),
         "bm25": float((j + i) % 7), "cross": float((j * 3 + i) % 11)}
        for j in range(50)]} for i in range(count)]


class FakeData:
    def __init__(self, rows):
        self.rows = rows
        self.records = {c["record_id"]: "passage " + c["record_id"] for c in rows[0]["candidates"]}
        self.qrels = {r["query_id"]: {f"r_{(i * 3) % 20:02}": 1.} for i, r in enumerate(rows)}

    def audio_items(self, split):
        return [{"query_id": r["query_id"]} for r in self.rows]

    def load_qrels(self, split):
        return self.qrels

    def record_text(self, rid):
        return self.records[rid]


class FakeGenerator:
    def __init__(self, answer='{"score": 3}'):
        self.answer = answer
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return {"answer": self.answer}


class PolicyTests(unittest.TestCase):
    def test_query_folds_are_deterministic_and_balanced(self):
        rows = feature_rows()
        a = policies.folds(rows, 5)
        self.assertEqual(a, policies.folds(rows[::-1], 5))
        self.assertEqual([list(a.values()).count(i) for i in range(5)], [2] * 5)

    def test_constant_features_and_empty_training_contrasts_are_finite(self):
        row = feature_rows(1)[0]
        for c in row["candidates"]:
            c.update(dense=1, cross=1, bm25=1)
        model = policies.fit_fusion([row], {"q_0": {"not_in_pool": 1}}, 20, 1., 0)
        ranked, scores = policies.rank_fusion(row["candidates"], model)
        self.assertTrue(np.isfinite(list(scores.values())).all())
        self.assertEqual(ranked, [c["record_id"] for c in row["candidates"][:20]])

    def test_anchor_protects_two_but_can_change_remaining_top_five(self):
        row = feature_rows(1)[0]
        policy = {"depth": 20, "weights": [0, 1, 0, 0], "protect": 2}
        rank, _ = policies.rank_fusion(row["candidates"], policy)
        dense = [c["record_id"] for c in row["candidates"][:20]]
        self.assertEqual(rank[:2], dense[:2])
        self.assertNotEqual(rank[:5], dense[:5])

    def test_selection_is_twenty_unique_dense_order_ids_and_keeps_top_ten(self):
        row = feature_rows(1)[0]
        ids = policies.shortlist(row["candidates"])
        self.assertEqual(len(set(ids)), 20)
        self.assertEqual(ids[:10], [c["record_id"] for c in row["candidates"][:10]])
        self.assertEqual(ids, sorted(ids))
        self.assertTrue(set(ids) <= {c["record_id"] for c in row["candidates"][:40]})

    def test_missing_invalid_and_duplicate_features_fail(self):
        rows = feature_rows()
        rows[0]["candidates"][1]["dense"] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            policies.validate_features(rows)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            policies.unique_rows([{"query_id": "q"}, {"query_id": "q"}])

    def test_legacy_outcomes_do_not_fabricate_mrr(self):
        row = feature_rows(1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "point.log"
            path.write_text("[1/1] q_0 R@5=0.333 R@10=0.667 ELM calls=20\n")
            outcome = policies.pointwise_log_outcomes(path, [row], {"q_0": dict.fromkeys(["r_00", "r_01", "r_02"], 1)})
            self.assertEqual(outcome["q_0"]["pointwise"]["Recall@5"], 1 / 3)
            self.assertIsNone(outcome["q_0"]["pointwise"]["MRR@10"])


class CacheTests(unittest.TestCase):
    def test_persistent_cache_matches_existing_prompt_and_dense_ties(self):
        rows = feature_rows(1)
        data = FakeData(rows)
        ids = [c["record_id"] for c in rows[0]["candidates"][:20]]
        reference = FakeGenerator()
        original, _ = LLMReranker(data, reference).rerank("question", ids, "pointwise")
        with tempfile.TemporaryDirectory() as tmp:
            generator = FakeGenerator()
            path = Path(tmp) / "scores.jsonl"
            cache = CachedPointwise(data, path, "model", "endpoint", lambda: generator)
            rank, stats, details, error = cache.rerank("q_0", "question", ids)
            self.assertEqual(rank, original)
            self.assertEqual(generator.prompts, reference.prompts)
            self.assertEqual(stats["requests"], 20)
            self.assertEqual(len(details), 20)
            self.assertIsNone(error)
            restored = CachedPointwise(data, path, "model", "endpoint",
                                      lambda: self.fail("API factory must not run"), cache_only=True)
            rank, stats, _, _ = restored.rerank("q_0", "question", ids)
            self.assertEqual(stats["cache_hits"], 20)
            self.assertEqual(stats["requests"], 0)

    def test_query_model_text_and_prompt_label_invalidate_cache(self):
        data = FakeData(feature_rows(1))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.jsonl"
            generator = FakeGenerator()
            cache = CachedPointwise(data, path, "model", "endpoint", lambda: generator)
            cache.rerank("q_0", "one", ["r_00", "r_01"])
            with self.assertRaisesRegex(RuntimeError, "cache missing"):
                CachedPointwise(data, path, "other-model", "endpoint", lambda: generator, cache_only=True).rerank(
                    "q_0", "one", ["r_00"])
            cache.cache_only = True
            with self.assertRaises(RuntimeError):
                cache.rerank("q_0", "changed", ["r_00"])
            with self.assertRaises(RuntimeError):
                cache.rerank("q_0", "one", ["r_01", "r_00"])
            data.records["r_00"] = "changed passage"
            with self.assertRaises(RuntimeError):
                cache.rerank("q_0", "one", ["r_00"])

    def test_invalid_score_falls_back_without_caching_invalid_data(self):
        data = FakeData(feature_rows(1))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.jsonl"
            cache = CachedPointwise(data, path, "model", "endpoint", lambda: FakeGenerator('{"score": 99}'))
            rank, stats, _, error = cache.rerank("q_0", "question", ["r_00", "r_01"])
            self.assertEqual(rank, ["r_00", "r_01"])
            self.assertEqual(stats["invalid_outputs"], 1)
            self.assertTrue(error)
            self.assertFalse(path.exists())


class EvaluatorTests(unittest.TestCase):
    def offline_args(self, path, command):
        return SimpleNamespace(features=path, diagnostics=None, limit=None, command=command,
                               depth=20, folds=5, seed=17, feature_set="dense", max_call_fraction=.5)

    def test_outer_fusion_predictions_ignore_held_out_labels(self):
        rows = feature_rows()
        data = FakeData(rows)
        assignment = policies.folds(rows, 5)
        heldout = {q for q, f in assignment.items() if f == 0}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows))
            args = self.offline_args(path, "fusion")
            before, _, _ = script.offline(args, data)
            for q in heldout:
                data.qrels[q] = {"r_49": 1}
            after, _, _ = script.offline(args, data)
            for a, b in zip(before, after):
                if a["query_id"] in heldout:
                    self.assertEqual(a["policy"], b["policy"])
                    self.assertEqual(a["methods"]["fusion"]["ranked_ids"], b["methods"]["fusion"]["ranked_ids"])

    def test_outer_gate_predictions_ignore_held_out_pointwise_outcomes(self):
        rows = feature_rows()
        data = FakeData(rows)
        heldout = {q for q, f in policies.folds(rows, 5).items() if f == 0}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows))
            log = Path(tmp) / "point.log"
            log.write_text("\n".join(f"{r['query_id']} R@5=1.000 R@10=1.000" for r in rows))
            args = self.offline_args(path, "gate")
            args.pointwise_log = log
            before, _, _ = script.offline(args, data)
            log.write_text("\n".join(f"{r['query_id']} R@5={0 if r['query_id'] in heldout else 1}.000 R@10=1.000" for r in rows))
            after, _, _ = script.offline(args, data)
            for a, b in zip(before, after):
                if a["query_id"] in heldout:
                    self.assertEqual(a["policy"], b["policy"])
                    self.assertEqual(a["call_pointwise"], b["call_pointwise"])

    def test_missing_transcript_fails_without_starting_any_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "transcripts.jsonl"
            path.write_text(json.dumps({"query_id": "q_0", "transcript": "present"}))
            with self.assertRaisesRegex(ValueError, "q_1"):
                script.read_transcripts(path, ["q_0", "q_1"])

    def test_gpu_online_paths_use_fake_models_and_no_external_requests(self):
        rows = feature_rows(5)
        by_id = {r["query_id"]: r for r in rows}
        data = FakeData(rows)

        class Runtime:
            index_seconds = 0

            def __init__(self, *args, **kwargs):
                pass

            def collect(self, qid, transcript, depth):
                return {**by_id[qid], "candidates": by_id[qid]["candidates"][:depth],
                        "cross_pair_scores": depth,
                        "timing_ms": {"dense": 1, "cross": 2, "bm25": 1, "feature_collection": 4}}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcripts = root / "transcripts.jsonl"
            transcripts.write_text("\n".join(json.dumps({"query_id": r["query_id"], "transcript": r["transcript"]}) for r in rows))
            selections = root / "selections.jsonl"
            selections.write_text("\n".join(json.dumps({**script.identity(r), "selected_ids": policies.shortlist(r["candidates"])}) for r in rows))
            args = SimpleNamespace(command="shortlist-online", shortlists=selections, limit=5,
                                   transcript_cache=transcripts, data_dir=root, device="cuda", batch_size=16,
                                   pointwise_cache=root / "pointwise.jsonl", cache_only=False, output=root / "online.jsonl")
            generator = FakeGenerator()
            with patch.object(script, "LocalFeatures", Runtime), patch("dotenv.load_dotenv"), \
                    patch("src.generator.ELMGenerator", return_value=generator), redirect_stdout(io.StringIO()):
                outputs, config, _ = script.online(args, data)
            self.assertEqual(len(outputs), 5)
            self.assertEqual(len(policies.load_jsonl(args.output)), 5)
            self.assertTrue(all(r["cross_pair_scores"] == 40 for r in outputs))
            for row in outputs:
                for method in ("pointwise_dense20", "pointwise_selected20"):
                    self.assertEqual(row["methods"][method]["api"]["logical_pair_scores"], 20)
                    self.assertEqual(len(row["methods"][method]["pointwise_details"]), 20)
            self.assertEqual(config["actual_requests_across_unique_scoring_invocations"], len(generator.prompts))
            self.assertLessEqual(len(generator.prompts), 150)  # top ten prompts shared per query

            fusion_file = root / "fusion.jsonl"
            fusion_file.write_text("\n".join(json.dumps({**script.identity(r), "fold": i % 5,
                "policy": {"depth": 20, "weights": [0, 0, 0, 1], "protect": 2}})
                for i, r in enumerate(rows)))
            args.policies = fusion_file
            args.output = root / "fusion-online.jsonl"
            with patch.object(script, "LocalFeatures", Runtime), redirect_stdout(io.StringIO()):
                fusion_rows, _, _ = script.fusion_gpu(args, data)
            self.assertEqual(len(fusion_rows), 5)
            self.assertTrue(all(len(r["methods"]["fusion"]["ranked_ids"]) == 20 for r in fusion_rows))
            self.assertTrue(all(r["methods"]["fusion"]["api"]["requests"] == 0 for r in fusion_rows))

            gate_file = root / "gate.jsonl"
            gate_file.write_text("\n".join(json.dumps({**script.identity(r), "fold": i % 5,
                "policy": {"kind": "always" if i % 2 else "never", "feature_set": "dense"}})
                for i, r in enumerate(rows)))
            args.command = "gate-online"
            args.policies = gate_file
            args.compare_pointwise = True
            args.output = root / "gate-online.jsonl"
            args.pointwise_cache = root / "gate-cache.jsonl"
            generator = FakeGenerator()
            with patch.object(script, "LocalFeatures", Runtime), patch("dotenv.load_dotenv"), \
                    patch("src.generator.ELMGenerator", return_value=generator), redirect_stdout(io.StringIO()):
                gate_rows, config, _ = script.online(args, data)
            self.assertEqual(config["actual_requests_across_unique_scoring_invocations"], 100)
            self.assertEqual(len(generator.prompts), 100)
            self.assertEqual(sum(r["methods"]["gated"]["api"]["requests"] for r in gate_rows), 40)
            self.assertTrue(all(r["methods"]["pointwise"]["api"]["logical_pair_scores"] == 20 for r in gate_rows))


if __name__ == "__main__":
    unittest.main()
