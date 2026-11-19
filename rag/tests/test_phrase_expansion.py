"""Controlled tests of graph links, candidate provenance and evaluation accounting."""
import importlib.util
import json
from math import log
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_phrase_expansion.py"
spec = importlib.util.spec_from_file_location("evaluate_phrase_expansion", SCRIPT)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
from extensions.phrase_graph import PhraseGraphIndex
from extensions.lexical_retrieval import BM25Retriever


class PhraseGraphTests(unittest.TestCase):
    def test_df_counts_documents_and_filters_common_phrases(self):
        index = PhraseGraphIndex({
            "seed": "rare phrase rare phrase common words", "a": "rare phrase common words",
            "b": "common words", "c": "unrelated content",
        }, max_document_frequency=2)
        self.assertEqual(index.postings["rare phrase"], ["a", "seed"])
        self.assertNotIn("common words", index.postings)
        hits = index.expand(["seed"], {"seed"})
        self.assertEqual([h["record_id"] for h in hits], ["a"])
        rare_links = [link for link in hits[0]["links"] if link["phrase"] == "rare phrase"]
        self.assertEqual(len(rare_links), 1)
        self.assertAlmostEqual(rare_links[0]["contribution"], log(5 / 3))
        self.assertAlmostEqual(hits[0]["score"], sum(link["contribution"] for link in hits[0]["links"]))

    def test_links_sum_across_seeds_with_inverse_rank_and_rarity(self):
        index = PhraseGraphIndex({
            "s1": "red apple", "s2": "blue berry", "a": "red apple blue berry",
            "b": "red apple", "c": "unrelated content",
        })
        hits = {h["record_id"]: h for h in index.expand(["s1", "s2"], {"s1", "s2"})}
        self.assertAlmostEqual(hits["a"]["score"], log(6 / 4) + log(6 / 3) / 2)
        self.assertGreater(hits["a"]["score"], hits["b"]["score"])
        self.assertEqual({l["seed_id"] for l in hits["a"]["links"]}, {"s1", "s2"})

    def test_one_hop_only_and_id_tie_order(self):
        index = PhraseGraphIndex({"seed": "red apple", "bridge": "red apple blue berry",
                                  "remote": "blue berry", "a": "red apple"})
        hits = index.expand(["seed"], {"seed"})
        self.assertEqual([h["record_id"] for h in hits], ["a", "bridge"])
        self.assertNotIn("remote", [h["record_id"] for h in hits])


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.dense = [f"d{i:02}" for i in range(80)]

    def test_dedup_exclusion_budget_and_backfill_provenance(self):
        hits = [{"record_id": rid} for rid in ["d00", "d01", "new", "new", "d25"]]
        pool = experiment.build_pool(self.dense, hits, "phrase_graph")
        ids = [c["record_id"] for c in pool]
        self.assertEqual(len(ids), 40)
        self.assertEqual(len(set(ids)), 40)
        self.assertEqual(ids[:20], self.dense[:20])
        self.assertEqual(ids[20:22], ["new", "d25"])
        self.assertEqual(pool[20]["source"], "phrase_graph_expansion")
        self.assertEqual(pool[22]["source"], "dense_backfill")

    def test_bm25_search_fetches_past_prefix_overlap(self):
        bm25 = BM25Retriever({rid: "matching phrase" for rid in self.dense})
        result = experiment.measure_method("bm25", "matching", self.dense, 3,
                                           None, bm25, None)
        self.assertEqual(len(result["candidates"]), 40)
        self.assertEqual(len([c for c in result["candidates"]
                              if c["source"] == "bm25_expansion"]), 20)
        self.assertIsNone(result["reranked_ids"])

    def test_gains_losses_rescue_and_expansion_at_cutoffs(self):
        pool = experiment.build_pool(self.dense, [{"record_id": "new"}], "phrase_graph")
        ids = [c["record_id"] for c in pool]
        result = {"candidates": pool, "reranked_ids": ["new"] + [r for r in ids if r != "new"]}
        experiment.evaluate_pool(result, {"new": 1, "d39": 1}, self.dense)
        self.assertEqual(result["candidate_recall@40"], 0.5)
        self.assertEqual(result["gained_relevant_ids_vs_dense40"], ["new"])
        self.assertEqual(result["lost_relevant_ids_vs_dense40"], ["d39"])
        self.assertTrue(result["rescued_empty_dense20"])
        self.assertEqual(result["expansion_relevant_at_cutoff"]["5"], 1)
        self.assertEqual(result["gained_relevant_vs_dense40_at_cutoff"]["10"], 1)

    def test_exactly_40_ce_scores_per_method_and_total_timing(self):
        class Scorer:
            def __init__(self):
                self.calls = []

            def score_candidates(self, transcript, ids):
                self.calls.append(list(ids))
                return dict.fromkeys(ids, 1.0)

        scorer = Scorer()
        for method in experiment.METHODS:
            with patch.object(experiment, "perf_counter", side_effect=[0, .010, .012, .030]):
                result = experiment.measure_method(
                    method, "question", self.dense, 7,
                    type("Graph", (), {"expand": lambda *a, **k: []})(),
                    type("BM25", (), {"retrieve": lambda *a, **k: []})(), scorer)
            self.assertEqual(result["cross_encoder_score_count"], 40)
            self.assertAlmostEqual(result["timing_ms"]["total"], 37)
            self.assertEqual(result["reranked_ids"], sorted(self.dense[:40]))
        self.assertEqual([len(c) for c in scorer.calls], [40, 40, 40])

    def test_missing_and_conflicting_transcripts_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.jsonl"
            path.write_text(json.dumps({"query_id": "q1", "transcript": "cached"}) + "\n")
            with self.assertRaisesRegex(ValueError, "Missing.*q2"):
                experiment.load_transcripts(path, [{"query_id": "q2"}])
            with path.open("a") as f:
                f.write(json.dumps({"query_id": "q1", "transcript": "changed"}) + "\n")
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                experiment.load_transcripts(path, [{"query_id": "q1"}])

    def test_five_query_candidate_only_harness(self):
        records = {rid: ("rare phrase" if rid in ("d00", "d50") else "other text")
                   for rid in self.dense}
        graph = PhraseGraphIndex(records, max_document_frequency=2)
        bm25 = BM25Retriever(records)
        rows = []
        for i in range(5):
            methods = {}
            for method in experiment.METHODS:
                result = experiment.measure_method(method, "rare phrase", self.dense, 1,
                                                   graph, bm25, None)
                methods[method] = experiment.evaluate_pool(result, {"d50": 1}, self.dense)
                ids = [c["record_id"] for c in result["candidates"]]
                self.assertEqual(len(set(ids)), 40)
                if method != "dense":
                    self.assertTrue(set(self.dense[:20]) <= set(ids))
                    self.assertTrue(result["expansion_ids"])
                    self.assertTrue(result["dense_backfill_ids"])
            rows.append({"methods": methods})
        summary = experiment.summarize(rows, {"candidate_only": True}, {})
        self.assertEqual(summary["methods"]["dense"]["candidate_recall@40"], 0)
        self.assertEqual(summary["methods"]["phrase_graph"]["candidate_recall@40"], 1)
        self.assertEqual(summary["methods"]["phrase_graph"]["queries_rescued_from_empty_dense20"], 5)
        self.assertEqual(summary["methods"]["phrase_graph"]["cross_encoder_scores"], 0)
        json.dumps(rows, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
