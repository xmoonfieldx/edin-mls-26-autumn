"""Exercise diagnostic conclusions without course assets or model dependencies."""
import importlib.util
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/diagnose_retrieval.py"
spec = importlib.util.spec_from_file_location("diagnose_retrieval", SCRIPT)
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


class FakeData:
    def record_text(self, rid):
        return f"Text of {rid}"


def diagnose(relevant, ranking):
    return diagnostics.diagnose("q_test", "A question", {
        "record_ids": ranking, "scores": [1 / i for i in range(1, len(ranking) + 1)],
        "latency_ms": 1.0,
    }, dict.fromkeys(relevant, 1.0), FakeData())


class RetrievalDiagnosticsTests(unittest.TestCase):
    def test_perfect_recall10_can_still_need_reranking_for_recall5(self):
        row = diagnose(["a", "b"], ["a", "x", "y", "z", "w", "b"])
        self.assertFalse(row["r10_candidate"])
        self.assertTrue(row["r5_candidate"])
        self.assertEqual(row["potential_reranking_gain_at5"], 0.5)
        self.assertEqual(row["opportunity"], "ranking")
        self.assertEqual(row["relevant_documents"][1]["rank"], 6)

    def test_missing_relevant_documents_need_candidate_coverage(self):
        row = diagnose(["a", "b"], ["a", "x"])
        self.assertEqual(row["opportunity"], "candidate_coverage")
        self.assertEqual(row["potential_reranking_gain_at5"], 0)
        self.assertEqual(row["candidate_coverage_gap_at5"], 0.5)
        self.assertIsNone(row["relevant_documents"][1]["rank"])

    def test_ranking_and_coverage_can_both_need_improvement(self):
        row = diagnose(["a", "b", "c"], ["a", "x", "y", "z", "w", "b"])
        self.assertEqual(row["opportunity"], "ranking_and_candidate_coverage")
        self.assertAlmostEqual(row["oracle_recall5_from_candidates"], 2 / 3)

    def test_more_than_five_relevant_records_limit_recall5(self):
        ranking = list("abcdef")
        row = diagnose(ranking, ranking)
        self.assertAlmostEqual(row["max_possible_recall5"], 5 / 6)
        self.assertEqual(row["opportunity"], "at_recall5_ceiling")
        self.assertEqual(row["potential_reranking_gain_at5"], 0)

    def test_missing_records_need_not_reduce_recall5_ceiling(self):
        row = diagnose(list("abcdefgh"), list("abcde"))
        self.assertEqual(row["opportunity"], "at_recall5_ceiling")
        self.assertEqual(row["relevant_missing_from_candidates"], 3)
        self.assertEqual(row["candidate_coverage_gap_at5"], 0)

    def test_report_preserves_aggregate_and_flags_both_cutoffs(self):
        rows = [diagnose(["a", "b"], ["a", "x", "y", "z", "w", "b"]),
                diagnose(["a", "b"], ["a", "x"])]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summary = diagnostics.save_reports(out, rows, "practice", 100,
                                               out / "cache.jsonl", True, 20)
            self.assertEqual(summary["r10_candidates"], 1)
            self.assertEqual(summary["r10_perfect_but_r5_imperfect"], 1)
            self.assertEqual(summary["metrics"]["Recall@5"], 0.5)
            self.assertEqual(json.loads((out / "summary.json").read_text()), summary)
            self.assertIn("outside candidate pool", (out / "report.md").read_text())
            self.assertIn("oracle_recall5_from_candidates", (out / "queries.csv").read_text())

    def test_cache_only_command_writes_outputs_without_asr(self):
        from src.asr_adapter import ASRAdapter

        class Encoder:
            def __init__(self, *args, **kwargs):
                pass

            def encode(self, transcript):
                return transcript

        class Retriever:
            def __init__(self, *args):
                pass

            def retrieve(self, vector, k):
                return {"record_ids": ["a"], "scores": [0.9], "latency_ms": 1}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data/rag_corpus"
            data.mkdir(parents=True)
            (data / "corpus.jsonl").write_text('\n'.join(
                json.dumps({"record_id": rid, "text": rid}) for rid in ("a", "b")))
            (data / "student_audio_manifest.jsonl").write_text(json.dumps({
                "query_id": "q_test", "split": "practice", "audio_file": "absent.wav"}))
            (data / "qrels_practice.tsv").write_text(
                "query_id\trecord_id\tscore\nq_test\ta\t1\nq_test\tb\t1\n")
            cache = root / "cache.jsonl"
            cache.write_text(json.dumps({"query_id": "q_test", "transcript": "Cached question"}) + "\n")
            modules = {
                "src.encoder": types.SimpleNamespace(LocalEncoder=Encoder),
                "src.retriever": types.SimpleNamespace(ExactDenseRetriever=Retriever),
            }
            argv = [str(SCRIPT), "--cache-only", "--device", "cpu",
                    "--transcript-cache", str(cache), "--out-dir", str(root / "out")]
            with patch.object(diagnostics, "ROOT", root), patch.object(sys, "argv", argv), \
                    patch.dict(sys.modules, modules), redirect_stdout(io.StringIO()), \
                    patch.object(ASRAdapter, "prepare", side_effect=AssertionError("ASR started")):
                diagnostics.main()
            output = root / "out"
            self.assertEqual(len(list(output.iterdir())), 5)
            row = json.loads((output / "r10_candidates.jsonl").read_text())
            self.assertEqual(row["transcript"], "Cached question")
            self.assertEqual(row["retrieved_documents"][0]["text"], "a")
            self.assertEqual(row["metrics"]["Recall@100"], 0.5)
            self.assertEqual(json.loads((output / "queries.jsonl").read_text()), row)


if __name__ == "__main__":
    unittest.main()
