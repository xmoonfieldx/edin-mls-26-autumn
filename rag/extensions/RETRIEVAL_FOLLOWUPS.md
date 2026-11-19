# Retrieval experiments 1–3

Use `rag/scripts/evaluate_retrieval_followups.py`. The existing evaluator and
feature-export format still work. Only the explicit `gate-online` and
`shortlist-online` commands can call ELM. No answers are generated or judged.

From the repository root, activate the existing environment:

```bash
source "$HOME/mls-rag-env/bin/activate"
```

The saved `rag/results/learned_reranker/practice-top50.jsonl` supports all offline
pilots. Offline fitting uses NumPy on CPU; no GPU is needed to reuse cached
scores. The online modes run the encoder and any required MiniLM scoring on
`--device cuda`. ELM pointwise scoring runs remotely, not on that GPU.

## Inputs and controls

Tuning accepts practice IDs only. Fusion/gate use deterministic balanced outer
query folds, with up to three inner query folds to select settings. Outer test
labels never enter policy fitting/selection. Ranking functions have no qrels
argument. Query records include the fold, policy, training-query hash, IDs and
available transcript hash. The additional `.policy.json` is fitted on all
selected practice queries for future deployment; it is **not** held-out evidence.
GPU replay takes the **per-query JSONL**, not this all-practice policy file.

`--diagnostics` verifies dense IDs against the saved diagnostic and attaches
transcript hashes to legacy feature rows. The pointwise log contains rounded
recall only. The gate reconstructs hit counts using qrels, leaves unavailable
pointwise MRR null, and records the assumption that the old run used matched
inputs: a metric-only log cannot authenticate its transcript/model settings.

Online modes require a complete transcript cache. Missing/empty/conflicting
transcripts fail without ASR fallback. Fresh dense IDs and available transcript
hashes must match the offline artifact before pointwise scoring. If device or
asset changes alter IDs, regenerate matched features and policies.

Optional 100-query GPU feature export (zero ELM calls, 50 MiniLM pairs/query):

```bash
python rag/scripts/evaluate_retrieval_followups.py export \
  --transcript-cache rag/.cache/practice_transcripts.jsonl \
  --device cuda --batch-size 16 --limit 100 \
  --output rag/results/followups_features_gpu_pilot.jsonl
```

That feature file can replace the existing feature path below. Keep diagnostic
verification when reusing the old pointwise log.

## Experiment 1: dense-preserving feature fusion

The pool stays dense20. A linear model uses per-query normalised dense, MiniLM
and BM25 scores plus a dense-rank prior. It fits positive-versus-unjudged pair
differences with ridge regularisation centred on the dense prior. Queries have
equal aggregate training weight; unjudged passages are contrasts, not verified
negatives. Inner CV selects ridge strength 0.1/1/10 and protection of zero/two
dense hits. Protecting all five would prevent Recall@5 improvement and is excluded.

Offline pilot, no inference or API calls:

```bash
python rag/scripts/evaluate_retrieval_followups.py fusion \
  --features rag/results/learned_reranker/practice-top50.jsonl \
  --diagnostics rag/results/practice_diagnostics/queries.jsonl \
  --limit 100 --folds 5 \
  --output rag/results/followups_fusion_pilot.jsonl
```

Five-query GPU replay of out-of-fold policies, zero ELM calls:

```bash
python rag/scripts/evaluate_retrieval_followups.py fusion-online \
  --policies rag/results/followups_fusion_pilot.jsonl \
  --transcript-cache rag/.cache/practice_transcripts.jsonl \
  --device cuda --batch-size 16 --limit 5 \
  --output rag/results/followups_fusion_gpu_smoke.jsonl
```

This scores 20 local MiniLM pairs/query and compares dense, MiniLM and fusion.
Timings include measured shared stages: dense only for the dense control;
dense+MiniLM for MiniLM; dense+BM25+MiniLM+policy for fusion. Model/index loading,
labels and writes are excluded. Offline CV/fitting cost is reported separately.

## Experiment 2: selective pointwise

The gate chooses a complete dense or pointwise ranking. Default dense-only
features are score gaps at ranks 1/2 and 5/6. `--feature-set all` additionally
allows MiniLM/BM25 top-five disagreement, paying those online scoring/index costs.
Inner CV selects never/always or a training-quantile threshold. The inner-CV call
fraction is constrained by `--max-call-fraction`; held-out observed rates can differ.

Offline simulation using saved metric outcomes:

```bash
python rag/scripts/evaluate_retrieval_followups.py gate \
  --features rag/results/learned_reranker/practice-top50.jsonl \
  --diagnostics rag/results/practice_diagnostics/queries.jsonl \
  --pointwise-log pointwise-depth20-practice.log \
  --feature-set dense --max-call-fraction 0.5 \
  --limit 100 --folds 5 \
  --output rag/results/followups_gate_pilot.jsonl
```

Explicit five-query GPU/ELM pilot, **up to 100 new logical ELM calls**:

```bash
python rag/scripts/evaluate_retrieval_followups.py gate-online \
  --policies rag/results/followups_gate_pilot.jsonl \
  --transcript-cache rag/.cache/practice_transcripts.jsonl \
  --device cuda --batch-size 16 --limit 5 --compare-pointwise \
  --pointwise-cache rag/.cache/followups_pointwise_scores.jsonl \
  --output rag/results/followups_gate_gpu_smoke.jsonl
```

`--compare-pointwise` also scores skipped queries for an always-pointwise control.
Omit it to call pointwise only for selected queries, 20 calls each. Dense-only
gating uses the GPU encoder but does not load MiniLM or build a BM25 index.

## Experiment 3: deeper selection at the same pointwise budget

Keep dense top ten and select ten from ranks 11–40 using MiniLM. The resulting
20 IDs retain original dense order, including ordinal-score tie breaking. Qrels
do not participate in selection.

Candidate-only pilot, zero model/API calls:

```bash
python rag/scripts/evaluate_retrieval_followups.py shortlist \
  --features rag/results/learned_reranker/practice-top50.jsonl \
  --diagnostics rag/results/practice_diagnostics/queries.jsonl \
  --limit 100 \
  --output rag/results/followups_shortlist_pilot.jsonl
```

Inspect candidate Recall@20, oracle Recall@5 and labelled gains/losses versus
dense20 before choosing to run this five-query pointwise pilot:

```bash
python rag/scripts/evaluate_retrieval_followups.py shortlist-online \
  --shortlists rag/results/followups_shortlist_pilot.jsonl \
  --transcript-cache rag/.cache/practice_transcripts.jsonl \
  --device cuda --batch-size 16 --limit 5 \
  --pointwise-cache rag/.cache/followups_pointwise_scores.jsonl \
  --output rag/results/followups_shortlist_gpu_smoke.jsonl
```

Both methods have 20 logical pointwise scores/query and pay the same measured
dense+40-pair MiniLM path. Selection is recomputed from fresh local scores;
`saved_selection_matches` records whether it changed. Shared exact prompts use
the persistent cache. A fresh cache needs at most **150 ELM calls for five
queries**, because the first ten prompts are shared. Reusing the gate cache
reduces this. Model/ID/input changes are checked rather than silently ignored.

## Reports, caches and full runs

Choose fresh output paths; existing JSONL, summary and policy files are preserved.
Each run saves an adjacent `.summary.json`. Offline fusion/gate also save a
`.policy.json`. Recall/MRR values are fractions. Summaries include paired
wins/losses/ties and paired bootstrap intervals versus dense (and MiniLM where
present). These intervals condition on the out-of-fold predictions and do not
refit training policies. Five-query smoke results are not quality estimates.

Pointwise reuses the existing 0–4 prompt/parser, 1600-character truncation and
dense tie breaking. Its new persistent cache stores ordinal scores, passage/query
hashes, exact prompt-label identity, model/endpoint and request time. Per-query
exports add dense ranks/scores, final ranks and hit movements. Prompt labels are
part of cache identity because position changes the prompt. Use `--cache-only`
to prohibit new calls. Invalid scores fall back to the method's original pool;
valid partial scores survive interrupted runs. Use one writer per score cache.
If an unchanged model alias refers to a different backend setup, use a fresh
cache namespace.

Online timings include actual cache lookups and requests. Cache hits and logical
requests are reported separately: **cached paths are not comparable cold-API
latencies**. Shared gate/control results are marked; do not sum their duplicated
per-method counts. The config reports actual requests across unique scoring
invocations. SDK retries can make HTTP attempts exceed logical requests. Prompt
characters are recorded; billing tokens are not inferred from character counts.

For 350-query practice runs, remove `--limit 100` from offline commands and choose
new full-run output names. Point GPU commands to those full per-query artifacts,
remove `--limit 5`, and use new outputs. Keep device, batch size, transcripts and
cache state controlled. Dev is not used in any fitting or pilot command.

```bash
python -m unittest discover -s rag/tests -v
```
