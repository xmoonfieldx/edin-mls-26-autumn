# RAG Retrieval and Reranking Experiments

This submission contains our changes to the supplied RAG system.
We compared retrieval and reranking methods to study their effect
on recall and per-query latency.

## 1. Prerequisites

First, follow the setup instructions in the supplied repository:

https://github.com/ed-aisys/edin-mls-26-autumn/blob/main/rag/README.md

This includes installing dependencies, downloading the course assets,
configuring ASR and adding your own ELM credentials.

The submitted files run inside that repository. They are not a
standalone application.

## 2. Add our files

Copy the submitted files into the matching folders below.
Replace existing files where the names match.

```text
rag/
├── scripts/
│   ├── evaluate_rag.py
│   ├── diagnose_retrieval.py
│   ├── evaluate_retrieval_followups_depth.py
│   └── compare_answers.py
└── extensions/
    ├── llm_reranking.py
    ├── cross_encoder_reranking.py
    ├── lexical_retrieval.py
    └── retrieval_followups.py
```

Keep the remaining supplied framework files in place.

Run all commands below from the repository root, with your RAG
Python environment activated.

## 3. Run the dense baseline

```bash
python -u rag/scripts/evaluate_rag.py \
  --device cpu \
  --split practice \
  --transcript-cache rag/.cache/practice_transcripts.jsonl
```

The first run may take longer because missing transcripts must be
generated using ASR. The script saves them in the transcript cache.
Later runs reuse these transcripts so each method receives the
same input.

Without `--limit`, the script evaluates all 350 practice queries.
Add `--limit 5` for a small initial test.

The script prints mean recall, ranking metrics and median query
latency. These timings exclude transcription and answer generation.

## 4. Compare retrieval and reranking methods

Use the same transcript cache for each comparison.

### BM25 reranking

```bash
python -u rag/scripts/evaluate_rag.py \
  --device cpu --split practice \
  --transcript-cache rag/.cache/practice_transcripts.jsonl \
  --rerank-depth 20 --bm25-rerank
```

### Dense and BM25 rank fusion

```bash
python -u rag/scripts/evaluate_rag.py \
  --device cpu --split practice \
  --transcript-cache rag/.cache/practice_transcripts.jsonl \
  --hybrid-rrf
```

### MiniLM cross-encoder reranking

```bash
python -u rag/scripts/evaluate_rag.py \
  --device cpu --split practice \
  --transcript-cache rag/.cache/practice_transcripts.jsonl \
  --rerank-depth 20 --cross-encoder
```

The cross-encoder may download its checkpoint on first use.
To test a deeper candidate pool, change `--rerank-depth` to
50 or 100.

### Pointwise ELM reranking

```bash
POINTWISE_PROMPT_VARIANT=baseline \
python -u rag/scripts/evaluate_rag.py \
  --device cpu --split practice \
  --transcript-cache rag/.cache/practice_transcripts.jsonl \
  --rerank-depth 20 --llm-rerank pointwise
```

This scores each candidate separately. A full 350-query run at
depth 20 requires 7,000 scoring requests if every query completes.
This command does not reuse a pointwise-score cache.

To try the alternative prompts, replace `baseline` with
`evidence` or `asr`.

### Listwise ELM reranking

```bash
python -u rag/scripts/evaluate_rag.py \
  --device cpu --split practice \
  --transcript-cache rag/.cache/practice_transcripts.jsonl \
  --rerank-depth 20 --llm-rerank listwise
```

This uses one ELM request per query in the tested configuration.
Invalid responses fall back to the original dense order.

BM25 and MiniLM evaluation do not make ELM requests.
Pointwise and listwise evaluation require your own ELM credentials.

Use `--device cuda` instead of `cpu` if running on an allocated GPU.
This accelerates local models; ELM inference runs on the remote service.

## 5. Diagnose the dense ranking

After generating the full transcript cache, run:

```bash
python -u rag/scripts/diagnose_retrieval.py \
  --split practice \
  --device cpu \
  --transcript-cache rag/.cache/practice_transcripts.jsonl \
  --cache-only \
  --k-max 100 \
  --out-dir rag/results/submission_diagnostics
```

This examines whether relevant passages are ranked too low or
missing from the top 100. It writes a readable report and detailed
query results.

`--cache-only` requires complete cached transcripts. It prevents
new transcription, but query encoding and retrieval still run.
No ELM requests are made.

Choose a new output directory if one already exists.

## 6. Evaluate learned score fusion

First export dense, BM25 and cross-encoder features:

```bash
python -u rag/scripts/evaluate_retrieval_followups_depth.py export \
  --depth 100 \
  --device cpu \
  --batch-size 16 \
  --transcript-cache rag/.cache/practice_transcripts.jsonl \
  --output rag/results/submission_top100_features.jsonl
```

Then fit and evaluate fusion using those saved features:

```bash
python -u rag/scripts/evaluate_retrieval_followups_depth.py fusion \
  --features rag/results/submission_top100_features.jsonl \
  --depth 100 \
  --folds 5 \
  --seed 17 \
  --output rag/results/submission_fusion100.jsonl
```

Both commands make zero ELM requests. Feature export runs local
models; the fusion step reuses their saved scores.

Fusion uses five outer folds split by query. Hyperparameters are
selected within the training queries using inner cross-validation.

The script saves per-query results and an adjacent summary file.
Use fresh output filenames when repeating an experiment.

## 7. Compare saved answer scores

If the answer files and completed judge outputs are included,
run:

```bash
answers="PATH_TO_SAVED_ANSWER_DIRECTORY"

python rag/scripts/compare_answers.py \
  "$answers/dense.jsonl" \
  "$answers/pointwise.jsonl" \
  "$answers/judge_dense" \
  "$answers/judge_pointwise" \
  --k 10
```

Replace the placeholder with the directory containing the saved files.
Each judge folder must contain `answer_judge_per_query.jsonl`.

This command reads existing answers and judgements. It does not
generate answers or make new ELM requests. It prints the paired
mean score difference and a bootstrap confidence interval.

## Notes

- Use the same transcripts, corpus and device when comparing methods.
- Cached-response timings are not equivalent to uncached API latency.
- The scripts in this minimal submission cover the main experiments.
  Additional appendix experiments require their corresponding scripts.
- Credentials, supplied model weights, corpus assets and Python
  environments are not included.