#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

rag_python="${RAG_PYTHON:-$HOME/mls-rag-env/bin/python}"
colbert_python="${COLBERT_PYTHON:-$HOME/mls-colbert-env/bin/python}"
stage="${1:-smoke}"
case "$stage" in
    smoke) selection=(--limit 5) ;;
    full) selection=() ;;
    *) printf 'Usage: %s [smoke|full]\n' "$0" >&2; exit 2 ;;
esac

for interpreter in "$rag_python" "$colbert_python"; do
    if [[ ! -x "$interpreter" ]]; then
        printf 'Missing Python executable: %s\n' "$interpreter" >&2
        exit 2
    fi
done

output="rag/results/retrieval_methods_${stage}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$output"
common=(
    --device cuda --split practice
    --transcript-cache rag/.cache/practice_transcripts.jsonl
    "${selection[@]}"
)
failed=0
for method in dense colbert query2doc splade; do
    interpreter="$rag_python"
    if [[ "$method" == colbert ]]; then
        interpreter="$colbert_python"
    fi
    printf '\nRunning %s (%s)\n' "$method" "$stage"
    if ! "$interpreter" -u rag/scripts/evaluate_retrieval_methods.py \
        "${common[@]}" --method "$method" --rankings-output "$output/$method.top20.jsonl" 2>&1 | tee "$output/$method.log"; then
        printf '%s failed. See %s/%s.log\n' "$method" "$output" "$method" >&2
        failed=1
    fi
done
printf '\nLogs: %s\n' "$output"
exit "$failed"
