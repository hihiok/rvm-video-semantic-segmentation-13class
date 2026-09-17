#!/usr/bin/env bash
set -euo pipefail
set +x
# Required explicit paths. Use a new root initially; same root is safe for interrupted stage resume.
: "${VOTE_ROOT:?Set VOTE_ROOT}"
: "${VOTE_INPUT:?Set VOTE_INPUT}"
: "${MOBILE_PY:?Set MOBILE_PY to the proven MobileCLIP venv/bin/python}"
: "${QWEN_PY:?Set QWEN_PY to the separate teacher venv/bin/python}"
: "${VOTE_GPUS:?Set one or two idle physical GPU IDs, e.g. 2,3}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IFS=',' read -r -a cards <<< "$VOTE_GPUS"
if (( ${#cards[@]} < 1 || ${#cards[@]} > 2 )); then exit 2; fi
for card in "${cards[@]}"; do [[ "$card" =~ ^[0-9]+$ ]] || exit 2; done
if (( ${#cards[@]} == 2 )) && [[ "${cards[0]}" == "${cards[1]}" ]]; then exit 2; fi
pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
"$MOBILE_PY" "$HERE/prepare.py" --input-root "$VOTE_INPUT" --output-root "$VOTE_ROOT"
mkdir -p "$VOTE_ROOT/logs"
# Download only once; no races between workers. Existing fingerprinted snapshot is reused.
"$QWEN_PY" "$HERE/qwen.py" --output-root "$VOTE_ROOT" --download-only --insecure-downloads
run_mobile() {
  CUDA_VISIBLE_DEVICES="${cards[0]}" "$MOBILE_PY" "$HERE/mobile.py" --output-root "$VOTE_ROOT" --limit "$1" --batch-size 16
}
run_qwen() {
  local limit="$1" result=0
  pids=()
  for shard in "${!cards[@]}"; do
    CUDA_VISIBLE_DEVICES="${cards[$shard]}" "$QWEN_PY" "$HERE/qwen.py" --output-root "$VOTE_ROOT" \
      --shards "${#cards[@]}" --shard "$shard" --limit "$limit" >> "$VOTE_ROOT/logs/qwen_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || result=1; done
  pids=()
  (( result == 0 ))
}
# First render 200. Then CONTINUE automatically; no user confirmation gate here.
run_mobile 200
run_qwen 200
"$MOBILE_PY" "$HERE/merge.py" --output-root "$VOTE_ROOT" --shards "${#cards[@]}" --preview-only
printf 'PREVIEW_READY: %s/preview200/index.html; continuing all remaining images\n' "$VOTE_ROOT"
run_mobile 0
run_qwen 0
"$MOBILE_PY" "$HERE/merge.py" --output-root "$VOTE_ROOT" --shards "${#cards[@]}"
printf 'ANNOTATION_COMPLETE_REVIEW_REQUIRED: %s/merged\n' "$VOTE_ROOT"
