#!/bin/sh
# Run remaining hybrid reruns sequentially (after e5_base if still running, wait).
set -eu

LOG_DIR=/app/artifacts/rag_eval/rerun_logs
mkdir -p "$LOG_DIR"

wait_for_ranking() {
  out_dir="$1"
  name="$2"
  while [ ! -f "$out_dir/ranking.json" ]; do
    echo "waiting for $name ranking.json..."
    sleep 60
  done
  echo "$name done: $out_dir/ranking.json"
}

run_one() {
  name="$1"
  config="$2"
  out_dir="$3"
  if [ -f "$out_dir/ranking.json" ]; then
    echo "SKIP $name (ranking exists)"
    return 0
  fi
  log="$LOG_DIR/${name}.log"
  echo "=== START $name $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee "$log"
  rag-eval run --config "$config" 2>&1 | tee -a "$log"
  echo "=== END $name $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$log"
}

echo "torch=$(python -c 'import torch; print(torch.__version__)')" | tee "$LOG_DIR/sequential.log"

# wait if e5_base still running
wait_for_ranking /app/artifacts/rag_eval/out_rerun_e5_base e5_base || true

run_one mpnet    /app/artifacts/rag_eval/rerun/mpnet_hybrid.yaml    /app/artifacts/rag_eval/out_rerun_mpnet
run_one e5_large /app/artifacts/rag_eval/rerun/e5_large_hybrid.yaml /app/artifacts/rag_eval/out_rerun_e5_large
run_one bge_m3   /app/artifacts/rag_eval/rerun/bge_m3_hybrid.yaml   /app/artifacts/rag_eval/out_rerun_bge_m3

python /app/artifacts/rag_eval/merge_rankings.py 2>&1 | tee "$LOG_DIR/merge.log"
echo "SEQUENTIAL RERUNS DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_DIR/sequential.log"
