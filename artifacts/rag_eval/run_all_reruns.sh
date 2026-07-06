#!/bin/sh
set -eu

LOG_DIR=/app/artifacts/rag_eval/rerun_logs
mkdir -p "$LOG_DIR"

has_success_ranking() {
  out_dir="$1"
  python - "$out_dir/ranking.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
data = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(0 if data.get("ranking") else 1)
PY
}

install_bm25_package() {
  rm -rf /tmp/rag-embedding-benchmark
  cp -r /packages/rag-embedding-benchmark /tmp/rag-embedding-benchmark
  pip install --no-cache-dir -q "/tmp/rag-embedding-benchmark[sql-qdrant-bm25]"
}

run_one() {
  name="$1"
  config="$2"
  out_dir="$3"
  if has_success_ranking "$out_dir"; then
    echo "SKIP $name (successful ranking exists)"
    return 0
  fi
  rm -rf "$out_dir"
  log="$LOG_DIR/${name}.log"
  echo "=== START $name $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee "$log"
  rag-eval validate --config "$config" 2>&1 | tee -a "$log"
  rag-eval run --config "$config" 2>&1 | tee -a "$log"
  echo "=== END $name $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$log"
}

echo "torch=$(python -c 'import torch; print(torch.__version__)')" | tee "$LOG_DIR/all_reruns.log"
install_bm25_package
python -c "from rag_eval.config import SqlQdrantConfig; c=SqlQdrantConfig(); print(f'qdrant_timeout_sec={c.qdrant_timeout_sec} embed_batch={c.embed_batch_size}')" | tee -a "$LOG_DIR/all_reruns.log"

run_one e5_base   /app/artifacts/rag_eval/rerun/e5_base_hybrid.yaml   /app/artifacts/rag_eval/out_rerun_e5_base
run_one mpnet     /app/artifacts/rag_eval/rerun/mpnet_hybrid.yaml     /app/artifacts/rag_eval/out_rerun_mpnet
run_one e5_large  /app/artifacts/rag_eval/rerun/e5_large_hybrid.yaml  /app/artifacts/rag_eval/out_rerun_e5_large
run_one bge_m3    /app/artifacts/rag_eval/rerun/bge_m3_hybrid.yaml    /app/artifacts/rag_eval/out_rerun_bge_m3

python /app/artifacts/rag_eval/merge_rankings.py 2>&1 | tee "$LOG_DIR/merge.log"
echo "ALL RERUNS DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_DIR/all_reruns.log"
