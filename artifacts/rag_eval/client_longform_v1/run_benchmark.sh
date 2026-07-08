#!/bin/sh
set -eu

BASE=/app/artifacts/rag_eval/client_longform_v1
LOG_DIR="$BASE/logs"
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

echo "CLIENT LONGFORM BENCHMARK START $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$LOG_DIR/master.log"
install_bm25_package
python -c "from rag_eval.config import SqlQdrantConfig; c=SqlQdrantConfig(); print(f'qdrant_timeout_sec={c.qdrant_timeout_sec}')" | tee -a "$LOG_DIR/master.log"

run_one dense_e5_small "$BASE/configs/dense_e5_small.yaml" "$BASE/out_dense_e5_small"
run_one bm25_only       "$BASE/configs/bm25_only.yaml"       "$BASE/out_bm25"
run_one hybrid_e5_small "$BASE/configs/hybrid_e5_small.yaml" "$BASE/out_hybrid_e5_small"
run_one hybrid_e5_base  "$BASE/configs/hybrid_e5_base.yaml"  "$BASE/out_hybrid_e5_base"
run_one hybrid_e5_large "$BASE/configs/hybrid_e5_large.yaml" "$BASE/out_hybrid_e5_large"
run_one hybrid_mpnet    "$BASE/configs/hybrid_mpnet.yaml"    "$BASE/out_hybrid_mpnet"
run_one hybrid_bge_m3   "$BASE/configs/hybrid_bge_m3.yaml"   "$BASE/out_hybrid_bge_m3"

python "$BASE/merge_rankings.py" 2>&1 | tee "$LOG_DIR/merge.log"
python "$BASE/finalize_winner.py" 2>&1 | tee "$LOG_DIR/finalize.log"
echo "CLIENT LONGFORM BENCHMARK DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_DIR/master.log"
