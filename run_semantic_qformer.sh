#!/usr/bin/env bash
set -euo pipefail
cd ~/Kiana2/signal-derived
CONFIG="configs/datasets/ptbxl_semantic_qformer.toml"
RUN_ID="ptbxl_medtsllm_convnext_semantic_qformer_$(date +%Y%m%d_%H%M%S)"
mkdir -p outputs/run_logs
cp "$CONFIG" "outputs/run_logs/${RUN_ID}_config.toml"
nohup python -u train.py \
  "$CONFIG" \
  "$RUN_ID" \
  > "outputs/run_logs/${RUN_ID}.log" 2>&1 &
PID=$!
echo "$PID" > "outputs/run_logs/${RUN_ID}.pid"
echo "PID: $PID"
echo "RUN_ID: $RUN_ID"
echo "LOG: outputs/run_logs/${RUN_ID}.log"
echo "RESULTS: outputs/results/${RUN_ID}.json"
tail -f "outputs/run_logs/${RUN_ID}.log"
