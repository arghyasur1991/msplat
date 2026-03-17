#!/bin/bash
# benchmark.sh — Run msplat on MipNeRF360 scenes via gs-server and report metrics.
# Usage: ./scripts/benchmark.sh [dataset_dir] [iterations]
set -euo pipefail

SERVER="http://localhost:8420"
DATASET_DIR="${1:-$HOME/datasets/mipnerf360}"
ITERS="${2:-7000}"
SCENES=(bicycle garden room counter)

echo "========================================="
echo " msplat Benchmark — ${ITERS} iterations"
echo " Dataset: ${DATASET_DIR}"
echo " Server:  ${SERVER}"
echo "========================================="
echo ""

RESULTS_FILE="/tmp/msplat_benchmark_$(date +%Y%m%d_%H%M%S).txt"
printf "%-12s %8s %8s %10s %8s\n" "Scene" "PSNR" "SSIM" "Gaussians" "Time" | tee "$RESULTS_FILE"
printf "%-12s %8s %8s %10s %8s\n" "-----" "----" "----" "---------" "----" | tee -a "$RESULTS_FILE"

for scene in "${SCENES[@]}"; do
  echo ""
  echo "=== $scene ($ITERS iters) ==="

  SCENE_DIR="$DATASET_DIR/$scene"
  if [ ! -d "$SCENE_DIR" ]; then
    echo "  SKIP: $SCENE_DIR not found"
    continue
  fi

  ZIP_FILE="/tmp/msplat_bench_${scene}.zip"
  echo "  Packaging..."
  (cd "$SCENE_DIR" && zip -rq "$ZIP_FILE" .)

  echo "  Uploading and starting training..."
  curl -s -X POST "$SERVER/upload?iterations=$ITERS" \
    --data-binary "@$ZIP_FILE" \
    -H "Content-Type: application/octet-stream" > /dev/null

  rm -f "$ZIP_FILE"

  echo "  Waiting for completion..."
  while true; do
    STATUS=$(curl -s "$SERVER/api/status")
    STATE=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin)['state'])" 2>/dev/null || echo "unknown")

    if [ "$STATE" = "done" ]; then
      PSNR=$(echo "$STATUS" | python3 -c "import sys,json; s=json.load(sys.stdin); print(s.get('eval_psnr','N/A'))")
      SSIM=$(echo "$STATUS" | python3 -c "import sys,json; s=json.load(sys.stdin); print(s.get('eval_ssim','N/A'))")
      GAUSS=$(echo "$STATUS" | python3 -c "import sys,json; s=json.load(sys.stdin); print(s.get('gaussian_count','N/A'))")
      TIME=$(echo "$STATUS" | python3 -c "import sys,json; s=json.load(sys.stdin); print(s.get('elapsed_seconds','N/A'))")
      printf "%-12s %8s %8s %10s %8ss\n" "$scene" "$PSNR" "$SSIM" "$GAUSS" "$TIME" | tee -a "$RESULTS_FILE"
      break
    elif [ "$STATE" = "error" ]; then
      MSG=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('message','unknown error'))" 2>/dev/null)
      echo "  ERROR: $MSG"
      printf "%-12s %8s\n" "$scene" "ERROR" | tee -a "$RESULTS_FILE"
      break
    fi
    sleep 5
  done
done

echo ""
echo "========================================="
echo " Results saved to: $RESULTS_FILE"
echo "========================================="
cat "$RESULTS_FILE"
