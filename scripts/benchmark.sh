#!/bin/bash
# benchmark.sh — Run msplat on MipNeRF360 scenes and report metrics.
# Usage: ./scripts/benchmark.sh [dataset_dir] [iterations] [extra_args...]
#
# Examples:
#   ./scripts/benchmark.sh                              # baseline 7K
#   ./scripts/benchmark.sh ~/datasets/mipnerf360 7000   # baseline 7K
#   ./scripts/benchmark.sh ~/datasets/mipnerf360 7000 --random-bg --3d-filter

set -euo pipefail

MSPLAT="${MSPLAT_BIN:-$(dirname "$0")/../build/msplat}"
DATASET_DIR="${1:-$HOME/datasets/mipnerf360}"
ITERS="${2:-7000}"
shift 2 2>/dev/null || true
EXTRA_ARGS="$@"

SCENES=(bicycle garden room counter)
RESULTS_DIR="/tmp/msplat_benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "================================================================"
echo "msplat Benchmark"
echo "  Binary:     $MSPLAT"
echo "  Dataset:    $DATASET_DIR"
echo "  Iterations: $ITERS"
echo "  Extra args: ${EXTRA_ARGS:-none}"
echo "  Results:    $RESULTS_DIR"
echo "================================================================"
echo ""

for scene in "${SCENES[@]}"; do
    SCENE_DIR="$DATASET_DIR/$scene"
    if [ ! -d "$SCENE_DIR" ]; then
        echo "[$scene] SKIP — directory not found: $SCENE_DIR"
        continue
    fi

    OUTPUT="$RESULTS_DIR/$scene/splat.ply"
    mkdir -p "$RESULTS_DIR/$scene"

    echo "[$scene] Starting training ($ITERS iterations)..."
    START_TIME=$(date +%s)

    "$MSPLAT" \
        "$SCENE_DIR" \
        -o "$OUTPUT" \
        -n "$ITERS" \
        --num-downscales 0 \
        --eval \
        $EXTRA_ARGS \
        2>&1 | tee "$RESULTS_DIR/$scene/log.txt"

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    echo "[$scene] Done in ${ELAPSED}s"
    echo ""
done

echo ""
echo "================================================================"
echo "RESULTS SUMMARY"
echo "================================================================"
printf "%-12s %8s %8s %12s %10s\n" "Scene" "PSNR" "SSIM" "Gaussians" "Time(s)"
echo "----------------------------------------------------------------"

for scene in "${SCENES[@]}"; do
    LOG="$RESULTS_DIR/$scene/log.txt"
    if [ ! -f "$LOG" ]; then
        printf "%-12s %8s %8s %12s %10s\n" "$scene" "N/A" "N/A" "N/A" "N/A"
        continue
    fi

    SUMMARY=$(grep "PSNR:.*SSIM:.*Gaussians:" "$LOG" | tail -1)
    if [ -n "$SUMMARY" ]; then
        PSNR=$(echo "$SUMMARY" | sed 's/.*PSNR:[[:space:]]*//' | awk '{print $1}')
        SSIM=$(echo "$SUMMARY" | sed 's/.*SSIM:[[:space:]]*//' | awk '{print $1}')
        GAUSSIANS=$(echo "$SUMMARY" | sed 's/.*Gaussians:[[:space:]]*//' | awk '{print $1}')
    else
        PSNR="N/A"; SSIM="N/A"; GAUSSIANS="N/A"
    fi

    TIME="N/A"
    if grep -q "Done in" "$LOG"; then
        TIME=$(grep "Done in" "$LOG" | sed 's/.*Done in \([0-9]*\)s.*/\1/')
    fi

    printf "%-12s %8s %8s %12s %10s\n" "$scene" "$PSNR" "$SSIM" "$GAUSSIANS" "$TIME"
done

echo "================================================================"
echo ""
echo "Full logs at: $RESULTS_DIR"
