#!/bin/bash
# benchmark.sh — Run msplat on MipNeRF360 scenes and report metrics.
# Usage: ./scripts/benchmark.sh [dataset_dir] [iterations] [extra_args...]
#
# Examples:
#   ./scripts/benchmark.sh                              # baseline 7K
#   ./scripts/benchmark.sh ~/datasets/mipnerf360 7000   # baseline 7K
#   ./scripts/benchmark.sh ~/datasets/mipnerf360 7000 --random-bg --3d-filter  # with features

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
        --input "$SCENE_DIR" \
        --output "$OUTPUT" \
        --num-iters "$ITERS" \
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

    PSNR=$(grep -oP 'PSNR:\s+\K[\d.]+' "$LOG" | tail -1 || echo "N/A")
    SSIM=$(grep -oP 'SSIM:\s+\K[\d.]+' "$LOG" | tail -1 || echo "N/A")
    GAUSSIANS=$(grep -oP 'Gaussians:\s+\K[\d,]+' "$LOG" | tail -1 || echo "N/A")

    START_LINE=$(grep -n "Starting training" "$RESULTS_DIR/$scene/log.txt" 2>/dev/null | head -1 | cut -d: -f1)
    TIME="N/A"
    if [ -f "$RESULTS_DIR/$scene/log.txt" ]; then
        TIME=$(grep "Done in" "$RESULTS_DIR/$scene/log.txt" | grep -oP '\d+(?=s)' || echo "N/A")
    fi

    printf "%-12s %8s %8s %12s %10s\n" "$scene" "$PSNR" "$SSIM" "$GAUSSIANS" "$TIME"
done

echo "================================================================"
echo ""
echo "Full logs at: $RESULTS_DIR"
