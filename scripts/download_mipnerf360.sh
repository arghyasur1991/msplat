#!/bin/bash
# Download MipNeRF360 benchmark scenes (COLMAP format) for msplat evaluation.
# Usage: ./scripts/download_mipnerf360.sh [output_dir]

set -euo pipefail

DEST="${1:-$HOME/datasets/mipnerf360}"
BASE_URL="https://data.ciirc.cvut.cz/public/projects/2023NerfBaselines/data/gaussian-splatting/mipnerf360"
SCENES=(bicycle garden room counter)

mkdir -p "$DEST"
cd "$DEST"

for scene in "${SCENES[@]}"; do
  if [ -d "$scene" ]; then
    echo "[$scene] already exists, skipping"
    continue
  fi
  echo "[$scene] downloading..."
  curl -LO "${BASE_URL}/${scene}.zip"
  unzip -q "${scene}.zip" -d "${scene}"
  rm "${scene}.zip"
  echo "[$scene] done"
done

echo ""
echo "All scenes downloaded to: $DEST"
ls -ld "$DEST"/*/
