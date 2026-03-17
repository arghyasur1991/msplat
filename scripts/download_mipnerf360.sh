#!/bin/bash
# Download MipNeRF360 benchmark scenes (COLMAP format) for msplat evaluation.
# Downloads the official 360_v2.zip from Google Research.
# Usage: ./scripts/download_mipnerf360.sh [output_dir]

set -euo pipefail

DEST="${1:-$HOME/datasets/mipnerf360}"
URL="http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"

mkdir -p "$DEST"

if [ -d "$DEST/bicycle" ] && [ -d "$DEST/garden" ]; then
    echo "Dataset already exists at $DEST"
    ls -ld "$DEST"/*/
    exit 0
fi

echo "Downloading 360_v2.zip (~11.6 GB)..."
cd "$(dirname "$DEST")"
curl -LO "$URL"
echo "Unzipping..."
unzip -q 360_v2.zip -d mipnerf360_tmp
mv mipnerf360_tmp/360_v2/* "$DEST/" 2>/dev/null || mv mipnerf360_tmp/* "$DEST/"
rm -rf mipnerf360_tmp 360_v2.zip

echo ""
echo "All scenes downloaded to: $DEST"
ls -ld "$DEST"/*/
