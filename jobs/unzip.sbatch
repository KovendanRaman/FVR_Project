#!/bin/bash
#SBATCH --job-name=lolblur-unzip
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home-mscluster/kraman/FVR_Project/unzip-%j.log

set -uo pipefail
RAW=~/FVR_Project/data/raw/lolblur

echo "=== what's in raw/lolblur (before) ==="
find "$RAW" -maxdepth 3 | sort

echo ""
echo "=== locating and unzipping train_all.zip and test.zip ==="
# find handles the space in 'LOLBlur Dataset/' safely via -print0
find "$RAW" \( -name "train_all.zip" -o -name "test.zip" \) -print0 | while IFS= read -r -d '' zip; do
    echo ">>> unzipping: $zip"
    dest="$(dirname "$zip")"
    unzip -q -o "$zip" -d "$dest"
    echo "    done: $zip"
done

echo ""
echo "=== resulting structure (after) ==="
find "$RAW" -maxdepth 4 -type d | sort
echo ""
echo "=== sample files (first 20) ==="
find "$RAW" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.json" \) | head -20
echo ""
echo "=== the meta.json contents (structure description) ==="
find "$RAW" -name "*meta*.json" -exec cat {} \;
