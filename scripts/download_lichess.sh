#!/bin/bash
# Download Lichess CC0 database games
# Usage: bash scripts/download_lichess.sh

set -e

DATA_DIR="data"
mkdir -p "$DATA_DIR"

echo "Downloading Lichess CC0 database..."
echo "This will download a sample of rated games (latest month)"
echo ""

# URL for latest standard rated games (change YYYY-MM as needed)
# See https://database.lichess.org/ for available archives
YEAR_MONTH="2024-01"
URL="https://database.lichess.org/standard/lichess_db_standard_rated_${YEAR_MONTH}.pgn.zst"

OUTPUT_FILE="$DATA_DIR/lichess_${YEAR_MONTH}.pgn.zst"

if [ -f "$OUTPUT_FILE" ]; then
    echo "File already exists: $OUTPUT_FILE"
    echo "Skipping download. Delete the file to re-download."
else
    echo "Downloading from: $URL"
    echo "Output: $OUTPUT_FILE"
    echo ""

    # Download with progress bar
    curl -L --progress-bar "$URL" -o "$OUTPUT_FILE"

    echo ""
    echo "Download complete!"
fi

FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
echo "File size: $FILE_SIZE"

echo ""
echo "To extract and filter games:"
echo "  python -m train.dataset --pgn \"$OUTPUT_FILE\" --out data/positions.jsonl --elo-min 1800 --sample-rate 0.1 --max-positions 1000000"

echo ""
echo "Note: You'll need zstd installed to decompress .zst files:"
echo "  macOS: brew install zstd"
echo "  Ubuntu: sudo apt-get install zstd"
echo "  Or use python-zstandard (pip install zstandard)"
