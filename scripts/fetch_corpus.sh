#!/usr/bin/env bash
# Fetch the dbt documentation corpus (sparse checkout of website/docs).
# Usage: ./scripts/fetch_corpus.sh [dest_dir]
set -euo pipefail

DEST="${1:-data/corpus}"
REPO="https://github.com/dbt-labs/docs.getdbt.com.git"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "cloning docs (sparse, blobless)..."
git clone --depth 1 --filter=blob:none --sparse "$REPO" "$TMP/docs"
git -C "$TMP/docs" sparse-checkout set website/docs

mkdir -p "$DEST"
cp -R "$TMP/docs/website/docs/." "$DEST/"

count=$(find "$DEST" \( -name '*.md' -o -name '*.mdx' \) | wc -l | tr -d ' ')
echo "corpus ready: $count files in $DEST"
