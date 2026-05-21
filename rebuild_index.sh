#!/bin/bash
# Rebuild vector stores for any project.
# Usage: ./rebuild_index.sh [/path/to/project]
#   or:  ./rebuild_index.sh --delta [/path/to/project]
#
# Modes:
#   (default)  Full rebuild — scans all files, saves enriched_vectors.npz
#   --delta    Delta rebuild — only changed files (git diff), saves delta.npz + delta_texts.json
#
# Builds:
#   - data/enriched_vectors.npz  — flat AST search
#   - data/delta.npz             — delta vectors (changed files only)
#   - data/delta_texts.json      — delta chunk texts
#   - data/tree_vectors.npz      — tree-sitter AST index
#   - data/tree_index.json       — hierarchical AST nodes

set -e
DELTA=false
PROJECT="."

while [[ $# -gt 0 ]]; do
    case "$1" in
        --delta) DELTA=true; shift ;;
        -h|--help) head -12 "$0"; exit ;;
        *) PROJECT="$1"; shift ;;
    esac
done

PROJECT="$(cd "$PROJECT" && pwd)"
export PROJECT
export DELTA
# --- Environment Setup ---
ME="$(cd "$(dirname "$0")" && pwd)"
VENV="$ME/venv"
if [ -d "$VENV" ]; then
    PYTHON="$VENV/bin/python3"
else
    PYTHON="python3"
fi

# --- Flat index via embedder.py ---
if [ "$DELTA" = false ]; then
    $PYTHON "$ME/embedder.py" --build-flat --data-dir "$DATA_DIR" --root "$PROJECT"
else
    $PYTHON "$ME/embedder.py" --build-flat --delta --data-dir "$DATA_DIR" --root "$PROJECT"
fi

# --- Tree index via tree-sitter ---
if [ "$DELTA" = false ]; then
    $PYTHON "$ME/tree_ast_parser.py" --root "$PROJECT" --data-dir "$DATA_DIR"
else
    $PYTHON "$ME/tree_ast_parser.py" --root "$PROJECT" --data-dir "$DATA_DIR" --delta
fi
