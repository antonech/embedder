#!/bin/bash
# Rebuild vector stores for any project.
# Usage: ./rebuild_index.sh [/path/to/project]
#   or:  ./rebuild_index.sh --delta [/path/to/project]
#
# Modes:
#   (default)  Full rebuild — scans all files, saves enriched_vectors.npz
#   --delta    Delta rebuild — only changed files (git diff), saves delta.npz + delta_texts.json
#
# Builds (per-project in embedder_store/$(basename <project>)/):
#   - enriched_vectors.npz       — flat AST search
#   - delta.npz                  — delta vectors (changed files only)
#   - delta_texts.json           — delta chunk texts
#   - tree_vectors.npz           — tree-sitter AST index
#   - tree_index.json            — hierarchical AST nodes

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

# --- Tree index via tree-sitter (build first so flat index can reference node_ids) ---
if [ "$DELTA" = false ]; then
    $PYTHON "$ME/tree_ast_parser.py" --root "$PROJECT"
else
    $PYTHON "$ME/tree_ast_parser.py" --root "$PROJECT" --delta
fi

# --- Flat index via embedder.py (data_dir computed from config + project name) ---
if [ "$DELTA" = false ]; then
    $PYTHON "$ME/embedder.py" --build-flat --root "$PROJECT"
else
    $PYTHON "$ME/embedder.py" --build-flat --delta --root "$PROJECT"
fi
