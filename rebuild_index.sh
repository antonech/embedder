#!/bin/bash
# Rebuild vector stores for any project.
# Usage: ./rebuild_index.sh /path/to/project
#   or:  ./rebuild_index.sh                         # defaults to current dir
#
# Builds:
#   - data/enriched_vectors.npz  — flat AST search
#   - data/tree_vectors.npz      — tree-sitter AST index
#   - data/tree_index.json       — hierarchical AST nodes

set -e
PROJECT="${1:-.}"
PROJECT="$(cd "$PROJECT" && pwd)"
export PROJECT
ME="$(cd "$(dirname "$0")" && pwd)"

cd "$ME"
source "$ME/venv/bin/activate"
export PYTHONPATH="$ME:$PYTHONPATH"
DATA_DIR="$PROJECT/data"
mkdir -p "$DATA_DIR"

# --- Flat index via ASTParser ---
python3 << PYEOF
import os, json, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from embedder import EmbeddingModel, VectorStore, StorageIO, ASTParser

project = os.environ['PROJECT']
cfg_path = os.path.join(project, 'config.json')
model_name = 'all-MiniLM-L6-v2'
device = None
if os.path.exists(cfg_path):
    with open(cfg_path) as f:
        cfg = json.load(f)
        model_name = cfg.get('model_name', model_name)
        device = cfg.get('device')
enc = EmbeddingModel(model_name, device=device)
store = VectorStore()

chunks = ASTParser.scan_project(project)
vecs = enc.embed_many(chunks)
store.add_many(vecs, chunks)

out = f'{project}/data/enriched_vectors.npz'
os.makedirs(os.path.dirname(out), exist_ok=True)
StorageIO.save(out, store.vectors, store.texts, enc.dim)
print(f"Flat index: {len(chunks)} chunks -> {out}")
PYEOF

# --- Tree index via tree-sitter ---
python3 tree_ast_parser.py --root "$PROJECT" --data-dir "$DATA_DIR"
