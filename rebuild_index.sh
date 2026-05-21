#!/bin/bash
# Rebuild enriched vector store for any Python project.
# Usage: ./rebuild_index.sh /path/to/project
#   or:  ./rebuild_index.sh                         # defaults to current dir

set -e
export PROJECT="${1:-.}"
PROJECT="$(cd "$PROJECT" && pwd)"
ME="$(cd "$(dirname "$0")" && pwd)"

cd "$ME"
source "$ME/venv/bin/activate"
export PYTHONPATH="$ME:$PYTHONPATH"

python3 << PYEOF
import os, json, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from embedder import EmbeddingModel, VectorStore, StorageIO, ASTParser

project = os.environ['PROJECT']
cfg_path = os.path.join(project, 'config.json')
model_name = 'all-MiniLM-L6-v2'
if os.path.exists(cfg_path):
    with open(cfg_path) as f:
        model_name = json.load(f).get('model_name', model_name)
enc = EmbeddingModel(model_name)
store = VectorStore()

chunks = ASTParser.scan_project(project)
vecs = enc.embed_many(chunks)
store.add_many(vecs, chunks)

out = f'{project}/data/enriched_vectors.npz'
os.makedirs(os.path.dirname(out), exist_ok=True)
StorageIO.save(out, store.vectors, store.texts, enc.dim)
print(f"Rebuilt: {len(chunks)} chunks from {project} -> {out}")
PYEOF
