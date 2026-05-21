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
ME="$(cd "$(dirname "$0")" && pwd)"

cd "$ME"
source "$ME/venv/bin/activate"
export PYTHONPATH="$ME:$PYTHONPATH"
DATA_DIR="$PROJECT/data"
mkdir -p "$DATA_DIR"

# --- Flat index via ASTParser ---
python3 << PYEOF
import os, json, subprocess, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from embedder import EmbeddingModel, VectorStore, StorageIO, ASTParser

project = os.environ['PROJECT']
delta = os.environ.get('DELTA') == 'true'

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

if delta:
    # --- Delta mode: only changed files ---
    os.chdir(project)
    result = subprocess.run(
        ['git', 'diff', '--name-only', 'HEAD'],
        capture_output=True, text=True, timeout=30
    )
    changed = [f.strip() for f in result.stdout.split('\n') if f.strip()]
    if not changed:
        print("Delta: no changed files")
        StorageIO.save(f'{project}/data/delta.npz', [], [], enc.dim)
        with open(f'{project}/data/delta_texts.json', 'w') as f:
            json.dump({"files": [], "texts": [], "model": model_name}, f)
        sys.exit(0)

    print(f"Delta: {len(changed)} changed files")
    chunks = []
    for fp in changed:
        abspath = os.path.join(project, fp)
        if not os.path.isfile(abspath):
            continue
        try:
            file_chunks = ASTParser.parse_file(abspath, path_hint=fp)
            if file_chunks:
                chunks.append(f"[FILE] {fp}")
                chunks.extend(file_chunks)
        except Exception as e:
            chunks.append(f"[FILE] {fp} (read error: {e})")

    if not chunks:
        print("Delta: no parseable chunks")
        StorageIO.save(f'{project}/data/delta.npz', [], [], enc.dim)
        with open(f'{project}/data/delta_texts.json', 'w') as f:
            json.dump({"files": changed, "texts": [], "model": model_name}, f)
        sys.exit(0)

    vecs = enc.embed_many(chunks)
    store.add_many(vecs, chunks)

    out_vec = f'{project}/data/delta.npz'
    os.makedirs(os.path.dirname(out_vec), exist_ok=True)
    StorageIO.save(out_vec, store.vectors, store.texts, enc.dim)

    delta_data = {
        "files": changed,
        "texts": chunks,
        "model": model_name,
    }
    with open(f'{project}/data/delta_texts.json', 'w') as f:
        json.dump(delta_data, f, ensure_ascii=False)

    print(f"Delta index: {len(chunks)} chunks from {len(changed)} files -> {out_vec}")
else:
    # --- Full rebuild ---
    chunks = ASTParser.scan_project(project)
    vecs = enc.embed_many(chunks)
    store.add_many(vecs, chunks)

    out = f'{project}/data/enriched_vectors.npz'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    StorageIO.save(out, store.vectors, store.texts, enc.dim)
    print(f"Flat index: {len(chunks)} chunks -> {out}")
PYEOF

# --- Tree index via tree-sitter ---
if [ "$DELTA" = false ]; then
    python3 tree_ast_parser.py --root "$PROJECT" --data-dir "$DATA_DIR"
else
    python3 tree_ast_parser.py --root "$PROJECT" --data-dir "$DATA_DIR" --delta
fi
