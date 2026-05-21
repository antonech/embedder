---
name: rebuild-index
description: Rebuild enriched vector index and tree-sitter AST index from source files. Parses AST, enriches with signatures and docstrings, embeds with sentence-transformers, saves both flat and hierarchical indices.
---

# Rebuild Vector Index

Parses `.py`, `.js`, `.ts`, `.go`, `.rs`, `.cpp` and more — extracts classes, functions, methods via AST enrichment (kind + name + signature + docstring), embeds with sentence-transformers.

Builds both indices in one run:
- `data/enriched_vectors.npz` — flat AST search
- `data/tree_vectors.npz` + `data/tree_index.json` — hierarchical AST context (parent/children/siblings)

Accepts any project directory — pass the path as argument.

## Usage

```bash
# current directory
/path/to/embedder/rebuild_index.sh

# any project
/path/to/embedder/rebuild_index.sh /path/to/project
```

## Enrichment strategies

Configured in project's `config.json`:

```json
{
    "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
    "enrichment": ["kind", "name", "signature", "docstring"]
}
```

Available: `kind`, `name`, `signature`, `docstring`, `body` — combined via `CompositeStrategy`.

## After rebuild

Load into running MCP:

```
init_store(data/enriched_vectors.npz)
```

Or restart MCP with `--data-dir`:

```bash
python mcp_server.py --data-dir /path/to/project/data
```
