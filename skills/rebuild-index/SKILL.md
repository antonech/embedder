---
name: rebuild-index
description: Rebuild the enriched vector index from Python source files. Parses AST, enriches with class/function signatures and docstrings, embeds with all-MiniLM-L6-v2, saves to data/enriched_vectors.npz.
---

# Rebuild Vector Index

Parses `.py`, `.js`, `.ts`, `.go`, `.rs`, `.cpp` and more — extracts classes, functions, methods via AST enrichment (kind + name + signature + docstring), embeds with sentence-transformers, saves to `data/enriched_vectors.npz`.

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
