---
name: rebuild-index
description: Rebuild the enriched vector index from Python source files. Parses AST, enriches with class/function signatures and docstrings, embeds with all-MiniLM-L6-v2, saves to data/enriched_vectors.npz.
---

# Rebuild Vector Index

Parses `.py` files in a project, extracts classes and functions via AST enrichment (name + signature + docstring), embeds, saves to `data/enriched_vectors.npz`.

Accepts any project directory — pass the path as argument.

## Usage

```bash
# current directory
/path/to/embedder/rebuild_index.sh

# any project
/path/to/embedder/rebuild_index.sh /path/to/project
```

## After rebuild

Load into running MCP:

```
init_store(data/enriched_vectors.npz)
```

Or restart MCP with `--data` flag.
