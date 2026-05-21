# Embedder

Semantic code search with AST enrichment. Multi-language source code parser → embedding → vector search with AST context (parent/children/siblings).

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

System deps: `libclang-18-dev` (for C++ parsing).

## Usage

### Rebuild index
```bash
./rebuild_index.sh /path/to/project
```
Scans project, parses AST, embeds with sentence-transformers, saves to `data/enriched_vectors.npz`.

### MCP server (OpenCode integration)
```bash
python3 mcp_server.py --data data/enriched_vectors.npz
```
Registers tools: `search`, `tree_search`, `embed`, `embed_many`, `init_store`, `add_document`, `add_documents`, `save_store`, `store_info`.

### OpenCode skills (optional)
```bash
cp -r skills/* ~/.config/opencode/skills/
```

## Components

| File | Responsibility |
|---|---|
| `embedder.py` | ASTParser (multi-language), EmbeddingModel, VectorStore, StorageIO, EnrichmentStrategy chain |
| `mcp_server.py` | FastMCP server — 9 tools for semantic search |
| `tree_search.py` | TreeIndex — AST context overlay (parent/children/siblings) |
| `tree_ast_parser.py` | Build tree_index.json from source via tree-sitter |
| `rebuild_index.sh` | Full pipeline: scan → embed → persist |
| `config.json` | Model name, enrichment keys |

## Architecture

```
Source files
    ↓
ASTParser (Python native / libclang / tree-sitter)
    ↓
EnrichmentStrategy chain (kind | name | signature | docstring)
    ↓
EmbeddingModel → VectorStore → enriched_vectors.npz
    ↓
search() ← flat search
tree_search() ← adds AST context from TreeIndex
```

## Model

Default: `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, 50+ languages). Override in `config.json`.

## Configuration (`config.json`)

```json
{
    "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
    "enrichment": ["kind", "name", "signature", "docstring"]
}
```

Priority: explicit arg > config.json > defaults.
