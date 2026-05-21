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

### Rebuild flat index
```bash
./rebuild_index.sh /path/to/project
```
Scans project, parses AST (Python, C++, JavaScript, TypeScript, Go, Rust), embeds with sentence-transformers, saves to `data/enriched_vectors.npz`.

### Rebuild tree index (AST context)
```bash
python tree_ast_parser.py --root /path/to/project --data-dir /path/to/project/data
```
Builds `tree_vectors.npz` + `tree_index.json` with hierarchical AST nodes (parent/children/siblings).

### MCP server (OpenCode integration)
```bash
python mcp_server.py --data-dir data
```
Or for another project:
```bash
python mcp_server.py --data-dir /path/to/project/data
```

Registers tools: `search`, `tree_search`, `embed`, `embed_many`, `init_store`, `add_document`, `add_documents`, `save_store`, `store_info`.

### OpenCode skill
```bash
cp -r skills/rebuild-index ~/.config/opencode/skills/
```
Then trigger via opencode: `skill rebuild-index /path/to/project`

## Components

| File | Responsibility |
|---|---|
| `embedder.py` | ASTParser (Python/C++/JS/TS/Go/Rust), EmbeddingModel, VectorStore, StorageIO, EnrichmentStrategy chain |
| `mcp_server.py` | FastMCP server — 9 tools for semantic search |
| `tree_search.py` | TreeIndex — AST context overlay (parent/children/siblings) |
| `tree_ast_parser.py` | Build tree_index.json + tree_vectors.npz via tree-sitter |
| `rebuild_index.sh` | Full pipeline: scan → embed → persist |
| `skills/rebuild-index/` | OpenCode skill definition |
| `config.json` | Model name, enrichment keys |

## Architecture

```
Source files (.py .cpp .js .ts .go .rs ...)
    ↓
ASTParser (Python ast / libclang / tree-sitter)
    ↓
EnrichmentStrategy chain (kind | name | signature | docstring)
    ↓
EmbeddingModel → VectorStore → enriched_vectors.npz
    ↓                                ↓
search() ← flat search     TreeIndex (tree_vectors.npz)
                                  ↓
                           tree_search() ← context (parent/children/siblings)
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
