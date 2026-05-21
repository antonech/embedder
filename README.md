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

### Rebuild indices
```bash
./rebuild_index.sh /path/to/project
```
Scans project, parses AST (Python, C++, JavaScript, TypeScript, Go, Rust), embeds with sentence-transformers.
Builds both:
- `data/enriched_vectors.npz` — flat AST search
- `data/tree_vectors.npz` + `data/tree_index.json` — hierarchical AST context (parent/children/siblings)

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

## Model (embedding)

Default: `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, 50+ languages). Override in `config.json`.

Any sentence-transformers model works. Popular alternatives:
- `all-MiniLM-L6-v2` — fastest, 384-dim, English-optimized
- `all-mpnet-base-v2` — higher quality, 768-dim, slower
- `intfloat/multilingual-e5-small` — good multilingual, 384-dim

## Enrichment strategies

Configured in `config.json` under `"enrichment"` key (array of strategy names):
- `kind` — node type (class/function/method)
- `name` — symbol name
- `signature` — arguments and return type
- `docstring` — doc comments
- `body` — method body summary (first N lines)

Order matters: `["kind", "name", "signature", "docstring"]` produces e.g. `[CLASS] UserService | find(id) | Finds user by id`.

## Configuration (`config.json`)

```json
{
    "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
    "enrichment": ["kind", "name", "signature", "docstring"]
}
```

Priority: explicit arg > config.json > defaults.
