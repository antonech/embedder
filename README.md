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
./rebuild_index.sh --delta /path/to/project   # only changed files
```
Scans project, parses AST (Python, C++, JavaScript, TypeScript, Go, Rust), embeds with sentence-transformers.
Builds per-project in `embedder_store/<project_name>/`:
- `enriched_vectors.npz` — flat AST search
- `tree_vectors.npz` + `tree_index.json` — hierarchical AST context (parent/children/siblings)
- `delta.npz` + `delta_texts.json` — incremental delta index (when `--delta`)

### MCP server (OpenCode integration)
```bash
python mcp_server.py --root /path/to/project
```
Or via opencode.json:
```json
"command": ["./venv/bin/python", "./mcp_server.py", "--root", "/path/to/project"]
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
| `rebuild_index.sh` | Full pipeline: scan → embed → persist (supports `--delta`) |
| `skills/rebuild-index/` | OpenCode skill definition |
| `config.json` | Model name, enrichment keys, `embedding_store` path |
| `labels.json` | Tree-sitter node type → label mapping |

## Architecture

```
Source files (.py .cpp .js .ts .go .rs ...)
    ↓
ASTParser (Python ast / libclang / tree-sitter)
    ↓
EnrichmentStrategy chain (kind | name | signature | docstring)
    ↓
EmbeddingModel → VectorStore → embedder_store/<project>/
                                    ↓
        ┌───────────────────────────┼───────────────────────┐
        ↓                           ↓                       ↓
 enriched_vectors.npz       tree_vectors.npz          delta.npz
  (flat search)         + tree_index.json          + delta_texts.json
                             (tree search,           (incremental)
                          parent/children/siblings)
```

## Model (embedding)

Default: `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, 50+ languages). Override in `config.json`.

Device: `"cpu"` (default) or `"cuda"`. Set `"device": "cuda"` in `config.json` to use GPU. Remove the line for auto-detect.

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
    "enrichment": ["kind", "name", "signature", "docstring"],
    "embedding_store": "../embedder_store",
    "device": "cpu"
}
```

- `embedding_store` — base directory for per-project indices (resolved relative to embedder)
- `device` — `"cpu"`, `"cuda"`, or omit for auto-detect

Priority: explicit arg > config.json > defaults.
