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

Search tool supports reranking with a cross-encoder:
```
search("hash table lookup", rerank=True)   # re-rank with cross-encoder
search("hash table lookup", rerank=False)  # bi-encoder + BM25 only (default)
search("hash table lookup")                # auto: uses cross-encoder if loaded
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
| `mcp_server.py` | FastMCP server — search with bi-encoder, BM25, tree fusion, and optional cross-encoder reranking |
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

## Models

### Bi-encoder (embedding)

Default: `all-MiniLM-L6-v2` (384-dim, English-optimized). Override in `config.json`.

Device: `"cpu"` (default) or `"cuda"`. Set `"device": "cuda"` in `config.json` to use GPU. Remove the line for auto-detect.

Any sentence-transformers model works. Popular alternatives:
- `paraphrase-multilingual-MiniLM-L12-v2` — 384-dim, 50+ languages
- `all-mpnet-base-v2` — higher quality, 768-dim, slower
- `intfloat/multilingual-e5-small` — good multilingual, 384-dim

### Cross-encoder (reranking)

Optional reranker that re-scores the top retrieval candidates for better precision.
Configured via `cross_encoder_model` in `config.json`. Recommended:

- `cross-encoder/ms-marco-MiniLM-L-6-v2` — fast, good for code search

On each reranked search, the cross-encoder scores (query, candidate) pairs through
a BERT-style classification head and replaces the bi-encoder/BM25 scores with
sigmoid-normalized relevance probabilities [0, 1]. Adds ~2ms per candidate on GPU.

Usage: `search("query", rerank=True)` — defaults to auto (enabled if model loaded).

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
    "model_name": "all-MiniLM-L6-v2",
    "device": "cuda",
    "top_k": 5,
    "enrichment": ["kind", "name", "signature", "body", "docstring"],
    "use_clang": true,
    "embedding_store": "~/project/embedder_store",
    "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross_encoder_device": "cuda"
}
```

- `model_name` — sentence-transformers model for embedding
- `device` — `"cpu"`, `"cuda"`, or omit for auto-detect
- `top_k` — default number of search results
- `enrichment` — ordered list of strategy keys for flat chunk construction
- `use_clang` — enable libclang for C++ parsing (vs tree-sitter)
- `embedding_store` — base directory for per-project indices (supports `~` and `$VAR` expansion)
- `cross_encoder_model` — optional cross-encoder model for reranking (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `cross_encoder_device` — device for cross-encoder (defaults to `device` value)

Priority: explicit arg > config.json > defaults.
