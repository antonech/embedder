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
    EnrichmentStrategy chain (<kind> <file> <name> | signature | body | docstring)
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

Default: `intfloat/e5-small-v2` (384-dim, English). Override `"model_name"` in `config.json`.

Device options: `"cuda"`, `"cuda:N"`, `"cpu"`, or omit for auto-detect. See [Configuration](#configuration-configjson).

Any sentence-transformers model works. The system auto-detects instruction prefix requirements:

| Model family | Pattern | Prefix | Example |
|---|---|---|---|
| E5 (`intfloat/e5-*`, `intfloat/multilingual-e5-*`) | `query:` / `passage:` | Applied automatically | `e5-small-v2`, `multilingual-e5-small` |
| BGE (`BAAI/bge-*`) | `Represent this sentence...` | Applied automatically | `bge-small-en-v1.5` |
| Others (MiniLM, MPNet, etc.) | No prefix | Raw text | `all-MiniLM-L6-v2`, `all-mpnet-base-v2` |

Override auto-detection with `"query_prefix"` / `"passage_prefix"` in config.

Popular alternatives:
- `all-MiniLM-L6-v2` — 384-dim, no prefix, fast, English (was the default)
- `paraphrase-multilingual-MiniLM-L12-v2` — 384-dim, 50+ languages
- `all-mpnet-base-v2` — higher quality, 768-dim, slower
- `intfloat/multilingual-e5-small` — 384-dim, multilingual, requires `passage:`/`query:` prefixes
- `BAAI/bge-small-en-v1.5` — 384-dim, good for retrieval, uses `Represent this sentence...` prefix

**Important:** When switching models, always rebuild the index — vector dimensions may differ.

### Cross-encoder (reranking)

Optional reranker that re-scores the top retrieval candidates for better precision.
Configured via `"cross_encoder_model"` in `config.json`. Recommended:

- `cross-encoder/ms-marco-MiniLM-L-6-v2` — fast, good for code search

On each reranked search, the cross-encoder scores (query, candidate) pairs through
a BERT-style classification head and replaces the bi-encoder/BM25 scores with
sigmoid-normalized relevance probabilities [0, 1]. Adds ~2ms per candidate on GPU.

Usage: `search("query", rerank=True)` — defaults to auto (enabled if model loaded).

## Enrichment strategies

Configured in `config.json` under `"enrichment"` key (array of strategy names).
Applied to each AST node to build the chunk text as `<kind> <file> <name> | <strategy fields...>`.

Available strategies (order matters):
- `signature` — arguments and return type
- `body` — method body / fields / bases summary (first N lines)
- `docstring` — doc comments
- `kind` — node type (class/function/method; already in prefix)
- `name` — symbol name (already in prefix)

Default: `["signature", "body", "docstring"]` produces e.g. `Class utils.py UserService | find(id) | Methods: create, delete | Finds user by id`.

## Configuration (`config.json`)

```json
{
    "model_name": "intfloat/e5-small-v2",
    "batch_size": 1024,
    // "device": "cuda",       // uncomment for GPU-only; omit for auto multi (GPU+CPU) or CPU
    "enrichment": ["signature", "body", "docstring"],
    "use_clang": true,
    "embedding_store": "~/project/embedder_store",
    "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
}
```

- `model_name` — sentence-transformers model (see [Models](#models) for compatible models)
- `float_type` — `"fp16"` for half precision (faster, less VRAM) or `"fp32"` (default)
- `batch_size` — texts per GPU batch (default: 1024); increase for throughput, decrease for low VRAM
- `device` — controls where embedding runs (omit for auto):
  - `"cuda"` / `"cuda:0"` / `"cuda:1"` — GPU only
  - `"cpu"` — CPU only
  - omitted — auto-detect: GPU + CPU parallel (`multi`) if CUDA available, else `"cpu"`; CLI `--embed-mode` overrides
- `enrichment` — ordered list of strategy keys for flat chunk construction (default: `["signature", "body", "docstring"]`)
- `use_clang` — enable libclang for C++ parsing (vs tree-sitter)
- `embedding_store` — base directory for per-project indices (supports `~` and `$VAR` expansion)
- `query_prefix` / `passage_prefix` — override auto-detected E5/BGE instruction prefixes (set to `""` to disable)
- `cross_encoder_model` — optional cross-encoder model for reranking (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`); uses `device` setting

Priority: explicit arg > config.json > defaults.
