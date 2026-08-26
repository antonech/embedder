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
TypeScript is parsed with the JavaScript grammar, so TS-only syntax (type annotations, interfaces,
enums) is not extracted as its own node — functions and classes are.
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

Registers tools: `search`, `embed`, `embed_many`, `store_info`, `init_store`, `add_document`,
`add_documents`, `save_store`, `load_delta`, `clear_delta`. Every tool takes a `project` argument.

AST context (parent/children/siblings) is not a separate tool — `search` attaches it to each hit
from the tree index.

`init_store` can swap in an index from any directory, so one server can search another project or
an index built from a temporary source:
```
init_store("/path/to/embedder_store/other_project/enriched_vectors.npz")
init_store("enriched_vectors.npz")   # bare filename: the directory currently served
```

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
    EnrichmentStrategy chain (<kind> <file> <name> | signature | structure | content | docstring)
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

On each reranked search, retrieval widens to `top_k * RERANK_CANDIDATES` (4x) candidates
and the cross-encoder scores (query, candidate) pairs through a BERT-style classification
head, replacing the bi-encoder/BM25 scores with sigmoid-normalized relevance
probabilities [0, 1] before truncating to `top_k`. Adds ~2ms per candidate on GPU.

Usage: `search("query", rerank=True)` — defaults to auto (enabled if model loaded).

## Index format

`*.npz` files store chunk texts as a length-prefixed UTF-8 blob — smaller and faster than a numpy
object array, and it loads without pickle. Older indices that stored a pickled `texts` array are
still read as they are.

`tree_index.json` is JSON Lines: a header (`{"format": "tree-jsonl", "count": N, "includes": {…}}`)
followed by one node per line. The reader streams it, so only one node is in memory at a time, and
per-file `#include` lists live in the header instead of being copied onto every node of that file.
Indices written before this layout — a single JSON object with `nodes` and a parallel `texts` array —
are still read, but whole-file parsing costs several times the memory. Rebuild to get the new one:

| clickhouse tree index (102,651 nodes) | old | new |
|---|---|---|
| file size | 105 MB | 32 MB |
| resident after load | 273 MB | 80 MB |
| peak while loading | 457 MB | 80 MB |
| load time | 1.34 s | 0.62 s |

Vectors are held as a single `(N, dim)` matrix rather than a list of rows plus a stacked copy of the
same data; on that index the flat store went from ~1.9 GB to ~1.25 GB resident, and a full server
(model + flat index + tree index + BM25) from 3.2 GB to 2.4 GB.

BM25 postings (`bm25.py`) are computed once by the index builders and persisted inside the same
`.npz` (`bm25_*` keys). Loading them takes ~0.16 s instead of re-tokenizing the whole corpus
(~10.6 s on clickhouse), scoring touches only documents containing the query terms (~0.7 ms vs
~175 ms median per query), and deltas/`add_document` extend a small in-memory overlay instead of
rebuilding. Scores match `rank_bm25.BM25Okapi` to within 1e-14 on the real index. Indices without
postings fall back to `rank_bm25` unchanged. Rebuild to get them.

## Enrichment strategies

Configured in `config.json` under `"enrichment"` key (array of strategy names).
Applied to each AST node to build the chunk text as `<kind> <file> <name> | <strategy fields...>`.

Available strategies (order matters):
- `signature` — arguments and return type
- `structure` — owning class, methods, fields, bases (`In class: X`, `Methods: …`, `Inherits: …`)
- `content` — prose/summary text: the body of non-code files (`.md`, `.json`, unknown types) and
  the C++ class summary
- `docstring` — doc comments
- `statements` — first few source statements of a function body (**opt-in**, see below)
- `kind` — node type (class/function/method; already in prefix)
- `name` — symbol name (already in prefix)
- `body` — legacy alias for `structure` + `content` + `statements`

Default: `["signature", "structure", "content", "docstring"]`, producing e.g.

```
Class    svc.py UserService | Methods: create_user, delete_user | Manages user records.
Method   svc.py create_user | (self, name, email) | In class: UserService | Create a new user record.
Function util.cpp find      | (int key) | In class: HashMap
File     NOTES.md           | The service reads DATABASE_URL from the environment and retries…
```

### Why `statements` is opt-in

Raw body lines are mostly noise for retrieval: control flow and local variable names rarely
describe what a symbol is *for*, and they dilute the signature and docstring that do. They also
duplicate what the tree index already covers. Enable them only if you specifically want to match
on implementation details:

```json
"enrichment": ["signature", "structure", "content", "statements", "docstring"]
```

Statements exclude the docstring expression, so it is no longer indexed twice.

**Changing `enrichment` requires a rebuild** — chunk text is baked into the vectors.

## Configuration (`config.json`)

```json
{
    "model_name": "intfloat/e5-small-v2",
    "batch_size": 4096,
    "enrichment": ["signature", "structure", "content", "docstring"],
    "use_clang": true,
    "embedding_store": "~/project/embedder_store",
    "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
}
```

Strict JSON — comments are not allowed. Add `"device": "cuda"` for GPU-only; omit
`device` for auto (GPU+CPU `multi` if CUDA is available, else CPU).

- `model_name` — sentence-transformers model (see [Models](#models) for compatible models)
- `float_type` — `"fp16"` for half precision (faster, less VRAM) or `"fp32"` (default)
- `batch_size` — texts per GPU batch (default: 1024); increase for throughput, decrease for low VRAM
- `device` — controls where embedding runs (omit for auto):
  - `"cuda"` / `"cuda:0"` / `"cuda:1"` — GPU only
  - `"cpu"` — CPU only
  - omitted — auto-detect: GPU + CPU parallel (`multi`) if CUDA available, else `"cpu"`; CLI `--embed-mode` overrides
- `enrichment` — ordered list of strategy keys for flat chunk construction
  (default: `["signature", "structure", "content", "docstring"]`; see
  [Enrichment strategies](#enrichment-strategies))
- `use_clang` — try libclang first for C++ parsing. It needs the project's real include paths and
  compile flags to resolve a translation unit; when it errors, parsing falls back to tree-sitter
  (structured), not to raw line chunks. Set `false` to use tree-sitter directly.
- `embedding_store` — base directory for per-project indices (supports `~` and `$VAR` expansion)
- `query_prefix` / `passage_prefix` — override auto-detected E5/BGE instruction prefixes (set to `""` to disable)
- `cross_encoder_model` — optional cross-encoder model for reranking (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`); uses `device` setting

Priority: explicit arg > `<--root>/config.json` > the embedder's own `config.json` > defaults.

A scanned project may keep its own `config.json` to override model/enrichment settings for
that project. A `config.json` that is not an embedder config (unrelated keys, comments,
not a JSON object) is ignored with a warning on stderr — indexing an unrelated repository
that happens to ship a `config.json` neither fails nor silently switches models. The
embedder's own `config.json` and `labels.json` are still strict: a broken file there is an
error, not a silent fallback.

`--data-dir` is used as given by all builders (`--build-all`, `--build-flat`, `tree_ast_parser.py`);
a relative value is relative to the current directory, not to `--root`. Prefer absolute paths.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest                                   # offline, no model download
python -m pytest --cov --cov-report=term-missing
```

Tests stub out sentence-transformers (`tests/conftest.py`), so they need no network and no GPU.
