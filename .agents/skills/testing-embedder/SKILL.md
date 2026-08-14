---
name: testing-embedder
description: How to set up, run, and end-to-end test the embedder CLI (embedder.py / tree_ast_parser.py) and the MCP stdio server (mcp_server.py), including how to force error paths and verify stdout stays pure JSON-RPC.
---

# Testing the embedder CLI + MCP server

There is no UI. All testing is shell + JSON-RPC over stdio, so screen recording is not useful —
capture command stdout/stderr/exit codes as evidence instead.

## Environment setup

```bash
cd <repo>
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt   # large: torch + sentence-transformers
./venv/bin/pip install -r requirements-dev.txt                       # pytest, pytest-cov
```

Gotchas that have bitten before (check these first if something fails at import time):

- **`mcp<2` is mandatory.** `mcp.server.fastmcp` does **not** exist in `mcp` 2.x; a venv that
  resolves to mcp 2.0.0 fails with `ModuleNotFoundError: No module named 'mcp.server.fastmcp'`.
  `requirements.txt` pins `mcp>=1.0,<2` (1.27/1.29 work) — don't relax it.
- `rank-bm25` is in `requirements.txt`; without it the MCP server dies during startup in
  `_build_bm25()`. An existing venv predating that pin may still be missing it.
- `requirements.txt` alone does **not** install pytest; it lives in `requirements-dev.txt`.
- Pre-warm the HF cache before timing anything, otherwise server startup is network-bound:
  `intfloat/e5-small-v2` (bi-encoder) and `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranker,
  loaded unconditionally in `EmbedderApp.__init__`).
- If `sentence_transformers` cannot be installed, a stub module works for everything except
  embedding quality — but real model download over CPU is fast enough for small trees (~6s build).

## Unit tests (fastest signal — start here)

```bash
./venv/bin/python -m pytest                       # ~5s, 187 tests, no network
./venv/bin/python -m pytest --cov --cov-report=term-missing
```

The suite is fully offline: `tests/conftest.py` provides `FakeSentenceTransformer` and
`tests/test_mcp_server.py` a `FakeEncoder`, so no model is ever downloaded. Prefer adding a test
here over a shell repro; only fall back to the stdio harness below for transport-level behaviour
(JSON-RPC framing, stdout purity, startup) that unit tests cannot reach.

## Building an index

```bash
./venv/bin/python embedder.py --build-all --root /tmp/sample --data-dir /tmp/store/sample --embed-mode cpu
./venv/bin/python tree_ast_parser.py --root /tmp/sample --data-dir /tmp/store/sample [--delta]
./venv/bin/python embedder.py --build-flat --delta --root /tmp/sample --data-dir /tmp/store/sample
```

Always pass `--data-dir` explicitly in tests; otherwise it resolves from `config.json`'s
`embedding_store` (default `~/project/embedder_store`) and you will clobber real indices.
Use an **absolute** `--data-dir`: relative values are resolved against the cwd (not `--root`) by
all three builders, so a bare `--data-dir out` writes into whatever directory you invoked from.
`--embed-mode cpu` avoids the CUDA path on GPU-less boxes.

## Driving the MCP server over stdio

`mcp_server.py` reads `config.json` from **its own script directory**, not the cwd. To point it at a
scratch store without editing the repo, **copy** the modules into a scratch dir plus a custom config:

```bash
mkdir /tmp/srv && cd /tmp/srv
cp <repo>/{embedder.py,mcp_server.py,tree_search.py,tree_ast_parser.py,common.py,labels.json} .
# config.json with "embedding_store": "/tmp/store"
```
Then `python mcp_server.py --project <name>` loads `/tmp/store/<name>/enriched_vectors.npz`.

**Do not symlink the modules.** CPython resolves symlinks when computing `sys.path[0]` for the
entry-point script, so a symlinked `mcp_server.py` puts the *real repo dir* on `sys.path`; `import
common` then resolves to the repo copy and `common.CONFIG_PATH` points at the **repo's**
`config.json`. The scratch `config.json` is ignored and the server silently serves the real
`~/project/embedder_store` index. Verify isolation before trusting a run:

```bash
cd /tmp/srv && python -c "import common; print(common.CONFIG_PATH)"   # must print /tmp/srv/config.json
```
`common.py` is required (since the shared-utils extraction) — omitting it fails at import.

JSON-RPC sequence: `initialize` → `notifications/initialized` → `tools/call`.
**Read stdout incrementally (background reader threads), never wait for process exit** — the server
is long-lived, so a driver that blocks on `communicate()` will hang and then `BrokenPipeError`.
Registered tools: `search`, `embed`, `embed_many`, `store_info`, `init_store`, `add_document`,
`add_documents`, `save_store`, `load_delta`, `clear_delta`.

Key invariant to assert: **every non-empty stdout line must parse as JSON with `"jsonrpc":"2.0"`**;
all logging must land on stderr. `EMBEDDER_LOG_LEVEL=DEBUG` is the strongest way to stress this.

## Forcing error paths (useful recipes)

| Scenario | Recipe |
|---|---|
| git-diff failure in delta build | run `--delta` with `--root` pointing at a non-git dir → expect `RuntimeError: git diff failed ... (exit 129)` |
| single unparseable file | `chmod 000 somefile.py` → build continues, warning names the file |
| all files fail to parse | `chmod 000` every source file → `RuntimeError: all N files failed...`; assert pre-existing `tree_index.json`/`tree_vectors.npz` md5s are **unchanged** |
| corrupt tree index | truncate `tree_index.json` → `RuntimeError: cannot read tree index <path>` |
| tree index missing keys | write `{"texts": []}` → `RuntimeError: ... is missing 'nodes'` |
| cross-encoder load failure | set `cross_encoder_model` to a bogus repo id in config.json → `store_info` must include `cross_encoder_error`. Note `HF_HUB_OFFLINE=1` does **not** work once the model is cached |
| missing config.json | server dir with no `config.json` → must still start (regression guard for `store_root` UnboundLocalError) |
| embedding/worker failure | small harness that imports `embedder`, monkeypatches `EmbeddingModel.embed_many` or `embedder._parse_file_worker` to raise, then calls `build_all` |

## Chunk-text invariants

Chunk text is baked into the vectors, so any change here needs a rebuild before search output can
be compared. `ASTParser.scan_project(root)` prints the exact chunks without embedding anything —
use it as the fast check:

```bash
./venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from embedder import ASTParser
for c in ASTParser.scan_project('/tmp/sample'): print(' *', c)"
```

- Default enrichment is `embedder.DEFAULT_ENRICHMENT` = signature + structure + content + docstring.
  **`statements` is deliberately excluded**; `body` is a legacy alias for structure+content+statements
  kept only because `CompositeStrategy.from_keys` silently drops unknown keys.
- `structure` (in_class/methods/fields/bases) and `content` (str bodies) must both stay in the
  default: `content` is the *only* text non-code files (`.md`, `.json`, unknown) carry, so dropping
  it reduces them to a bare path.
- Python methods get their own chunk (`kind: "Method"`, `in_class` set). Before that they existed
  only as a name in the class's `Methods:` list and were unreachable except via tree fusion.
- `_python_body_lines` strips the docstring expression, so the docstring is not indexed twice.
- C++: clang is tried first when `use_clang`, and returns `[]` on failure so `_parse_cpp` degrades to
  **tree-sitter**, not to raw line blocks. Without real include paths clang errors on most files, so
  the tree-sitter path is what actually runs in practice — test that path.
- C++ out-of-line definitions (`int HashMap::find(...)`) are named via `common.ts_declarator_name`
  (qualified_identifier), which also yields the owning class. A `declaration` inside a
  `compound_statement` is a local variable and is skipped — otherwise `int idx = ...` gets indexed.

## Store/path invariants worth re-checking

These were all bugs at some point; each has a unit-test guard now, so break them and the suite
should go red:

- `load_delta`/`clear_delta` rebuild the BM25 index. Without it, `search` in `rrf`/`bm25`/alpha
  mode after a delta load raises `IndexError: index N is out of bounds for axis 0 with size M`.
- `init()` clears `_tree`, `_tree_store` **and** `_tree_to_flat`. The map is keyed by flat index,
  so a stale entry silently boosts an unrelated chunk in `_fuse_with_tree` — wrong scores, no error.
- `add_document`/`add_documents` only mark BM25 dirty; `_candidates` rebuilds it lazily. Assert via
  `app._bm25_dirty`, not by expecting an immediate rebuild.
- `init_store`/`load_delta`/`save_store` take paths as given: any `.npz` is loadable, including
  another project's index. A bare filename resolves against the directory currently being served
  (`data_dir`). There is deliberately **no path sandbox** — this is a local single-user tool driven
  by an agent that can already write files directly, and an earlier sandbox only ever obstructed
  legitimate use. Do not re-add one.
- Argument types are enforced by FastMCP's pydantic layer from the tool signatures (it coerces
  `top_k="7"` and rejects `texts="notalist"`), so manual `isinstance`/`int()` checks are redundant.
  Only `mode` and `fmt` are validated by hand, since the type hints cannot express those enums.
- `save_store` persists `node_ids`; otherwise a save→`init_store` round-trip loses tree links and
  AST context silently degrades to regex text matching.
- `VectorStore` keeps one `(N, dim)` matrix: `store.vectors` is that array (a property), not a list.
  Appends go through `add`/`add_many` and are folded in on the next read; `truncate(n)` drops the
  tail. Anything doing `store.vectors.extend(...)` or slicing it in place is a regression back to
  holding the same vectors twice — assert `store.vectors is store._get_array()`.
- `tree_index.json` is JSON Lines (`common.write_tree_index` / `common.iter_tree_index`): header
  line, then one node per line, `includes` deduplicated per file into the header. Read it with
  `iter_tree_index` in tests, never `json.loads(path.read_text())`. Both pre-JSONL layouts
  (compact and `indent=2`) must keep loading — the fixtures in `tests/test_tree_search.py` and
  `tests/test_mcp_server.py` deliberately still write the old one.

Expected (not a bug) — the delta index is **additive**, and `main()` auto-loads `delta.npz` at
startup. A changed file's chunks therefore exist twice (stale base copy + fresh delta copy) and
`search` can return the same text at two ranks with near-identical scores. Don't write assertions
that expect result texts to be unique after a delta build.

## Comparing against master

`git worktree add /tmp/master-wt master`, build a scratch server dir from **copies** of that
worktree's modules (see the symlink warning above), and run the same query specs against the same
`--data-dir` store to diff ranking output.

To check that a new unit test really guards a fix, run the *new* test file against the *old* code:

```bash
git worktree add --detach /tmp/pre-fix <commit-before-fix>
cp tests/test_*.py /tmp/pre-fix/tests/
cd /tmp/pre-fix && <repo>/venv/bin/python -m pytest -q   # the new tests must FAIL here
```
