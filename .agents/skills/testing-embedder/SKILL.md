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
./venv/bin/pip install "mcp<2" rank_bm25
```

Gotchas that have bitten before (check these first if something fails at import time):

- **`mcp<2` is mandatory.** `mcp.server.fastmcp` does **not** exist in `mcp` 2.x; a venv that
  resolves to mcp 2.0.0 fails with `ModuleNotFoundError: No module named 'mcp.server.fastmcp'`.
  Pin `mcp<2` (1.29.0 works).
- **`rank_bm25` may be missing from requirements.txt.** Without it the MCP server dies during
  startup in `_build_bm25()`. Install it explicitly.
- Pre-warm the HF cache before timing anything, otherwise server startup is network-bound:
  `intfloat/e5-small-v2` (bi-encoder) and `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranker,
  loaded unconditionally in `EmbedderApp.__init__`).
- If `sentence_transformers` cannot be installed, a stub module works for everything except
  embedding quality — but real model download over CPU is fast enough for small trees (~6s build).

## Building an index

```bash
./venv/bin/python embedder.py --build-all --root /tmp/sample --data-dir /tmp/store/sample --embed-mode cpu
./venv/bin/python tree_ast_parser.py --root /tmp/sample --data-dir /tmp/store/sample [--delta]
./venv/bin/python embedder.py --build-flat --delta --root /tmp/sample --data-dir /tmp/store/sample
```

Always pass `--data-dir` explicitly in tests; otherwise it resolves from `config.json`'s
`embedding_store` (default `~/project/embedder_store`) and you will clobber real indices.
`--embed-mode cpu` avoids the CUDA path on GPU-less boxes.

## Driving the MCP server over stdio

`mcp_server.py` reads `config.json` from **its own script directory**, not the cwd. To point it at a
scratch store without editing the repo, make a dir of symlinks plus a custom config:

```bash
mkdir /tmp/srv && cd /tmp/srv
for f in embedder.py mcp_server.py tree_search.py tree_ast_parser.py labels.json; do ln -sf <repo>/$f .; done
# config.json with "embedding_store": "/tmp/store"
```
Then `python mcp_server.py --project <name>` loads `/tmp/store/<name>/enriched_vectors.npz`.

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

## Known issue to watch for

`load_delta` extends `store.vectors/texts/node_ids` but does **not** rebuild the BM25 index, so any
`search` in `rrf`/`bm25`/alpha mode after a delta load raises
`IndexError: index N is out of bounds for axis 0 with size M`. Present on master too (not a
regression). If you are testing delta search, this will bite; a fix would call `_build_bm25()` at
the end of `load_delta` and `clear_delta`.

## Comparing against master

`git worktree add /tmp/master-wt master`, build a symlink server dir pointing at it, and run the
same query specs against the same `--data-dir` store to diff ranking output.
