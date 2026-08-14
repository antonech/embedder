import json

import numpy as np
import pytest

import mcp_server
from conftest import FakeSentenceTransformer
from embedder import StorageIO


class FakeEncoder:
    """Stand-in for EmbeddingModel (deterministic, no model download)."""

    def __init__(self, model_name="fake", device=None, float_type="fp32"):
        self.model_name = model_name
        self.device = device
        self.float_type = float_type
        self.dim = FakeSentenceTransformer.DIM
        self.query_prefix = ""
        self.passage_prefix = ""
        self._st = FakeSentenceTransformer(model_name, device)

    def embed(self, text):
        return self._st.encode(text)

    def embed_many(self, texts):
        return self._st.encode(list(texts))

    def as_passages(self, texts):
        return [self.passage_prefix + t for t in texts] if self.passage_prefix else list(texts)

    def embed_passage(self, text):
        return self.embed(self.passage_prefix + text if self.passage_prefix else text)

    def embed_query(self, query):
        return self.embed(self.query_prefix + query if self.query_prefix else query)


@pytest.fixture
def fake_encoder(monkeypatch):
    monkeypatch.setattr(mcp_server, "EmbeddingModel", FakeEncoder)
    return FakeEncoder


@pytest.fixture
def app(fake_encoder, tmp_path):
    return mcp_server.EmbedderApp("proj", "fake-model", data_dir=str(tmp_path))


TEXTS = [
    "Class svc.py Service | user service",
    "Function svc.py create_user | creates a user record",
    "Function svc.py delete_user | deletes a user record",
    "Function util.py hash_table_lookup | lookup in a hash table",
]


@pytest.fixture
def store_path(tmp_path, fake_encoder):
    enc = FakeEncoder()
    vecs = enc.embed_many(TEXTS)
    path = tmp_path / "enriched_vectors.npz"
    StorageIO.save(str(path), vecs, TEXTS, enc.dim, node_ids=[0, 1, None, None])
    return path


@pytest.fixture
def tree_dir(store_path, fake_encoder):
    data_dir = store_path.parent
    nodes = [
        {"id": 0, "parent_id": -1, "type": "class_definition", "name": "Service",
         "file": "svc.py", "start_line": 1, "end_line": 30, "text": "Class svc.py Service"},
        {"id": 1, "parent_id": 0, "type": "function_definition", "name": "create_user",
         "file": "svc.py", "start_line": 4, "end_line": 8, "text": "Function svc.py create_user"},
        {"id": 2, "parent_id": 0, "type": "function_definition", "name": "delete_user",
         "file": "svc.py", "start_line": 9, "end_line": 12, "text": "Function svc.py delete_user"},
    ]
    tree_texts = [n["text"] for n in nodes]
    (data_dir / "tree_index.json").write_text(json.dumps({"nodes": nodes, "texts": tree_texts}))
    enc = FakeEncoder()
    StorageIO.save(str(data_dir / "tree_vectors.npz"), enc.embed_many(tree_texts), tree_texts, enc.dim)
    return data_dir


# --- construction ---

def test_cuda_falls_back_to_cpu_without_gpu(fake_encoder, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    app = mcp_server.EmbedderApp("proj", "fake-model", device="cuda")
    assert app.device == "cpu"


def test_cross_encoder_load_failure_is_reported(fake_encoder, caplog, monkeypatch):
    import transformers

    def boom(*_a, **_kw):
        raise OSError("model not found")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", boom)
    app = mcp_server.EmbedderApp("proj", "fake-model", cross_encoder_model="does/not-exist")
    assert app.cross_encoder is None
    assert "failed to load cross-encoder" in caplog.text


# --- tokenization ---

@pytest.mark.parametrize(
    "text,expected",
    [
        ("hash_table lookup", ["hash", "table", "lookup"]),
        ("HashTableLookup", ["hash", "table", "lookup"]),
        ("get2Items", ["get", "2", "items"]),
        ("", []),
        ("...", []),
    ],
)
def test_tokenize(app, text, expected):
    assert app._tokenize(text) == expected


# --- store lifecycle ---

def test_init_loads_store_and_bm25(app, store_path):
    msg = app.init(str(store_path))
    assert msg == f"Loaded {len(TEXTS)} vectors, dim={FakeSentenceTransformer.DIM}"
    assert app.store.texts == TEXTS
    assert app.store.node_ids == [0, 1, None, None]
    assert app._bm25 is not None
    assert app.data_dir == str(store_path.parent)


def test_init_resets_cached_tree(app, tree_dir, store_path):
    app.init(str(store_path))
    assert app._get_tree() is not None
    app.init(str(store_path))
    assert not hasattr(app, "_tree")


def test_init_invalidates_tree_vectors_and_flat_map(app, tree_dir, store_path, fake_encoder):
    """A reloaded store must not keep the previous tree->flat map.

    The map is keyed by flat index, so a stale entry would boost whatever chunk
    now happens to sit at that index.
    """
    app.init(str(store_path))
    assert app._get_tree_store() is not None
    assert app._tree_to_flat == {0: 0, 1: 1}

    other = tree_dir / "other.npz"
    enc = FakeEncoder()
    texts = ["Function other.py unrelated | nothing to do with Service"]
    StorageIO.save(str(other), enc.embed_many(texts), texts, enc.dim, node_ids=[None])
    app.init(str(other))

    assert not hasattr(app, "_tree_store")
    assert not hasattr(app, "_tree_to_flat")
    app._get_tree_store()
    assert app._tree_to_flat == {}


def test_loads_index_from_another_directory(app, store_path, tmp_path, fake_encoder):
    """Any readable .npz is fair game, including another project's index."""
    other = tmp_path / "elsewhere" / "enriched_vectors.npz"
    other.parent.mkdir()
    enc = FakeEncoder()
    texts = ["Function other.py sibling_fn | from another project"]
    StorageIO.save(str(other), enc.embed_many(texts), texts, enc.dim)

    app.init(str(store_path))
    assert app.init(str(other)).startswith("Loaded")
    assert app.store.texts == texts
    # data_dir follows the loaded index so its tree files resolve alongside it.
    assert app.data_dir == str(tmp_path / "elsewhere")


def test_bare_filename_means_the_served_directory(app, store_path):
    app.init(str(store_path))
    assert app.init("enriched_vectors.npz").startswith("Loaded")


def test_missing_index_reports_an_error_instead_of_raising(monkeypatch, app, tmp_path):
    monkeypatch.setattr(mcp_server, "projects", {"proj": app})
    out = _fn(mcp_server.init_store)(str(tmp_path / "nope.npz"), "proj")
    assert out.startswith("Error: ")


def test_build_bm25_noop_on_empty_store(app):
    app._build_bm25()
    assert app._bm25 is None


def test_delta_load_and_clear(app, store_path, tmp_path, fake_encoder):
    app.init(str(store_path))
    enc = FakeEncoder()
    delta_texts = ["Function new.py added_fn | brand new"]
    delta_path = tmp_path / "delta.npz"
    StorageIO.save(str(delta_path), enc.embed_many(delta_texts), delta_texts, enc.dim)

    assert app.load_delta(str(delta_path)) == "Loaded 1 delta vectors"
    assert len(app.store) == len(TEXTS) + 1
    assert app.clear_delta() == "Delta cleared"
    assert app.store.texts == TEXTS
    assert app.clear_delta() == "No delta to clear"


def test_load_delta_missing_file(app, tmp_path):
    with pytest.raises(FileNotFoundError):
        app.load_delta(str(tmp_path / "nope.npz"))


def test_load_delta_rejects_dimension_mismatch(app, store_path, tmp_path):
    app.init(str(store_path))
    delta_texts = ["Function new.py added_fn | brand new"]
    delta_path = tmp_path / "delta.npz"
    StorageIO.save(str(delta_path), np.ones((1, 8), dtype=np.float32), delta_texts, 8)

    with pytest.raises(ValueError, match="does not match the store"):
        app.load_delta(str(delta_path))
    assert app.store.texts == TEXTS
    assert app.search("user", top_k=1)


# --- formatting ---

def test_format_text_and_json():
    hits = [{"text": "Class svc.py Service", "score": 0.5}]
    assert mcp_server.EmbedderApp._format(hits) == "1. [0.500] Class svc.py Service"
    assert json.loads(mcp_server.EmbedderApp._format(hits, "json")) == hits


def test_format_markdown_with_context():
    hits = [{
        "text": "Class svc.py Service",
        "score": 0.75,
        "context": {
            "parent": {"name": "Module", "type": "module"},
            "children": [{"name": f"c{i}"} for i in range(6)],
            "siblings": [{"name": f"s{i}"} for i in range(6)],
        },
    }]
    out = mcp_server.EmbedderApp._format(hits, "markdown")
    assert "### 1. [0.750] Class svc.py Service" in out
    assert "**Parent:** Module (module)" in out
    assert "**Children:** c0, c1, c2, c3, c4..." in out
    assert "**Siblings:** s0, s1, s2, s3, s4..." in out


def test_format_markdown_without_context():
    out = mcp_server.EmbedderApp._format([{"text": "t", "score": 0.1}], "markdown")
    assert out == "### 1. [0.100] t"


# --- tree fusion / annotation ---

def test_fuse_with_tree_without_tree_store(app, store_path):
    app.init(str(store_path))
    hits = app.store.search(app._embed_query("user"), top_k=2)
    assert app._fuse_with_tree(hits, app._embed_query("user"), top_k=2) == hits


def test_fuse_with_tree_boosts_and_adds_hits(app, tree_dir, store_path):
    app.init(str(store_path))
    qv = app._embed_query("create user")
    flat_hits = app.store.search(qv, top_k=2)
    fused = app._fuse_with_tree(flat_hits, qv, top_k=4)
    assert len(fused) <= 4
    assert fused == sorted(fused, key=lambda h: -h["score"])
    assert {h["method"] for h in fused} <= {"embed", "tree", "tree_boosted", "reranked"}
    # delete_user has no node_id in the flat store, so its tree node stays standalone
    assert any(h["method"] in ("tree", "tree_boosted") for h in fused)


def test_annotate_uses_node_id_then_text(app, tree_dir, store_path):
    app.init(str(store_path))
    hits = [
        {"text": "irrelevant", "node_id": 1},
        {"text": "Class svc.py Service", "node_id": None},
        {"text": "Function nothing.py missing", "node_id": None},
    ]
    out = app._annotate(hits)
    assert out[0]["context"]["parent"] == {"name": "Service", "type": "class_definition"}
    assert [c["name"] for c in out[1]["context"]["children"]] == ["create_user", "delete_user"]
    assert out[1]["context"]["parent"] is None
    assert "context" not in out[2]


# --- reranking ---

class FakeCrossEncoder:
    device = "cpu"

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def __call__(self, **inputs):
        import torch

        self.calls.append(inputs)
        return type("Out", (), {"logits": torch.tensor(self.scores).unsqueeze(-1)})()


def _install_cross_encoder(app, scores):
    app.cross_encoder = FakeCrossEncoder(scores)
    app.cross_encoder_tokenizer = lambda pairs, padding, truncation, return_tensors: {}
    return app.cross_encoder


def test_rerank_sorts_and_truncates(app):
    _install_cross_encoder(app, [-5.0, 5.0])
    hits = [{"text": "a", "score": 0.9}, {"text": "b", "score": 0.1}]
    out = app._rerank("q", hits, top_k=1)
    assert [h["text"] for h in out] == ["b"]
    assert out[0]["method"] == "reranked"
    assert 0.9 < out[0]["score"] <= 1.0


def test_rerank_without_model_or_hits(app):
    hits = [{"text": "a", "score": 0.9}]
    assert app._rerank("q", hits, 1) is hits
    _install_cross_encoder(app, [1.0])
    assert app._rerank("q", [], 1) == []


def test_rerank_single_hit(app):
    _install_cross_encoder(app, [2.0])
    out = app._rerank("q", [{"text": "a", "score": 0.1}], top_k=5)
    assert len(out) == 1 and out[0]["score"] > 0.5


# --- query expansion & search ---

def test_expand_query_without_bm25(app):
    assert app._expand_query("hash lookup", np.zeros(4)) == ["hash", "lookup"]


def test_expand_query_adds_related_terms(app, store_path):
    app.init(str(store_path))
    qv = app._embed_query("user")
    expanded = app._expand_query("user", qv, top_k=3, max_terms=3)
    assert expanded[:1] == ["user"]
    assert len(expanded) > 1
    assert "user" not in expanded[1:]


def test_embed_query_applies_prefix(app):
    app.encoder.query_prefix = "query: "
    assert np.allclose(app._embed_query("x"), app.encoder.embed("query: x"))


@pytest.mark.parametrize("mode", ["embed", "bm25", "rrf"])
def test_search_modes_return_results(app, store_path, mode):
    app.init(str(store_path))
    out = app.search("create user", top_k=2, mode=mode)
    assert out
    assert out.splitlines()[0].startswith("1. [")


def test_search_embed_mode_without_bm25(app, store_path):
    app.init(str(store_path))
    app._bm25 = None
    out = app.search("create user", top_k=2, mode="rrf", fmt="json")
    hits = json.loads(out)
    assert len(hits) == 2
    assert all(h["method"] == "embed" for h in hits)


def test_search_alpha_one_is_embed_only(app, store_path):
    app.init(str(store_path))
    hits = json.loads(app.search("user", top_k=2, alpha=1.0, fmt="json"))
    assert all(h["method"] == "embed" for h in hits)


@pytest.mark.parametrize("mode", ["bm25", "rrf"])
def test_search_after_load_delta(app, store_path, tmp_path, fake_encoder, mode):
    app.init(str(store_path))
    enc = FakeEncoder()
    delta_texts = ["Function new.py added_fn | brand new"]
    delta_path = tmp_path / "delta.npz"
    StorageIO.save(str(delta_path), enc.embed_many(delta_texts), delta_texts, enc.dim)
    app.load_delta(str(delta_path))

    assert app.search("user", top_k=2, mode=mode)


def test_search_alpha_zero_is_bm25_only(app, store_path):
    app.init(str(store_path))
    bm25_only = json.loads(app.search("user", top_k=2, alpha=0.0, fmt="json"))
    assert all(h["method"] == "bm25" for h in bm25_only)
    assert all(0.0 <= h["score"] <= 1.0 for h in bm25_only)


def test_search_uses_reranker_when_available(app, store_path):
    app.init(str(store_path))
    ce = _install_cross_encoder(app, [0.0] * 16)
    hits = json.loads(app.search("user", top_k=2, fmt="json"))
    assert ce.calls
    assert all(h["method"] == "reranked" for h in hits)

    ce.calls.clear()
    json.loads(app.search("user", top_k=2, fmt="json", rerank=False))
    assert not ce.calls


def test_search_with_tree_fusion_and_annotation(app, tree_dir, store_path):
    app.init(str(store_path))
    hits = json.loads(app.search("create user", top_k=3, mode="embed", fmt="json"))
    assert any("context" in h for h in hits)


# --- store mutation / info ---

def test_embed_and_embed_many(app):
    assert len(app.embed("hello")) == FakeSentenceTransformer.DIM
    assert np.array(app.embed_many(["a", "b"])).shape == (2, FakeSentenceTransformer.DIM)


def test_add_document_and_documents(app):
    app.encoder.passage_prefix = "passage: "
    assert app.add_document("one") == "Added, total vectors: 1"
    assert app.add_documents(["two", "three"]) == "Added 2 docs, total vectors: 3"
    assert app.store.texts == ["one", "two", "three"]

    app.encoder.passage_prefix = ""
    app.add_document("four")
    app.add_documents(["five"])
    assert app.store.texts[-2:] == ["four", "five"]


def test_add_defers_bm25_rebuild_until_search(app, store_path):
    app.init(str(store_path))
    app.add_document("Function extra.py brand_new_helper | freshly added")
    app.add_documents(["Function extra.py second_helper | also new"])
    assert app._bm25_dirty

    hits = json.loads(app.search("brand new helper", top_k=3, mode="bm25", fmt="json"))
    assert not app._bm25_dirty
    # The deferred rebuild must cover the added docs, not just the loaded index.
    assert len(app._bm25.idf) > 0
    assert any("brand_new_helper" in h["text"] for h in hits)


def test_save_and_info(app, tmp_path):
    app.add_documents(["one", "two"])
    path = tmp_path / "out.npz"
    assert app.save(str(path)) == f"Saved 2 vectors to {path}"
    vecs, texts, dim, _ = StorageIO.load(str(path))
    assert texts == ["one", "two"] and dim == FakeSentenceTransformer.DIM

    info = json.loads(app.info())
    assert info == {"vectors": 2, "delta": 0, "sample_texts": ["one", "two"]}


def test_save_roundtrips_node_ids(app, store_path, tmp_path):
    app.init(str(store_path))
    path = tmp_path / "resaved.npz"
    app.save(str(path))
    assert app.init(str(path)).startswith("Loaded")
    assert app.store.node_ids == [0, 1, None, None]


def test_info_on_empty_store(app):
    assert json.loads(app.info()) == {"vectors": 0, "delta": 0, "sample_texts": []}


# --- MCP tool wrappers ---

TOOL_CALLS = [
    (mcp_server.search, ("q",), {}),
    (mcp_server.embed, ("t",), {}),
    (mcp_server.embed_many, (["t"],), {}),
    (mcp_server.store_info, (), {}),
    (mcp_server.init_store, ("p.npz",), {}),
    (mcp_server.add_document, ("t",), {}),
    (mcp_server.add_documents, (["t"],), {}),
    (mcp_server.save_store, ("p.npz",), {}),
    (mcp_server.load_delta, ("p.npz",), {}),
    (mcp_server.clear_delta, (), {}),
]


def _fn(tool):
    return tool.fn if hasattr(tool, "fn") else tool


@pytest.mark.parametrize("tool,args,kwargs", TOOL_CALLS)
def test_tools_require_initialized_server(monkeypatch, tool, args, kwargs):
    monkeypatch.setattr(mcp_server, "projects", {})
    assert _fn(tool)(*args, project="proj", **kwargs) == "Error: server not initialized"


@pytest.mark.parametrize("tool,args,kwargs", TOOL_CALLS)
def test_tools_reject_unknown_project(monkeypatch, app, tool, args, kwargs):
    monkeypatch.setattr(mcp_server, "projects", {"proj": app})
    assert _fn(tool)(*args, project="other", **kwargs) == "Error: project 'other' not found"


def test_tools_delegate_to_app(monkeypatch, app, store_path, tmp_path):
    monkeypatch.setattr(mcp_server, "projects", {"proj": app})
    assert _fn(mcp_server.init_store)(str(store_path), "proj").startswith("Loaded")
    assert _fn(mcp_server.search)("user", "proj", top_k=1)
    assert len(json.loads(_fn(mcp_server.embed)("x", "proj"))) == FakeSentenceTransformer.DIM
    assert len(json.loads(_fn(mcp_server.embed_many)(["x", "y"], "proj"))) == 2
    assert json.loads(_fn(mcp_server.store_info)("proj"))["vectors"] == len(TEXTS)
    assert _fn(mcp_server.add_document)("new doc", "proj").startswith("Added,")
    assert _fn(mcp_server.add_documents)(["a", "b"], "proj").startswith("Added 2 docs")
    assert _fn(mcp_server.save_store)(str(tmp_path / "saved.npz"), "proj").startswith("Saved")
    # Tools report failures as strings rather than raising, so the MCP layer keeps
    # returning a well-formed result instead of surfacing a traceback.
    assert _fn(mcp_server.load_delta)(str(tmp_path / "missing.npz"), "proj").startswith(
        "Error: delta file not found"
    )
    assert _fn(mcp_server.clear_delta)("proj") == "No delta to clear"
