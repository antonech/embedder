import json

import numpy as np
import pytest

import common
from common import iter_tree_index
import embedder
from embedder import (
    ASTParser,
    BodyStrategy,
    CompositeStrategy,
    ContentStrategy,
    DocstringStrategy,
    EmbeddingModel,
    KindStrategy,
    NameStrategy,
    SignatureStrategy,
    StatementsStrategy,
    StorageIO,
    StructureStrategy,
    VectorStore,
)


# --- Enrichment strategies ---

def test_name_and_kind_strategies():
    node = {"name": "run", "kind": "Function"}
    assert NameStrategy().enrich(node) == "run"
    assert KindStrategy().enrich(node) == "Function"
    assert NameStrategy().enrich({}) == ""
    assert KindStrategy().enrich({}) == ""
    assert NameStrategy.key() == "name" and KindStrategy.key() == "kind"


@pytest.mark.parametrize(
    "node,expected",
    [
        ({"signature": "(int x)"}, "(int x)"),
        ({"args": []}, ""),
        ({"args": ["a", "b"]}, "(a, b)"),
        ({"args": [{"name": "a"}, {"name": "b"}]}, "(a, b)"),
        ({"args": ["a"], "returns": "int"}, "(a) -> int"),
        ({"args": "self, x"}, "(self, x)"),
        ({}, ""),
    ],
)
def test_signature_strategy(node, expected):
    assert SignatureStrategy().enrich(node) == expected


def test_signature_strategy_ignores_non_string_signature():
    assert SignatureStrategy().enrich({"signature": 42, "args": ["a"]}) == "(a)"


def test_docstring_strategy_falls_back_to_doc_key():
    assert DocstringStrategy().enrich({"docstring": "d"}) == "d"
    assert DocstringStrategy().enrich({"doc": "alt"}) == "alt"
    assert DocstringStrategy().enrich({}) == ""


FULL_NODE = {
    "in_class": "Service",
    "methods": ["create", "delete", " "],
    "fields": ["width", ""],
    "bases": ["Base"],
    "body": ["line1", "  ", "line2"],
}


def test_structure_strategy_summarizes_shape_only():
    out = StructureStrategy().enrich(FULL_NODE)
    assert out == "In class: Service | Methods: create, delete | Fields: width | Inherits: Base"
    assert StructureStrategy().enrich({}) == ""
    assert StructureStrategy.key() == "structure"


def test_content_strategy_handles_string_bodies_only():
    assert ContentStrategy().enrich({"body": "x" * 300}) == "x" * 200
    assert ContentStrategy().enrich({"body": "   "}) == ""
    assert ContentStrategy().enrich({"body": ["a", "b"]}) == ""
    assert ContentStrategy().enrich({}) == ""
    assert ContentStrategy.key() == "content"


def test_statements_strategy_handles_list_bodies_only():
    assert StatementsStrategy().enrich(FULL_NODE) == "line1 | line2"
    assert StatementsStrategy().enrich({"body": "a string"}) == ""
    assert StatementsStrategy().enrich({}) == ""
    parts = StatementsStrategy().enrich(
        {"body": [f"line{i}" for i in range(10)]}).split(" | ")
    assert parts == [f"line{i}" for i in range(StatementsStrategy.MAX_LINES)]


def test_default_enrichment_excludes_statements():
    assert "statements" not in embedder.DEFAULT_ENRICHMENT
    out = CompositeStrategy.from_keys(embedder.DEFAULT_ENRICHMENT).enrich({
        "name": "run", "args": ["x"], "docstring": "Runs it.",
        "in_class": "Service", "body": ["secret = compute()"],
    })
    assert out == "(x) | In class: Service | Runs it."
    assert "secret" not in out


def test_body_strategy_is_legacy_alias_for_all_three():
    out = BodyStrategy().enrich(FULL_NODE)
    assert out == "In class: Service | Methods: create, delete | Fields: width | Inherits: Base | line1 | line2"
    assert BodyStrategy.key() == "body"


def test_body_strategy_with_string_body():
    assert BodyStrategy().enrich({"body": "x" * 300}) == "x" * 200
    assert BodyStrategy().enrich({"body": "   "}) == ""
    assert BodyStrategy().enrich({}) == ""


def test_composite_strategy_joins_non_empty_parts():
    comp = CompositeStrategy([KindStrategy(), NameStrategy(), DocstringStrategy()])
    assert comp.enrich({"kind": "Class", "name": "A"}) == "Class | A"
    assert CompositeStrategy.key() == "composite"


def test_composite_from_keys_ignores_unknown_keys():
    comp = CompositeStrategy.from_keys(["name", "nope", "kind"], sep="/")
    assert comp.enrich({"kind": "Class", "name": "A"}) == "A/Class"


def test_composite_default():
    comp = CompositeStrategy.default()
    out = comp.enrich({"kind": "Function", "name": "run", "args": ["x"], "docstring": "doc"})
    assert out == "Function | run | (x) | doc"


# --- ASTParser: Python ---

PY_SOURCE = '''
class Service(Base, mod.Mixin):
    """Service docstring."""

    def create(self, name):
        """Create."""
        return name


async def fetch(url):
    """Fetch."""
    x = 1
    return x
'''


def test_parse_python_classes_and_functions():
    chunks = ASTParser._parse_python(PY_SOURCE, "svc.py")
    by_name = {c["name"]: c for c in chunks}
    assert set(by_name) == {"Service", "create", "fetch"}
    svc = by_name["Service"]
    assert svc["kind"] == "Class"
    assert svc["file"] == "svc.py"
    assert svc["docstring"] == "Service docstring."
    assert svc["methods"] == ["create"]
    assert svc["bases"] == ["Base", "Mixin"]
    fetch = by_name["fetch"]
    assert fetch["kind"] == "Function"
    assert fetch["args"] == ["url"]
    assert fetch["in_class"] == ""
    # The docstring is not a body statement: it would duplicate DocstringStrategy.
    assert fetch["body"] == ["x = 1", "return x"]


def test_parse_python_indexes_methods_with_owning_class():
    """Methods need their own chunk; a name in the class's Methods: list is not findable."""
    by_name = {c["name"]: c for c in ASTParser._parse_python(PY_SOURCE, "svc.py")}
    create = by_name["create"]
    assert create["kind"] == "Method"
    assert create["in_class"] == "Service"
    assert create["args"] == ["self", "name"]
    assert create["docstring"] == "Create."
    assert create["body"] == ["return name"]


def test_python_body_lines_without_docstring():
    import ast

    fn = ast.parse("def f():\n    a = 1\n    b = 2\n    c = 3\n").body[0]
    assert ASTParser._python_body_lines(fn) == ["a = 1", "b = 2"]
    doc_only = ast.parse("def f():\n    'just a doc'\n").body[0]
    assert ASTParser._python_body_lines(doc_only) == []


def test_parse_python_syntax_error():
    chunks = ASTParser._parse_python("def broken(:\n", "bad.py")
    assert len(chunks) == 1
    assert chunks[0].startswith("[file] bad.py (parse error:")


# --- ASTParser: fallback ---

def test_parse_fallback_blocks():
    source = "short\n\n" + "a long enough paragraph line here\n" * 2 + "\ntrailing block that is long enough\n"
    chunks = ASTParser._parse_fallback(source, "notes.txt")
    default_label = embedder._LABELS["default"].get("file", "File")
    assert [c["kind"] for c in chunks] == [default_label, default_label]
    assert chunks[0]["file"] == "notes.txt"
    assert chunks[0]["body"].startswith("a long enough paragraph")
    assert chunks[1]["body"] == "trailing block that is long enough"


def test_parse_fallback_skips_short_blocks_and_uses_label():
    assert ASTParser._parse_fallback("tiny\n", "a.txt") == []
    chunks = ASTParser._parse_fallback("a" * 30 + "\n", "a.txt", "Block")
    assert chunks[0]["kind"] == "Block"


def test_parse_fallback_truncates_body():
    chunks = ASTParser._parse_fallback("z" * 900, "a.txt")
    assert len(chunks[0]["body"]) == 512


# --- ASTParser: dispatch / scanning ---

def test_handler_for_known_and_unknown_extensions():
    assert ASTParser._handler_for("a.py") is ASTParser._handlers["*.py"]
    assert ASTParser._handler_for("a.cpp") is ASTParser._handlers["*.cpp"]
    assert ASTParser._handler_for("a.unknown") == ASTParser._parse_fallback


def test_parse_file_reads_and_dispatches(tmp_path):
    path = tmp_path / "svc.py"
    path.write_text(PY_SOURCE)
    chunks = ASTParser.parse_file(str(path), path_hint="svc.py")
    assert {c["name"] for c in chunks} == {"Service", "create", "fetch"}


def test_parse_file_missing_file_returns_empty(tmp_path):
    assert ASTParser.parse_file(str(tmp_path / "nope.py")) == []


def test_parse_file_skips_empty_and_binary(tmp_path):
    empty = tmp_path / "empty.py"
    empty.write_text("   \n")
    assert ASTParser.parse_file(str(empty)) == []

    binary = tmp_path / "bin.dat"
    binary.write_bytes(b"abc\x00def")
    assert ASTParser.parse_file(str(binary)) == []


def test_load_strategy_reads_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"enrichment": ["name"], "use_clang": True}))
    try:
        strategy = ASTParser._load_strategy(str(tmp_path))
        assert strategy.enrich({"kind": "Class", "name": "A"}) == "A"
        assert ASTParser._use_clang is True

        # explicit keys win, but use_clang still comes from config
        strategy = ASTParser._load_strategy(str(tmp_path), ["kind"])
        assert strategy.enrich({"kind": "Class", "name": "A"}) == "Class"
        assert ASTParser._use_clang is True
    finally:
        ASTParser._use_clang = False


def test_load_strategy_without_config(tmp_path):
    strategy = ASTParser._load_strategy(str(tmp_path))
    assert ASTParser._use_clang is False
    assert strategy.enrich({"args": ["x"], "docstring": "d"}) == "(x) | d"
    strategy = ASTParser._load_strategy(str(tmp_path), ["name"])
    assert ASTParser._use_clang is False


def test_scan_project_survives_a_foreign_config_json(tmp_path):
    (tmp_path / "svc.py").write_text(PY_SOURCE)
    (tmp_path / "config.json").write_text('{\n  // dev server\n  "port": 3000\n}\n')

    chunks = ASTParser.scan_project(str(tmp_path))
    assert ASTParser._use_clang is False
    assert any("Class svc.py Service" in c for c in chunks)


def test_scan_project_prefixes_chunks_and_skips_dirs(tmp_path):
    (tmp_path / "svc.py").write_text(PY_SOURCE)
    (tmp_path / "venv").mkdir()
    (tmp_path / "venv" / "ignored.py").write_text("class Ignored:\n    pass\n")
    (tmp_path / "config.json").write_text(json.dumps({"enrichment": ["docstring"]}))

    chunks = ASTParser.scan_project(str(tmp_path))
    ASTParser._use_clang = False
    joined = "\n".join(chunks)
    assert "Class svc.py Service | Service docstring." in chunks
    assert "Ignored" not in joined


def test_scan_project_drops_skip_prefixed_strings(tmp_path):
    (tmp_path / "bad.py").write_text("def broken(:\n")
    assert ASTParser.scan_project(str(tmp_path), ["name"]) == []


def test_scan_project_handles_chunks_without_prefix(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("x = 1\n")
    monkeypatch.setattr(ASTParser, "parse_file",
                        classmethod(lambda cls, fp, path_hint="": [{"docstring": "solo"}]))
    assert ASTParser.scan_project(str(tmp_path), ["docstring"]) == ["solo"]


def test_scan_project_swallows_handler_errors(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("x = 1\n")
    monkeypatch.setattr(ASTParser, "parse_file",
                        classmethod(lambda cls, fp, path_hint="": (_ for _ in ()).throw(RuntimeError("boom"))))
    assert ASTParser.scan_project(str(tmp_path), ["name"]) == []


def test_register_adds_handler():
    sentinel = lambda source, path_hint="": ["chunk"]
    ASTParser.register("*.zzz", sentinel)
    try:
        assert ASTParser._handler_for("a.zzz") is sentinel
    finally:
        ASTParser._handlers.pop("*.zzz")


# --- ASTParser: tree-sitter handlers ---

JS_SOURCE = """
// greets
function greet(name) { return name; }

class Greeter {
    hello(x) { return x; }
}
"""


@pytest.mark.skipif(not embedder._TS_AVAILABLE, reason="tree-sitter not installed")
def test_parse_treesitter_javascript():
    chunks = ASTParser._parse_treesitter(JS_SOURCE, "app.js")
    by_name = {c["name"]: c for c in chunks}
    assert "greet" in by_name and "Greeter" in by_name
    assert by_name["greet"]["signature"] == "(name)"
    assert by_name["greet"]["file"] == "app.js"


@pytest.mark.skipif(not embedder._TS_AVAILABLE, reason="tree-sitter not installed")
def test_parse_treesitter_unknown_extension():
    assert ASTParser._parse_treesitter(JS_SOURCE, "app.txt") == []


@pytest.mark.skipif(not embedder._TS_AVAILABLE, reason="tree-sitter not installed")
def test_parse_treesitter_go_and_rust():
    go_chunks = ASTParser._parse_treesitter("func Add(a int) int { return a }\n", "m.go")
    assert [c["name"] for c in go_chunks] == ["Add"]
    rs_chunks = ASTParser._parse_treesitter("struct Point { x: i32 }\nfn area(p: Point) -> i32 { p.x }\n", "m.rs")
    assert {c["name"] for c in rs_chunks} == {"Point", "area"}


CPP_SOURCE = """
// A widget.
class Widget : public Base {
public:
    int compute(int x);
private:
    int width;
};

int add(int a, int b) { return a + b; }
"""


def test_parse_cpp_treesitter(monkeypatch):
    monkeypatch.setattr(ASTParser, "_use_clang", False)
    chunks = ASTParser._parse_cpp(CPP_SOURCE, "w.cpp")
    by_name = {c["name"]: c for c in chunks if isinstance(c, dict)}
    assert "Widget" in by_name and "add" in by_name
    assert by_name["Widget"]["bases"] == ["Base"]
    assert "compute" in by_name["Widget"]["body"]
    assert by_name["add"]["signature"] == "(int a, int b)"


RICH_CPP_SOURCE = """
class Rich : public Base {
public:
    Rich();
    virtual void run(int x);
    void inlined() {}
protected:
    int counter;
    double ratio;
};
"""


def test_parse_cpp_body_summary_covers_declaration_variants(monkeypatch):
    monkeypatch.setattr(ASTParser, "_use_clang", False)
    chunks = ASTParser._parse_cpp(RICH_CPP_SOURCE, "rich.cpp")
    rich = next(c for c in chunks if isinstance(c, dict) and c["name"] == "Rich")
    body = rich["body"]
    assert "public: run" in body
    assert "public: inlined()" in body
    assert "Fields: counter, ratio" in body


def test_parse_cpp_falls_back_when_treesitter_unavailable(monkeypatch):
    monkeypatch.setattr(ASTParser, "_use_clang", False)
    monkeypatch.setattr(embedder, "_TS_AVAILABLE", False)
    chunks = ASTParser._parse_cpp(CPP_SOURCE, "w.cpp")
    assert all(c["kind"] == "Block" for c in chunks)


def test_parse_cpp_empty_source_uses_fallback(monkeypatch):
    monkeypatch.setattr(ASTParser, "_use_clang", False)
    assert ASTParser._parse_cpp("int;\n", "w.cpp") == []


def test_parse_cpp_clang_without_clang(monkeypatch):
    """Returns nothing so _parse_cpp can pick a better fallback than raw lines."""
    monkeypatch.setattr(embedder, "_CLANG_AVAILABLE", False)
    assert ASTParser._parse_cpp_clang(CPP_SOURCE, "w.cpp") == []


def test_parse_cpp_degrades_to_treesitter_not_line_blocks(monkeypatch):
    """Clang needs real compile flags; failing it must not dump raw source lines."""
    monkeypatch.setattr(ASTParser, "_use_clang", True)
    monkeypatch.setattr(embedder, "_CLANG_AVAILABLE", True)
    monkeypatch.setattr(ASTParser, "_parse_cpp_clang", classmethod(lambda cls, s, p="": []))

    chunks = ASTParser._parse_cpp(CPP_SOURCE, "w.cpp")
    by_name = {c["name"]: c for c in chunks if isinstance(c, dict)}
    assert "Widget" in by_name and "add" in by_name
    assert not any(c.get("kind") == "Block" for c in chunks if isinstance(c, dict))


OUT_OF_LINE_CPP = """
struct HashMap {
    int find(int key);
};
int HashMap::find(int key) {
    int idx = hash(key) & mask_;
    return slots_[idx].value;
}
void Outer::Inner::run() {}
"""


def test_parse_cpp_indexes_out_of_line_definitions(monkeypatch):
    """int HashMap::find(...) names itself through a qualified_identifier."""
    monkeypatch.setattr(ASTParser, "_use_clang", False)
    chunks = [c for c in ASTParser._parse_cpp(OUT_OF_LINE_CPP, "m.cpp") if isinstance(c, dict)]
    by_name = {c["name"]: c for c in chunks}

    assert by_name["find"]["in_class"] == "HashMap"
    assert by_name["find"]["signature"] == "(int key)"
    assert by_name["run"]["in_class"] == "Inner"


def test_parse_cpp_skips_locals_inside_function_bodies(monkeypatch):
    """`int idx = ...` is a local, not API surface -- indexing it is noise."""
    monkeypatch.setattr(ASTParser, "_use_clang", False)
    chunks = [c for c in ASTParser._parse_cpp(OUT_OF_LINE_CPP, "m.cpp") if isinstance(c, dict)]
    assert "idx" not in {c["name"] for c in chunks}


# --- EmbeddingModel ---

@pytest.mark.parametrize(
    "model_name,query,passage",
    [
        ("intfloat/e5-small-v2", "query: ", "passage: "),
        ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
        ("all-MiniLM-L6-v2", "", ""),
    ],
)
def test_prefix_detection(model_name, query, passage):
    assert EmbeddingModel._detect_query_prefix(model_name) == query
    assert EmbeddingModel._detect_passage_prefix(model_name) == passage


def test_embedding_model_configures_model(fake_st):
    model = EmbeddingModel("intfloat/e5-small-v2")
    assert model.query_prefix == "query: "
    assert model.passage_prefix == "passage: "
    assert model.dim == fake_st.DIM
    assert model.model.max_seq_length == 512
    assert model.model.tokenizer.model_max_length == 512
    assert model.model.halved is False


def test_embedding_model_overrides_and_fp16(fake_st):
    model = EmbeddingModel("intfloat/e5-small-v2", query_prefix="", passage_prefix="P: ",
                           float_type="fp16")
    assert model.query_prefix == ""
    assert model.passage_prefix == "P: "
    assert model.model.halved is True


def test_embedding_model_respects_architecture_limit(fake_st, monkeypatch):
    class Limited(fake_st):
        def _first_module(self):
            return type("M", (), {"auto_model": type("A", (), {"config": type("C", (), {"max_position_embeddings": 128})()})()})()

    monkeypatch.setattr(embedder, "SentenceTransformer", Limited)
    assert EmbeddingModel("m").model.max_seq_length == 128


def test_embedding_model_embed_roundtrip(fake_st):
    model = EmbeddingModel("m")
    single = model.embed("hello")
    many = model.embed_many(["hello", "world"])
    assert single.shape == (fake_st.DIM,)
    assert many.shape == (2, fake_st.DIM)
    assert np.allclose(many[0], single)


# --- VectorStore ---

def _unit(*vals):
    vec = np.array(vals, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def test_vector_store_add_and_len():
    store = VectorStore()
    assert len(store) == 0
    assert store.search(_unit(1, 0)) == []

    store.add(_unit(1, 0), "a", node_id=7)
    store.add_many([_unit(0, 1), _unit(1, 1)], ["b", "c"])
    assert len(store) == 3
    assert store.node_ids == [7, None, None]


def test_vector_store_add_many_with_node_ids():
    store = VectorStore()
    store.add_many([_unit(1, 0)], ["a"], node_ids=[3])
    assert store.node_ids == [3]


def test_vector_store_search_ranks_by_cosine():
    store = VectorStore()
    store.add(_unit(1, 0), "a", node_id=1)
    store.add(_unit(0, 1), "b")
    hits = store.search(_unit(1, 0), top_k=2)
    assert [h["text"] for h in hits] == ["a", "b"]
    assert hits[0]["score"] == pytest.approx(1.0)
    assert hits[0]["node_id"] == 1 and hits[0]["idx"] == 0
    assert hits[0]["method"] == "embed"


def test_vector_store_array_cache_invalidated_on_add():
    store = VectorStore()
    store.add(_unit(1, 0), "a")
    assert store._get_array().shape == (1, 2)
    store.add(_unit(0, 1), "b")
    assert store._get_array().shape == (2, 2)


def test_vector_store_keeps_one_matrix():
    """vectors must not be a second copy of the search matrix."""
    store = VectorStore()
    store.vectors = np.stack([_unit(1, 0), _unit(0, 1)])
    store.texts = ["a", "b"]
    store.node_ids = [None, None]
    assert store.vectors is store._get_array()
    assert store.dim == 2 and len(store) == 2

    store.add_many(np.stack([_unit(1, 1)]), ["c"])
    assert len(store) == 3          # counted before the append is folded in
    assert store._get_array().shape == (3, 2)
    assert store.search(_unit(1, 1), top_k=1)[0]["text"] == "c"


def test_vector_store_truncate_drops_the_tail():
    store = VectorStore()
    store.add_many(np.stack([_unit(1, 0), _unit(0, 1)]), ["a", "b"], [7, None])
    store.truncate(1)
    assert len(store) == 1 and store.texts == ["a"] and store.node_ids == [7]
    assert store._get_array().shape == (1, 2)
    assert store.search(_unit(0, 1), top_k=2)[0]["text"] == "a"


def test_vector_store_empty_search_and_dim():
    store = VectorStore()
    assert len(store) == 0 and store.dim == 0
    assert store.search(_unit(1, 0)) == []


def test_vector_store_search_top_k_edges():
    # top_k selection partitions rather than sorting everything, so the boundaries
    # (k larger than the store, k == the store, k <= 0) need to stay well-defined.
    store = VectorStore()
    store.add_many(np.stack([_unit(1, 0), _unit(0, 1), _unit(1, 1)]), ["a", "b", "c"])

    assert [h["text"] for h in store.search(_unit(1, 0), top_k=99)] == ["a", "c", "b"]
    assert [h["text"] for h in store.search(_unit(1, 0), top_k=3)] == ["a", "c", "b"]
    assert [h["text"] for h in store.search(_unit(1, 0), top_k=1)] == ["a"]
    assert store.search(_unit(1, 0), top_k=0) == []


def test_vector_store_search_orders_by_score_descending():
    store = VectorStore()
    vecs = [_unit(1, 0), _unit(3, 1), _unit(1, 1), _unit(1, 3), _unit(0, 1)]
    store.add_many(np.stack(vecs), list("abcde"))
    hits = store.search(_unit(1, 0), top_k=4)
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert [h["text"] for h in hits] == ["a", "b", "c", "d"]


# --- StorageIO ---

def test_storage_roundtrip_with_node_ids(tmp_path):
    path = str(tmp_path / "vecs.npz")
    vectors = [_unit(1, 0), _unit(0, 1)]
    StorageIO.save(path, vectors, ["a", "b"], dim=2, node_ids=[5, None])
    loaded_vecs, texts, dim, node_ids = StorageIO.load(path)
    assert len(loaded_vecs) == 2
    assert np.allclose(loaded_vecs[0], vectors[0])
    assert texts == ["a", "b"]
    assert dim == 2
    assert node_ids == [5, None]


def test_storage_roundtrip_without_node_ids_and_from_array(tmp_path):
    path = str(tmp_path / "vecs.npz")
    StorageIO.save(path, np.stack([_unit(1, 0)]), ["a"], dim=2)
    vecs, texts, dim, node_ids = StorageIO.load(path)
    assert len(vecs) == 1 and texts == ["a"] and dim == 2
    assert node_ids is None


def test_storage_load_reports_non_npz_file(tmp_path):
    path = tmp_path / "notanindex.npz"
    path.write_text("plain text, not an archive")
    with pytest.raises(ValueError):
        StorageIO.load(str(path))


def test_storage_save_empty(tmp_path):
    path = str(tmp_path / "vecs.npz")
    StorageIO.save(path, [], [], dim=3)
    vecs, texts, dim, node_ids = StorageIO.load(path)
    assert len(vecs) == 0 and texts == [] and dim == 3 and node_ids is None


def test_storage_read_model_name_round_trips(tmp_path):
    path = str(tmp_path / "vecs.npz")
    StorageIO.save(path, np.stack([_unit(1, 0)]), ["a"], dim=2, model_name="e5-small-v2")
    assert StorageIO.read_model_name(path) == "e5-small-v2"


def test_storage_read_model_name_is_none_when_absent(tmp_path):
    path = str(tmp_path / "vecs.npz")
    StorageIO.save(path, np.stack([_unit(1, 0)]), ["a"], dim=2)
    assert StorageIO.read_model_name(path) is None


# --- build_flat_index ---

def _write_config(path, **overrides):
    cfg = {"model_name": "fake-model", "enrichment": ["docstring"]}
    cfg.update(overrides)
    path.write_text(json.dumps(cfg))


def test_build_flat_index_full(tmp_path, fake_st, capsys):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "svc.py").write_text(PY_SOURCE)
    _write_config(project / "config.json")

    embedder.build_flat_index(str(project), data_dir=str(project / "out"))
    ASTParser._use_clang = False

    out = project / "out" / "enriched_vectors.npz"
    assert out.exists()
    vecs, texts, dim, node_ids = StorageIO.load(str(out))
    assert any("Service" in t for t in texts)
    assert dim == fake_st.DIM
    assert node_ids == [None] * len(texts)


def test_build_flat_index_enriches_with_tree_context(tmp_path, fake_st):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "svc.py").write_text(PY_SOURCE)
    _write_config(project / "config.json")
    data_dir = project / "out"
    data_dir.mkdir()
    nodes = [
        {"id": 0, "parent_id": -1, "type": "class_definition", "name": "Service",
         "file": "svc.py", "start_line": 1, "end_line": 9, "text": "Class svc.py Service"},
        {"id": 1, "parent_id": 0, "type": "function_definition", "name": "create",
         "file": "svc.py", "start_line": 5, "end_line": 7, "text": "Function svc.py create"},
    ]
    (data_dir / "tree_index.json").write_text(json.dumps({"nodes": nodes, "texts": []}))

    try:
        embedder.build_flat_index(str(project), data_dir=str(data_dir))
    finally:
        ASTParser._use_clang = False

    _, texts, _, node_ids = StorageIO.load(str(data_dir / "enriched_vectors.npz"))
    service = next(t for t in texts if t.startswith("Class svc.py Service"))
    assert ". Children: create" in service
    assert any(nid is not None for nid in node_ids)


def test_build_flat_index_delta(tmp_path, fake_st, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "svc.py").write_text(PY_SOURCE)
    _write_config(project / "config.json")
    (project / "out").mkdir()

    monkeypatch.setattr(embedder, "changed_files", lambda _root: ["svc.py", "missing.py"])
    try:
        embedder.build_flat_index(str(project), data_dir=str(project / "out"), delta=True)
    finally:
        ASTParser._use_clang = False

    delta = json.loads((project / "out" / "delta_texts.json").read_text())
    assert delta["files"] == ["svc.py", "missing.py"]
    assert any("Service" in t for t in delta["texts"])
    assert (project / "out" / "delta.npz").exists()


def test_build_flat_index_delta_no_changes(tmp_path, fake_st, monkeypatch, capsys):
    project = tmp_path / "proj"
    project.mkdir()
    _write_config(project / "config.json")

    monkeypatch.setattr(embedder, "changed_files", lambda _root: [])
    try:
        # data_dir deliberately not pre-created: the empty-delta path must create it.
        embedder.build_flat_index(str(project), data_dir=str(project / "out"), delta=True)
    finally:
        ASTParser._use_clang = False

    assert "no changed files" in capsys.readouterr().out
    delta = json.loads((project / "out" / "delta_texts.json").read_text())
    assert delta == {"files": [], "texts": [], "model": "fake-model"}


def test_build_flat_index_delta_without_parseable_chunks(tmp_path, fake_st, monkeypatch, capsys):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "empty.py").write_text("\n")
    _write_config(project / "config.json")
    (project / "out").mkdir()

    monkeypatch.setattr(embedder, "changed_files", lambda _root: ["empty.py"])
    try:
        embedder.build_flat_index(str(project), data_dir=str(project / "out"), delta=True)
    finally:
        ASTParser._use_clang = False

    assert "no parseable chunks" in capsys.readouterr().out
    assert json.loads((project / "out" / "delta_texts.json").read_text())["texts"] == []


def test_build_flat_index_resolves_data_dir_from_embedder_config(tmp_path, fake_st, monkeypatch):
    script_dir = tmp_path / "script"
    script_dir.mkdir()
    store_root = tmp_path / "store"
    (script_dir / "config.json").write_text(json.dumps({
        "model_name": "fake-model", "embedding_store": str(store_root)}))
    project = tmp_path / "proj"
    project.mkdir()
    (project / "svc.py").write_text(PY_SOURCE)

    monkeypatch.setattr(common, "CONFIG_PATH", str(script_dir / "config.json"))
    embedder.build_flat_index(str(project))
    ASTParser._use_clang = False

    assert (store_root / "proj" / "enriched_vectors.npz").exists()


def test_relative_data_dir_is_cwd_relative_for_both_builders(tmp_path, fake_st, monkeypatch):
    """build_flat_index and build_all must agree on what a relative data_dir means."""
    project = tmp_path / "proj"
    project.mkdir()
    (project / "svc.py").write_text(PY_SOURCE)
    _write_config(project / "config.json")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    embedder.build_flat_index(str(project), data_dir="flat")
    embedder.build_all(str(project), data_dir="all", num_workers=1, embed_mode="cpu")
    ASTParser._use_clang = False

    assert (workdir / "flat" / "enriched_vectors.npz").exists()
    assert (workdir / "all" / "enriched_vectors.npz").exists()
    # Nothing may leak into the scanned project itself.
    assert not (project / "flat").exists()


# --- parallel parsing helpers ---

def test_parse_file_worker_returns_tree_and_flat(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    path = project / "svc.py"
    path.write_text(PY_SOURCE)
    from tree_ast_parser import LANGUAGES

    tree_nodes, flat_chunks, rel, errors = embedder._parse_file_worker(
        (str(path), "svc.py", str(project), tuple(LANGUAGES.keys()), set())
    )
    ASTParser._use_clang = False
    assert rel == "svc.py"
    assert errors == []
    assert [n["name"] for n in tree_nodes] == ["Service", "create", "fetch"]
    assert any("Service" in c for c in flat_chunks)


def test_parse_file_worker_honours_exclude(tmp_path):
    path = tmp_path / "venv" / "svc.py"
    path.parent.mkdir()
    path.write_text(PY_SOURCE)
    assert embedder._parse_file_worker((str(path), "venv/svc.py", str(tmp_path), (".py",), {"/venv/"})) == \
        ([], [], "venv/svc.py", [])


def test_parse_files_walks_project(tmp_path, capsys):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "svc.py").write_text(PY_SOURCE)
    (project / "node_modules").mkdir()
    (project / "node_modules" / "dep.py").write_text("class Dep:\n    pass\n")

    nodes, tree_texts, chunks = embedder._parse_files(str(project), num_workers=1)
    ASTParser._use_clang = False
    names = [n["name"] for n in nodes]
    assert "Service" in names and "Dep" not in names
    assert [n["id"] for n in nodes] == list(range(len(nodes)))
    assert len(tree_texts) == len(nodes)
    assert any("Service" in c for c in chunks)


def test_build_all_writes_tree_and_flat_indices(tmp_path, fake_st):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "svc.py").write_text(PY_SOURCE)
    _write_config(project / "config.json")
    data_dir = tmp_path / "out"

    embedder.build_all(str(project), data_dir=str(data_dir), num_workers=1, embed_mode="cpu")
    ASTParser._use_clang = False

    assert (data_dir / "tree_vectors.npz").exists()
    tree = list(iter_tree_index(str(data_dir / "tree_index.json")))
    assert [n["name"] for n, _text in tree] == ["Service", "create", "fetch"]

    _, texts, dim, node_ids = StorageIO.load(str(data_dir / "enriched_vectors.npz"))
    assert dim == fake_st.DIM
    assert any("Children: create" in t for t in texts)
    assert any(nid is not None for nid in node_ids)


def test_build_all_multi_mode_merges_worker_batches(tmp_path, fake_st, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "svc.py").write_text(PY_SOURCE)
    _write_config(project / "config.json", batch_size=1)
    data_dir = tmp_path / "out"

    embedder.build_all(str(project), data_dir=str(data_dir), num_workers=1, embed_mode="multi")
    ASTParser._use_clang = False

    vecs, texts, dim, _ = StorageIO.load(str(data_dir / "enriched_vectors.npz"))
    assert len(vecs) == len(texts) > 0
    assert dim == fake_st.DIM


def test_build_all_infers_cpu_mode_without_cuda(tmp_path, fake_st, monkeypatch, capsys):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    project = tmp_path / "proj"
    project.mkdir()
    (project / "svc.py").write_text(PY_SOURCE)
    _write_config(project / "config.json")

    embedder.build_all(str(project), data_dir=str(tmp_path / "out"), num_workers=1, embed_mode=None)
    ASTParser._use_clang = False
    assert "mode=cpu" in capsys.readouterr().out


def test_build_all_falls_back_to_embedder_config(tmp_path, fake_st, monkeypatch):
    script_dir = tmp_path / "script"
    script_dir.mkdir()
    store_root = tmp_path / "store"
    (script_dir / "config.json").write_text(json.dumps(
        {"model_name": "fake-model", "embedding_store": str(store_root), "batch_size": 2}))
    project = tmp_path / "proj"
    project.mkdir()
    (project / "svc.py").write_text(PY_SOURCE)

    monkeypatch.setattr(common, "CONFIG_PATH", str(script_dir / "config.json"))
    embedder.build_all(str(project), num_workers=1, embed_mode="cpu")
    ASTParser._use_clang = False
    assert (store_root / "proj" / "enriched_vectors.npz").exists()


def test_build_all_without_tree_nodes(tmp_path, fake_st, capsys):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "notes.md").write_text("a paragraph long enough to be chunked here\n")
    _write_config(project / "config.json")
    data_dir = tmp_path / "out"

    embedder.build_all(str(project), data_dir=str(data_dir), num_workers=1, embed_mode="cpu")
    ASTParser._use_clang = False
    assert "No tree nodes found" in capsys.readouterr().out
    assert not (data_dir / "tree_vectors.npz").exists()


def test_build_all_skips_existing_tree_index(tmp_path, fake_st, capsys):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "svc.py").write_text(PY_SOURCE)
    _write_config(project / "config.json")
    data_dir = tmp_path / "out"
    data_dir.mkdir()
    (data_dir / "tree_index.json").write_text(json.dumps({"nodes": [], "texts": []}))
    StorageIO.save(str(data_dir / "tree_vectors.npz"), [], [], 4)

    embedder.build_all(str(project), data_dir=str(data_dir), num_workers=1, embed_mode="cpu")
    ASTParser._use_clang = False
    assert "skipping tree embedding" in capsys.readouterr().out
