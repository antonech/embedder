import json

import pytest
from tree_sitter import Parser

import common
from common import iter_tree_index
import tree_ast_parser as tap

PY_SOURCE = '''
class Service:
    """Service docstring."""

    def create(self, name):
        """Create a thing."""
        return name


def helper(a, b=2):
    return a + b
'''

CPP_SOURCE = """
#include <vector>
#include "local.h"

// Base widget.
class Widget : public Base, public Other {
public:
    int compute(int x);
    void render() {}
private:
    int width;
};

template <typename T>
struct Holder {
    T value;
};
"""


def _parse(source: str, ext: str):
    parser = Parser(tap.LANGUAGES[ext])
    return parser.parse(source.encode()), source.encode()


def _find(node, type_name):
    if node.type == type_name:
        return node
    for child in node.children:
        found = _find(child, type_name)
        if found is not None:
            return found
    return None


def test_get_name():
    tree, src = _parse(PY_SOURCE, ".py")
    cls = _find(tree.root_node, "class_definition")
    assert tap.get_name(cls) == "Service"
    # a node without a `name` field yields the empty string
    assert tap.get_name(tree.root_node) == ""


def test_get_docstring_python_block():
    tree, src = _parse(PY_SOURCE, ".py")
    cls = _find(tree.root_node, "class_definition")
    assert tap.get_docstring(cls, src) == '"""Service docstring."""'


def test_get_docstring_leading_comment():
    tree, src = _parse(CPP_SOURCE, ".cpp")
    cls = _find(tree.root_node, "class_specifier")
    # class_specifier itself has no comment child; the whole file root does not either
    assert tap.get_docstring(cls, src) == ""


def test_get_docstring_without_children():
    tree, src = _parse("x = 1\n", ".py")
    ident = _find(tree.root_node, "identifier")
    assert tap.get_docstring(ident, src) == ""


def test_get_signature_python_and_cpp():
    tree, src = _parse(PY_SOURCE, ".py")
    fn = _find(tree.root_node, "function_definition")
    assert tap.get_signature(fn, src, ".py") == "(self, name)"

    ctree, csrc = _parse(CPP_SOURCE, ".cpp")
    cls = _find(ctree.root_node, "class_specifier")
    assert tap.get_signature(cls, csrc, ".py") == ""


def test_get_signature_cpp_declarator_fallback():
    tree, src = _parse("int add(int a, int b) { return a + b; }\n", ".cpp")
    fn = _find(tree.root_node, "function_definition")
    assert tap.get_signature(fn, src, "cpp") == "add(int a, int b)"


def test_get_base_classes():
    tree, src = _parse(CPP_SOURCE, ".cpp")
    cls = _find(tree.root_node, "class_specifier")
    bases, text = tap.ts_base_classes(cls)
    assert bases == ["Base", "Other"]
    assert text == ": Base, Other"

    ptree, _psrc = _parse(PY_SOURCE, ".py")
    pcls = _find(ptree.root_node, "class_definition")
    assert tap.ts_base_classes(pcls) == ([], "")


def test_get_includes():
    assert tap.get_includes(CPP_SOURCE.encode(), "cpp") == ["vector", "local.h"]
    assert tap.get_includes(CPP_SOURCE.encode(), "py") == []


def test_get_body_summary():
    tree, src = _parse(CPP_SOURCE, ".cpp")
    cls = _find(tree.root_node, "class_specifier")
    summary = tap.ts_body_summary(cls)
    assert summary == "Methods: public: compute(int x);, public: render(). Fields: width"


def test_get_body_summary_without_field_list():
    tree, src = _parse(PY_SOURCE, ".py")
    fn = _find(tree.root_node, "function_definition")
    assert tap.ts_body_summary(fn) == ""


def test_get_template_params():
    tree, src = _parse(CPP_SOURCE, ".cpp")
    tpl = _find(tree.root_node, "template_declaration")
    assert tap.get_template_params(tpl, src, "cpp") == "template <typename T>".split("template ")[1]
    assert tap.get_template_params(tpl, src, "py") == ""


def test_get_template_params_missing():
    tree, src = _parse("int x = 1;\n", ".cpp")
    assert tap.get_template_params(tree.root_node, src, "cpp") == ""


def test_collect_nodes_builds_parent_links(tmp_path):
    path = tmp_path / "svc.py"
    path.write_text(PY_SOURCE)
    tree, src = _parse(PY_SOURCE, ".py")
    nodes = []
    tap.collect_nodes(tree.root_node, src, str(path), ".py", nodes, root=str(tmp_path))

    by_name = {n["name"]: n for n in nodes}
    assert set(by_name) == {"Service", "create", "helper"}
    assert by_name["Service"]["parent_id"] == -1
    assert by_name["create"]["parent_id"] == by_name["Service"]["id"]
    assert by_name["helper"]["parent_id"] == -1
    assert by_name["create"]["file"] == "svc.py"
    assert by_name["create"]["signature"] == "(self, name)"
    assert "Create a thing." in by_name["create"]["docstring"]
    assert by_name["create"]["start_line"] == 5
    assert "Doc: " in by_name["create"]["text"]


def test_collect_nodes_cpp_extras(tmp_path):
    path = tmp_path / "w.cpp"
    path.write_text(CPP_SOURCE)
    tree, src = _parse(CPP_SOURCE, ".cpp")
    nodes = []
    tap.collect_nodes(tree.root_node, src, str(path), ".cpp", nodes,
                      root=str(tmp_path), file_includes=["vector"])
    widget = next(n for n in nodes if n["name"] == "Widget")
    assert widget["bases"] == ["Base", "Other"]
    assert widget["includes"] == ["vector"]
    assert ": Base, Other" in widget["text"]
    assert "{ Methods:" in widget["text"]


def test_parse_file_unknown_extension(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# hello\n")
    assert tap.parse_file(str(path)) == []


def test_parse_file_python(tmp_path):
    path = tmp_path / "svc.py"
    path.write_text(PY_SOURCE)
    nodes = tap.parse_file(str(path), root=str(tmp_path))
    assert [n["name"] for n in nodes] == ["Service", "create", "helper"]
    assert [n["id"] for n in nodes] == [0, 1, 2]


def test_parse_file_shares_id_counter(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("def one():\n    pass\n")
    b.write_text("def two():\n    pass\n")
    next_id = [0]
    first = tap.parse_file(str(a), next_id=next_id, root=str(tmp_path))
    second = tap.parse_file(str(b), next_id=next_id, root=str(tmp_path))
    assert first[0]["id"] == 0
    assert second[0]["id"] == 1


def test_parse_file_collects_includes(tmp_path):
    path = tmp_path / "w.cpp"
    path.write_text(CPP_SOURCE)
    nodes = tap.parse_file(str(path), root=str(tmp_path))
    assert all(n["includes"] == ["vector", "local.h"] for n in nodes)


def test_resolve_data_dir_explicit():
    assert tap.resolve_data_dir("/proj", "/tmp/out") == "/tmp/out"


def test_resolve_data_dir_from_config(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"embedding_store": str(tmp_path / "store")}))
    monkeypatch.setattr(common, "CONFIG_PATH", str(cfg))
    assert tap.resolve_data_dir("/some/proj", None) == str(tmp_path / "store" / "proj")


def test_resolve_data_dir_missing_config(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "CONFIG_PATH", str(tmp_path / "config.json"))
    assert tap.resolve_data_dir("/some/proj", None) == "data"


def test_resolve_data_dir_config_without_store(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"model_name": "x"}))
    monkeypatch.setattr(common, "CONFIG_PATH", str(cfg))
    assert tap.resolve_data_dir("/some/proj", None, default="") == ""


class _StubModel:
    dim = 3

    def embed_many(self, texts):
        import numpy as np

        return np.zeros((len(texts), self.dim), dtype=np.float32)


def test_build_index_writes_tree_artifacts(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "svc.py").write_text(PY_SOURCE)
    (project / "notes.md").write_text("ignored\n")
    data_dir = tmp_path / "out"

    monkeypatch.setattr(tap, "EmbeddingModel", lambda *a, **kw: _StubModel())
    model, store, nodes = tap.build_index(root=str(project), data_dir=str(data_dir))

    assert [n["name"] for n in nodes] == ["Service", "create", "helper"]
    assert len(store.texts) == 3
    assert (data_dir / "tree_vectors.npz").exists()
    saved = list(iter_tree_index(str(data_dir / "tree_index.json")))
    assert len(saved) == 3
    assert [text for _node, text in saved] == store.texts


def test_build_index_skips_unparseable_files(tmp_path, monkeypatch, capsys, caplog):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "ok.py").write_text("def f():\n    pass\n")
    (project / "bad.py").write_text("def g():\n    pass\n")
    data_dir = tmp_path / "out"

    real_parse = tap.parse_file

    def flaky(fp, **kwargs):
        if fp.endswith("bad.py"):
            raise RuntimeError("boom")
        return real_parse(fp, **kwargs)

    monkeypatch.setattr(tap, "parse_file", flaky)
    monkeypatch.setattr(tap, "EmbeddingModel", lambda *a, **kw: _StubModel())
    _, _, nodes = tap.build_index(root=str(project), data_dir=str(data_dir))

    assert [n["name"] for n in nodes] == ["f"]
    assert "WARNING: skipped 1/2 files" in capsys.readouterr().out
    assert "bad.py: parse failed" in caplog.text


def test_build_delta_without_changes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tap, "changed_files", lambda _root: [])
    assert tap.build_delta(root=str(tmp_path), data_dir=str(tmp_path / "out")) is None
    assert "No changed files" in capsys.readouterr().out


def test_build_delta_writes_delta_artifacts(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "svc.py").write_text(PY_SOURCE)
    (project / "notes.md").write_text("ignored\n")
    data_dir = tmp_path / "out"

    monkeypatch.setattr(tap, "changed_files", lambda _root: ["svc.py", "notes.md", "missing.py"])
    monkeypatch.setattr(tap, "EmbeddingModel", lambda *a, **kw: _StubModel())
    monkeypatch.chdir(tmp_path)
    tap.build_delta(root=str(project), data_dir=str(data_dir))

    assert (data_dir / "delta_tree_vectors.npz").exists()
    saved = list(iter_tree_index(str(data_dir / "delta_tree_index.json")))
    assert [n["name"] for n, _text in saved] == ["Service", "create", "helper"]


def test_build_delta_raises_when_every_file_fails(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "bad.py").write_text("def g():\n    pass\n")
    data_dir = tmp_path / "out"

    monkeypatch.setattr(tap, "changed_files", lambda _root: ["bad.py"])
    monkeypatch.setattr(tap, "parse_file", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(tap, "EmbeddingModel", lambda *a, **kw: _StubModel())
    with pytest.raises(RuntimeError, match="all 1 changed files failed to parse"):
        tap.build_delta(root=str(project), data_dir=str(data_dir))

    assert not (data_dir / "delta_tree_index.json").exists()


def test_build_delta_skips_failing_file(tmp_path, monkeypatch, capsys):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "ok.py").write_text("def f():\n    pass\n")
    (project / "bad.py").write_text("def g():\n    pass\n")
    data_dir = tmp_path / "out"

    real_parse = tap.parse_file

    def flaky(fp, **kwargs):
        if fp.endswith("bad.py"):
            raise RuntimeError("boom")
        return real_parse(fp, **kwargs)

    monkeypatch.setattr(tap, "changed_files", lambda _root: ["ok.py", "bad.py"])
    monkeypatch.setattr(tap, "parse_file", flaky)
    monkeypatch.setattr(tap, "EmbeddingModel", lambda *a, **kw: _StubModel())
    tap.build_delta(root=str(project), data_dir=str(data_dir))

    assert "WARNING: skipped 1/2 changed files" in capsys.readouterr().out
    names = [n["name"] for n, _text in iter_tree_index(str(data_dir / "delta_tree_index.json"))]
    assert names == ["f"]
