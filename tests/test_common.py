import json

import pytest

import common


def test_load_json_missing_and_invalid(tmp_path):
    assert common.load_json(str(tmp_path / "nope.json")) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(RuntimeError):
        common.load_json(str(bad))
    not_object = tmp_path / "list.json"
    not_object.write_text("[1, 2]")
    with pytest.raises(RuntimeError, match="expected a JSON object"):
        common.load_json(str(not_object))


def test_load_project_config_ignores_foreign_files(tmp_path, caplog):
    assert common.load_project_config(str(tmp_path / "nope.json")) == {}

    commented = tmp_path / "commented.json"
    commented.write_text('{\n  // dev server\n  "port": 3000\n}\n')
    with caplog.at_level("WARNING"):
        assert common.load_project_config(str(commented)) == {}
    assert "ignoring" in caplog.text

    other_app = tmp_path / "other.json"
    other_app.write_text(json.dumps({"port": 3000}))
    assert common.load_project_config(str(other_app)) == {}

    a_list = tmp_path / "list.json"
    a_list.write_text("[]")
    assert common.load_project_config(str(a_list)) == {}

    ours = tmp_path / "ours.json"
    ours.write_text(json.dumps({"model_name": "m", "port": 3000}))
    assert common.load_project_config(str(ours)) == {"model_name": "m", "port": 3000}


def test_load_labels_falls_back_to_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "LABELS_PATH", str(tmp_path / "labels.json"))
    assert common.load_labels() == common.DEFAULT_LABELS


def test_label_for_uses_mapping_then_default():
    assert common.label_for("class_definition") in (
        common.LABELS["mapping"].get("class_definition"), "class_definition")
    assert common.label_for("no_such_type", "fallback") == "fallback"


def test_expand_path(monkeypatch):
    monkeypatch.setenv("EMB_STORE", "/srv/store")
    assert common.expand_path("$EMB_STORE/x") == "/srv/store/x"


def test_resolve_device_without_torch(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    assert common.resolve_device("cuda:0") == "cpu"
    assert common.resolve_device("cpu") == "cpu"
    assert common.resolve_device(None) is None


def test_model_config_load_prefers_project_config(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"model_name": "proj-model", "batch_size": 7}))
    cfg = common.ModelConfig.load(str(tmp_path))
    assert (cfg.model_name, cfg.batch_size) == ("proj-model", 7)

    fallback = tmp_path / "fallback.json"
    fallback.write_text(json.dumps({"model_name": "own-model"}))
    monkeypatch.setattr(common, "CONFIG_PATH", str(fallback))
    assert common.ModelConfig.load(str(tmp_path / "elsewhere")).model_name == "own-model"


def test_model_config_load_falls_back_when_project_config_is_foreign(tmp_path, monkeypatch):
    fallback = tmp_path / "fallback.json"
    fallback.write_text(json.dumps({"model_name": "own-model"}))
    monkeypatch.setattr(common, "CONFIG_PATH", str(fallback))

    project = tmp_path / "proj"
    project.mkdir()
    (project / "config.json").write_text('{\n  // dev server\n  "port": 3000\n}\n')
    assert common.ModelConfig.load(str(project)).model_name == "own-model"


def _cpp_root(src: str):
    from tree_sitter import Language, Parser
    import tree_sitter_cpp

    return Parser(Language(tree_sitter_cpp.language())).parse(src.encode()).root_node


@pytest.mark.parametrize(
    "src,expected",
    [
        ("int HashMap::find(int key) { return 0; }", ("find", "HashMap")),
        ("void Outer::Inner::run() {}", ("run", "Inner")),
        ("int add(int a, int b) { return a; }", ("add", "")),
        ("int* make() { return 0; }", ("make", "")),
        ("int& ref() { static int x; return x; }", ("ref", "")),
        ("bool operator==(const A& a, const A& b) { return true; }", ("operator==", "")),
    ],
)
def test_ts_declarator_name(src, expected):
    fn = _cpp_root(src).children[0]
    decl = fn.child_by_field_name("declarator")
    assert common.ts_declarator_name(decl if decl is not None else fn) == expected


def test_ts_declarator_name_without_a_name():
    assert common.ts_declarator_name(_cpp_root("int;\n")) == ("", "")


def test_format_chunk_without_prefix():
    assert common.format_chunk({}, "body") == "body"
    assert common.format_chunk({"kind": "Class", "file": "a.py", "name": "A"}, "") == "Class a.py A"


class _Strategy:
    def enrich(self, node):
        return node["name"]


def test_enrich_chunks_skips_prefixed_strings():
    out = common.enrich_chunks(
        ["[file] a.py", "plain text", {"name": "A", "kind": "Class", "file": "a.py"}], _Strategy())
    assert out == ["plain text", "Class a.py A | A"]
    assert common.enrich_chunks(None, _Strategy()) == []


def test_add_tree_context_annotates_and_reports(tmp_path, capsys):
    nodes = [
        {"id": 0, "parent_id": -1, "type": "class_definition", "name": "Service",
         "file": "svc.py", "start_line": 1, "end_line": 9, "text": "Class svc.py Service"},
        {"id": 1, "parent_id": 0, "type": "function_definition", "name": "create",
         "file": "svc.py", "start_line": 2, "end_line": 4, "text": "Function svc.py create"},
    ]
    (tmp_path / "tree_index.json").write_text(json.dumps({"nodes": nodes, "texts": []}))
    chunks = ["Class svc.py Service | user service", "Function svc.py create | makes one"]
    node_ids = common.add_tree_context(chunks, str(tmp_path))

    assert node_ids[0] is not None
    assert ". Children: create" in chunks[0]
    assert ". Parent: Service" in chunks[1]
    assert "enriched 2/2" in capsys.readouterr().out


def test_add_tree_context_reports_progress(tmp_path, capsys):
    nodes = [
        {"id": i, "parent_id": -1, "type": "function_definition", "name": name,
         "file": "svc.py", "start_line": i, "end_line": i + 1, "text": f"Function svc.py {name}"}
        for i, name in enumerate(["one", "two"])
    ]
    (tmp_path / "tree_index.json").write_text(json.dumps({"nodes": nodes, "texts": []}))
    common.add_tree_context(
        ["Function svc.py one | a", "Function svc.py two | b"], str(tmp_path), progress_steps=2)
    assert "enrich [2/2] 2 matched" in capsys.readouterr().out


def test_add_tree_context_survives_bad_index(tmp_path, caplog):
    (tmp_path / "tree_index.json").write_text("{not json")
    assert common.add_tree_context(["a"], str(tmp_path)) == [None]
    assert "tree enrichment skipped" in caplog.text


def test_changed_files(monkeypatch, tmp_path):
    import subprocess

    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **kw: type("R", (), {"stdout": "a.py\n\n b.py \n", "stderr": "", "returncode": 0})(),
    )
    assert common.changed_files(str(tmp_path)) == ["a.py", "b.py"]


def test_changed_files_raises_when_git_fails(monkeypatch, tmp_path):
    import subprocess

    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **kw: type("R", (), {"stdout": "", "stderr": "not a git repository", "returncode": 128})(),
    )
    with pytest.raises(RuntimeError):
        common.changed_files(str(tmp_path))
