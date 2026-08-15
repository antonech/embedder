import json

import pytest

from common import write_tree_index
from tree_search import TreeIndex


def _node(nid, name, parent_id=-1, file="a.py", type="function_definition", start=1, end=2):
    return {
        "id": nid,
        "parent_id": parent_id,
        "type": type,
        "name": name,
        "file": file,
        "start_line": start,
        "end_line": end,
        "text": f"{type} {file} {name}",
    }


def _write(path, nodes):
    path.write_text(json.dumps({"nodes": nodes, "texts": [n["text"] for n in nodes]}))


@pytest.fixture
def data_dir(tmp_path):
    nodes = [
        _node(0, "Service", type="class_definition", start=1, end=20),
        _node(1, "create", parent_id=0, start=2, end=5),
        _node(2, "delete", parent_id=0, start=6, end=9),
        _node(3, "helper", start=22, end=25),
    ]
    _write(tmp_path / "tree_index.json", nodes)
    return tmp_path


def test_reads_the_json_lines_format(tmp_path):
    nodes = [
        _node(0, "Service", type="class_definition", start=1, end=20),
        _node(1, "create", parent_id=0, start=2, end=5),
    ]
    write_tree_index(str(tmp_path / "tree_index.json"), nodes, [n["text"] for n in nodes])

    idx = TreeIndex(data_dir=str(tmp_path))
    assert len(idx.nodes) == 2
    assert idx.texts == [n["text"] for n in nodes]
    assert idx.get_parent(1)["name"] == "Service"
    assert idx.match_node("class_definition a.py Service")["_uid"] == 0


def test_nodes_are_slots_not_dicts(data_dir):
    """A dict per node cost ~270 MB on a 100k-node index; slots keep it usable."""
    idx = TreeIndex(data_dir=str(data_dir))
    node = idx.nodes[0]
    assert not isinstance(node, dict)
    assert not hasattr(node, "__dict__")
    with pytest.raises(KeyError):
        node["docstring"]


def test_empty_dir_yields_empty_index(tmp_path):
    idx = TreeIndex(data_dir=str(tmp_path))
    assert idx.nodes == {}
    assert idx.texts == []
    assert idx.match_node("Class a.py Service") is None


def test_loads_nodes_texts_and_lookup(data_dir):
    idx = TreeIndex(data_dir=str(data_dir))
    assert len(idx.nodes) == 4
    assert len(idx.texts) == 4
    assert idx.lookup[("a.py", "Service")] == 0
    # internal id/parent_id keys are normalized to uids
    assert idx.nodes[1]["parent_id"] == 0
    assert idx.nodes[0]["parent_id"] == -1
    assert idx.nodes[0]["_uid"] == 0
    assert "_shifted_parent_id" not in idx.nodes[0]
    assert "id" not in idx.nodes[0]
    assert idx.nodes[0].get("docstring", "unset") == "unset"


def test_relations(data_dir):
    idx = TreeIndex(data_dir=str(data_dir))
    assert [c["name"] for c in idx.get_children(0)] == ["create", "delete"]
    assert idx.get_children(3) == []
    assert idx.get_parent(1)["name"] == "Service"
    assert idx.get_parent(0) is None
    assert idx.get_parent(999) is None
    assert [s["name"] for s in idx.get_siblings(1)] == ["delete"]
    assert idx.get_siblings(0) == []
    assert idx.get_siblings(999) == []


def test_get_node(data_dir):
    idx = TreeIndex(data_dir=str(data_dir))
    assert idx.get_node(2)["name"] == "delete"
    assert idx.get_node(42) is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("class_definition a.py Service", "Service"),
        ("function_definition a.py create,", "create"),
        ("function_definition a.py missing", None),
        ("tooshort", None),
        ("", None),
    ],
)
def test_match_node(data_dir, text, expected):
    idx = TreeIndex(data_dir=str(data_dir))
    node = idx.match_node(text)
    assert (node["name"] if node else None) == expected


def test_match_node_resolves_a_quoted_path_containing_spaces(tmp_path):
    # A bare 'my file.py' would be read as file='my', name='file.py' and silently
    # lose tree context, so common.format_chunk quotes such paths.
    nodes = [
        _node(0, "Service", file="my file.py", type="class_definition", start=1, end=20),
        _node(1, "run", file="my file.py", parent_id=0, start=2, end=5),
        _node(2, "Service", file="plain.py", type="class_definition", start=1, end=20),
    ]
    _write(tmp_path / "tree_index.json", nodes)
    idx = TreeIndex(data_dir=str(tmp_path))

    spaced = idx.match_node('Class "my file.py" Service | Methods: run')
    assert spaced["file"] == "my file.py" and spaced["start_line"] == 1
    assert [c["name"] for c in idx.get_children(spaced["_uid"])] == ["run"]
    # The '. Parent: X' suffix add_tree_context appends must not break the match.
    assert idx.match_node('Class "my file.py" Service | x. Parent: Service')["file"] == "my file.py"
    # Unquoted paths keep working unchanged.
    assert idx.match_node("Class plain.py Service | Methods: run")["file"] == "plain.py"


def test_match_node_disambiguates_method_from_module_function(tmp_path):
    nodes = [
        _node(0, "Service", type="class_definition", start=1, end=20),
        _node(1, "run", parent_id=0, start=2, end=5),
        _node(2, "run", start=30, end=35),
    ]
    _write(tmp_path / "tree_index.json", nodes)
    idx = TreeIndex(data_dir=str(tmp_path))

    method = idx.match_node("Method a.py run | (self) | In class: Service")
    assert method["start_line"] == 2
    module_fn = idx.match_node("Function a.py run | (project)")
    assert module_fn["start_line"] == 30


def test_annotate_adds_context(data_dir):
    idx = TreeIndex(data_dir=str(data_dir))
    hits = [
        {"text": "function_definition a.py create"},
        {"text": "class_definition a.py Service"},
        {"text": "function_definition a.py unknown"},
    ]
    out = idx.annotate(hits)
    assert out is hits
    child_ctx = hits[0]["context"]
    assert child_ctx["parent"] == {"name": "Service", "type": "class_definition"}
    assert child_ctx["siblings"] == [{"name": "delete", "type": "function_definition"}]
    assert child_ctx["children"] == []

    parent_ctx = hits[1]["context"]
    assert parent_ctx["parent"] is None
    assert parent_ctx["children"] == [
        {"name": "create", "type": "function_definition", "file": "a.py", "lines": "2-5"},
        {"name": "delete", "type": "function_definition", "file": "a.py", "lines": "6-9"},
    ]
    assert "context" not in hits[2]


def test_delta_index_is_merged_without_id_collisions(data_dir):
    delta_nodes = [
        _node(0, "Extra", file="b.py", type="class_definition"),
        _node(1, "run", parent_id=0, file="b.py"),
    ]
    _write(data_dir / "delta_tree_index.json", delta_nodes)

    idx = TreeIndex(data_dir=str(data_dir))
    assert len(idx.nodes) == 6
    assert len(idx.texts) == 6
    # main-index relations survive the merge
    assert idx.get_parent(1)["name"] == "Service"
    # delta relations are remapped onto fresh uids
    extra_uid = idx.lookup[("b.py", "Extra")]
    run_uid = idx.lookup[("b.py", "run")]
    assert extra_uid == 4 and run_uid == 5
    assert idx.get_parent(run_uid)["name"] == "Extra"
    assert [c["name"] for c in idx.get_children(extra_uid)] == ["run"]


def test_delta_nodes_get_distinct_uids(data_dir):
    delta_nodes = [
        _node(0, "Extra", file="b.py", type="class_definition"),
        _node(1, "run", parent_id=0, file="b.py"),
    ]
    _write(data_dir / "delta_tree_index.json", delta_nodes)

    idx = TreeIndex(data_dir=str(data_dir))
    assert len(idx.nodes) == 6
    assert len(idx.texts) == 6
    assert idx.lookup[("b.py", "Extra")] == 4
    assert idx.lookup[("b.py", "run")] == 5
    assert idx.get_parent(5)["name"] == "Extra"
    assert [c["name"] for c in idx.get_children(4)] == ["run"]


def test_lookup_skips_nodes_without_file_or_name(tmp_path):
    nodes = [_node(0, ""), _node(1, "ok", file="")]
    _write(tmp_path / "tree_index.json", nodes)
    idx = TreeIndex(data_dir=str(tmp_path))
    assert idx.lookup == {}
