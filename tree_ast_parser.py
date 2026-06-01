import os, json, glob
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tree_sitter import Language, Parser
import tree_sitter_python, tree_sitter_cpp, tree_sitter_bash
from embedder import EmbeddingModel, VectorStore, StorageIO


LANGUAGES = {
    ".py":  Language(tree_sitter_python.language()),
    ".cpp": Language(tree_sitter_cpp.language()),
    ".cxx": Language(tree_sitter_cpp.language()),
    ".cc":  Language(tree_sitter_cpp.language()),
    ".h":   Language(tree_sitter_cpp.language()),
    ".hpp": Language(tree_sitter_cpp.language()),
    ".sh":  Language(tree_sitter_bash.language()),
    ".bash": Language(tree_sitter_bash.language()),
}

SIGNIFICANT_TYPES = {
    "class_definition", "function_definition",
    "method_signature", "function_signature",
    "struct_specifier", "class_specifier",
    "template_function", "template_method",
    "enum_specifier", "alias_declaration",
    "declaration",
}


_LABELS_PATH = os.path.join(os.path.dirname(__file__), "labels.json")
if os.path.exists(_LABELS_PATH):
    with open(_LABELS_PATH) as f:
        _LABELS = json.load(f)
else:
    _LABELS = {"default": {"file": "[file]", "line": "[line]", "fallback": "[chunk]"}, "mapping": {}}


def get_name(node) -> str:
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf8", errors="ignore")
    return ""


def get_docstring(node, source: bytes) -> str:
    children = node.children
    if not children:
        return ""
    first = children[0]
    if first.type == "comment":
        return first.text.decode("utf8", errors="ignore")
    for i, c in enumerate(children):
        if c.type == "block" and c.children:
            first_in_block = c.children[0]
            if first_in_block.type in ("expression_statement", "string"):
                try:
                    s = first_in_block.text.decode("utf8", errors="ignore")
                    if '"""' in s or "'''" in s or s.strip().startswith('"'):
                        return s.strip()
                except Exception:
                    pass
    return ""


def get_signature(node, source: bytes, lang: str) -> str:
    params = node.child_by_field_name("parameters")
    if params:
        return params.text.decode("utf8", errors="ignore")
    if lang == "cpp":
        decl = node.child_by_field_name("declarator")
        if decl:
            return decl.text.decode("utf8", errors="ignore")
    return ""


def get_base_classes(node, source: bytes) -> tuple[list[str], str]:
    bases = []
    for child in node.children:
        if child.type == "base_class_clause":
            for cc in child.children:
                if cc.type == "type_identifier":
                    bases.append(cc.text.decode("utf8", errors="ignore"))
    if bases:
        return bases, ": " + ", ".join(bases)
    return [], ""


def get_includes(source: bytes, lang: str) -> list[str]:
    includes = []
    if lang in ("cpp", "h"):
        text = source.decode("utf8", errors="ignore")
        for line in text.splitlines():
            ls = line.strip()
            if ls.startswith("#include"):
                inc = ls.split(None, 1)[-1].strip("\"<>")
                if inc:
                    includes.append(inc)
    return includes


def get_body_summary(node, source: bytes) -> str:
    body = None
    for c in node.children:
        if c.type == "field_declaration_list":
            body = c
            break
    if not body:
        return ""
    access = "public"
    methods = []
    fields = []
    for c in body.children:
        if c.type == "access_specifier":
            access = c.text.decode("utf8", errors="ignore").strip()
        elif c.type == "declaration":
            txt = c.text.decode("utf8", errors="ignore").strip()
            if txt.startswith("virtual") or "(" in txt:
                methods.append(f"{access}: {txt.split('(')[0].split()[-1]}(...)")
        elif c.type == "function_definition":
            nname = c.child_by_field_name("name")
            sig = ""
            if c.child_by_field_name("parameters"):
                sig = c.child_by_field_name("parameters").text.decode("utf8", errors="ignore")[:40]
            if nname:
                methods.append(f"{access}: {nname.text.decode()}{sig}")
        elif c.type == "field_declaration":
            txt = c.text.decode("utf8", errors="ignore").strip()
            if "(" in txt and txt.split("(")[0].strip().split()[-1]:
                mname = txt.split("(")[0].strip().split()[-1]
                sig = "(" + txt.split("(")[1][:40]
                methods.append(f"{access}: {mname}{sig}")
            else:
                parts = txt.split()
                if parts and parts[-1] not in ("override", "= 0", "final"):
                    fname = parts[-1].rstrip(";=,")
                    if fname and not fname.startswith("//"):
                        fields.append(fname)
    result = []
    if methods:
        result.append("Methods: " + ", ".join(methods[:8]))
    if fields:
        result.append("Fields: " + ", ".join(fields[:6]))
    return ". ".join(result) if result else ""


def get_template_params(node, source: bytes, lang: str) -> str:
    if lang not in ("cpp", "h"):
        return ""
    for child in node.children:
        if child.type == "template_parameter_list":
            return child.text.decode("utf8", errors="ignore")
    return ""


def collect_nodes(node, source: bytes, filepath: str, lang: str, nodes: list, parent_id: int = -1, next_id: list | None = None, root: str = ".", file_includes: list | None = None):
    rel_filepath = os.path.relpath(filepath, root)
    if next_id is None:
        next_id = [0]
    node_id = next_id[0]

    if node.type in SIGNIFICANT_TYPES:
        name = get_name(node)
        if name:
            next_id[0] += 1
            sig = get_signature(node, source, lang)
            doc = get_docstring(node, source)

            extra_parts = []

            # Template parameters
            tpl = get_template_params(node, source, lang)
            if tpl:
                extra_parts.append(f"template {tpl}")

            # Base classes for class/struct
            bases_list = []
            if node.type in ("class_specifier", "struct_specifier"):
                bases_list, bases_text = get_base_classes(node, source)
                if bases_text:
                    extra_parts.append(bases_text)

            # Body summary
            body = get_body_summary(node, source)
            if body:
                extra_parts.append(f"{{ {body} }}")

            label = _LABELS["mapping"].get(node.type, node.type)
            text = f"{label} {rel_filepath} {name}"
            if extra_parts:
                text += ". " + ". ".join(extra_parts)
            if sig:
                text += f". Signature: {sig}"
            if doc:
                doc_short = doc.strip().replace("\n", " ")[:200]
                text += f". Doc: {doc_short}"

            entry = {
                "id": node_id,
                "parent_id": parent_id,
                "type": node.type,
                "name": name,
                "file": rel_filepath,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "signature": sig,
                "docstring": doc,
                "text": text,
            }
            if bases_list:
                entry["bases"] = bases_list
            if file_includes:
                entry["includes"] = file_includes
            nodes.append(entry)
            parent_id = node_id

    for child in node.children:
        collect_nodes(child, source, filepath, lang, nodes, parent_id, next_id, root, file_includes)


def parse_file(filepath: str, next_id: list | None = None, root: str = ".") -> list[dict]:
    ext = os.path.splitext(filepath)[1].lower()
    lang_obj = LANGUAGES.get(ext)
    if not lang_obj:
        return []

    with open(filepath, "rb") as f:
        source = f.read()

    # Language key for includes: cpp for C++ family, py for python, etc.
    lang_key = ".cpp" if ext in (".cpp", ".cc", ".cxx", ".h", ".hpp") else ext
    file_includes = get_includes(source, lang_key.lstrip("."))

    parser = Parser(lang_obj)
    tree = parser.parse(source)
    nodes = []
    collect_nodes(tree.root_node, source, filepath, ext, nodes, next_id=next_id, root=root, file_includes=file_includes)
    return nodes


def _resolve_data_dir(root: str, data_dir: str | None) -> str:
    if data_dir:
        return data_dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    embedder_cfg = os.path.join(script_dir, "config.json")
    if os.path.exists(embedder_cfg):
        with open(embedder_cfg) as f:
            cfg = json.load(f)
        store_root = cfg.get("embedding_store")
        if store_root:
            store_root = os.path.expandvars(os.path.expanduser(store_root))
            return os.path.join(store_root, os.path.basename(root))
    return ""


def build_index(root=".", data_dir=None, exclude={"/venv/", "/__pycache__/", "/.", "/node_modules/", "/.git/"}):
    data_dir = _resolve_data_dir(root, data_dir) or "data"
    model = EmbeddingModel()
    store = VectorStore()
    all_nodes = []

    exts = tuple(LANGUAGES.keys())
    files = glob.glob(f"{root}/**/*", recursive=True)
    files = [f for f in files if os.path.isfile(f) and f.endswith(exts)]
    files = [f for f in files if not any(x in f for x in exclude)]

    next_id = [0]
    for fp in sorted(files):
        try:
            nodes = parse_file(fp, next_id=next_id, root=root)
        except Exception as e:
            print(f"  SKIP {fp}: {e}")
            continue
        if not nodes:
            continue
        texts = [n["text"] for n in nodes]
        vecs = model.embed_many(texts)
        store.add_many(vecs, texts)
        all_nodes.extend(nodes)
        print(f"  {fp}: {len(nodes)} nodes")

    os.makedirs(data_dir, exist_ok=True)
    vec_path = os.path.join(data_dir, "tree_vectors.npz")
    json_path = os.path.join(data_dir, "tree_index.json")
    StorageIO.save(vec_path, store.vectors, store.texts, model.dim)

    tree_data = {"nodes": all_nodes, "texts": store.texts}
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(tree_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(all_nodes)} nodes to {vec_path} + {json_path}")
    return model, store, all_nodes


def build_delta(root=".", data_dir=None, exclude={"/venv/", "/__pycache__/", "/.", "/node_modules/", "/.git/"}):
    data_dir = _resolve_data_dir(root, data_dir) or "data"
    changed = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        capture_output=True, text=True, cwd=root
    ).stdout.strip().splitlines()
    changed = [os.path.join(root, f) for f in changed if f]
    if not changed:
        print("No changed files")
        return

    exts = tuple(LANGUAGES.keys())
    changed = [f for f in changed if os.path.isfile(f) and f.endswith(exts)]
    changed = [f for f in changed if not any(x in f for x in exclude)]

    model = EmbeddingModel()
    store = VectorStore()
    all_nodes = []
    next_id = [0]

    for fp in sorted(changed):
        try:
            nodes = parse_file(fp, next_id=next_id)
        except Exception as e:
            print(f"  SKIP {fp}: {e}")
            continue
        if not nodes:
            continue
        texts = [n["text"] for n in nodes]
        vecs = model.embed_many(texts)
        store.add_many(vecs, texts)
        all_nodes.extend(nodes)
        print(f"  {fp}: {len(nodes)} nodes")

    os.makedirs(data_dir, exist_ok=True)
    vec_path = os.path.join(data_dir, "delta_tree_vectors.npz")
    json_path = os.path.join(data_dir, "delta_tree_index.json")
    StorageIO.save(vec_path, store.vectors, store.texts, model.dim)

    tree_data = {"nodes": all_nodes, "texts": store.texts}
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(tree_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(all_nodes)} delta nodes to {vec_path} + {json_path}")


if __name__ == "__main__":
    import argparse, subprocess
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--delta", action="store_true", help="parse only files changed in HEAD")
    args = parser.parse_args()
    if args.delta:
        build_delta(root=args.root, data_dir=args.data_dir)
    else:
        build_index(root=args.root, data_dir=args.data_dir)
