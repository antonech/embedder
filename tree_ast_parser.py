import os, json

from tree_sitter import Language, Parser
import tree_sitter_python, tree_sitter_cpp, tree_sitter_bash
from embedder import EmbeddingModel, VectorStore, StorageIO, ASTParser
from common import (
    DEFAULT_EXCLUDE, changed_files, label_for, node_text, resolve_data_dir,
    ts_base_classes, ts_body_summary, ts_template_params,
)


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


def get_name(node) -> str:
    name_node = node.child_by_field_name("name")
    if name_node:
        return node_text(name_node)
    return ""


def get_docstring(node, source: bytes) -> str:
    children = node.children
    if not children:
        return ""
    first = children[0]
    if first.type == "comment":
        return node_text(first)
    for i, c in enumerate(children):
        if c.type == "block" and c.children:
            first_in_block = c.children[0]
            if first_in_block.type in ("expression_statement", "string"):
                try:
                    s = node_text(first_in_block)
                    if '"""' in s or "'''" in s or s.strip().startswith('"'):
                        return s.strip()
                except Exception:
                    pass
    return ""


def get_signature(node, source: bytes, lang: str) -> str:
    params = node.child_by_field_name("parameters")
    if params:
        return node_text(params)
    if lang == "cpp":
        decl = node.child_by_field_name("declarator")
        if decl:
            return node_text(decl)
    return ""


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


def get_template_params(node, source: bytes, lang: str) -> str:
    if lang not in ("cpp", "h"):
        return ""
    return ts_template_params(node)


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
                bases_list, bases_text = ts_base_classes(node)
                if bases_text:
                    extra_parts.append(bases_text)

            # Body summary
            body = ts_body_summary(node)
            if body:
                extra_parts.append(f"{{ {body} }}")

            label = label_for(node.type)
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


def _embed_files(files: list[str], root: str | None = None) -> tuple:
    """Parse and embed each file, returning (model, store, nodes)."""
    model = EmbeddingModel()
    store = VectorStore()
    all_nodes = []
    next_id = [0]
    for fp in sorted(files):
        try:
            nodes = parse_file(fp, next_id=next_id, **({"root": root} if root else {}))
        except Exception as e:
            print(f"  SKIP {fp}: {e}")
            continue
        if not nodes:
            continue
        texts = [n["text"] for n in nodes]
        store.add_many(model.embed_many(texts), texts)
        all_nodes.extend(nodes)
        print(f"  {fp}: {len(nodes)} nodes")
    return model, store, all_nodes


def _save_tree_index(data_dir: str, model, store, all_nodes: list, vec_name: str, json_name: str) -> tuple[str, str]:
    os.makedirs(data_dir, exist_ok=True)
    vec_path = os.path.join(data_dir, vec_name)
    json_path = os.path.join(data_dir, json_name)
    StorageIO.save(vec_path, store.vectors, store.texts, model.dim)
    with open(json_path, "w", encoding="utf8") as f:
        json.dump({"nodes": all_nodes, "texts": store.texts}, f, ensure_ascii=False, indent=2)
    return vec_path, json_path


def build_index(root=".", data_dir=None, exclude=DEFAULT_EXCLUDE):
    data_dir = resolve_data_dir(root, data_dir)

    exts = tuple(LANGUAGES.keys())
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ASTParser.SKIP_DIRS]
        if any(x in dirpath for x in exclude):
            continue
        for fn in sorted(filenames):
            if fn.endswith(exts):
                files.append(os.path.join(dirpath, fn))

    model, store, all_nodes = _embed_files(files, root=root)
    vec_path, json_path = _save_tree_index(data_dir, model, store, all_nodes,
                                           "tree_vectors.npz", "tree_index.json")
    print(f"\nSaved {len(all_nodes)} nodes to {vec_path} + {json_path}")
    return model, store, all_nodes


def build_delta(root=".", data_dir=None, exclude=DEFAULT_EXCLUDE):
    data_dir = resolve_data_dir(root, data_dir)
    changed = [os.path.join(root, f) for f in changed_files(root)]
    if not changed:
        print("No changed files")
        return

    exts = tuple(LANGUAGES.keys())
    changed = [f for f in changed if os.path.isfile(f) and f.endswith(exts)]
    changed = [f for f in changed if not any(x in f for x in exclude)]

    model, store, all_nodes = _embed_files(changed)
    vec_path, json_path = _save_tree_index(data_dir, model, store, all_nodes,
                                           "delta_tree_vectors.npz", "delta_tree_index.json")
    print(f"\nSaved {len(all_nodes)} delta nodes to {vec_path} + {json_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--delta", action="store_true", help="parse only files changed in HEAD")
    args = parser.parse_args()
    if args.delta:
        build_delta(root=args.root, data_dir=args.data_dir)
    else:
        build_index(root=args.root, data_dir=args.data_dir)
