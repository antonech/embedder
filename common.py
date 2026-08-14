"""Shared helpers used by the embedder, tree parser and MCP server."""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
LABELS_PATH = os.path.join(SCRIPT_DIR, "labels.json")

SKIP_DIRS = {"venv", ".git", "__pycache__", "node_modules"}
SKIP_PREFIXES = {"[line]", "[file]"}
DEFAULT_EXCLUDE = {"/venv/", "/__pycache__/", "/.", "/node_modules/", "/.git/"}

DEFAULT_LABELS = {
    "default": {"file": "[file]", "line": "[line]", "fallback": "[chunk]"},
    "mapping": {},
}


def load_json(path: str) -> dict:
    """Read a JSON object from path, or return {} if it does not exist.

    A file that exists but cannot be read or parsed is an error: silently
    falling back to defaults would hide a broken config or labels file.
    Use load_project_config() for a config.json the embedder does not own.
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise RuntimeError(f"cannot read {path}: {e}") from e
    if not isinstance(data, dict):
        raise RuntimeError(f"cannot read {path}: expected a JSON object, got {type(data).__name__}")
    return data


CONFIG_KEYS = frozenset({
    "model_name", "device", "query_prefix", "passage_prefix", "batch_size",
    "float_type", "cross_encoder_model", "enrichment", "use_clang", "embedding_store",
})


def load_project_config(path: str) -> dict:
    """Read a scanned project's config.json, or {} if it is not an embedder config.

    A scanned repository may ship its own unrelated config.json (a list, JSON
    with comments, application settings). Treating that as a broken embedder
    config would abort the whole index build, and reading settings out of it
    would embed with the wrong model, so warn and ignore it instead.
    """
    if not os.path.exists(path):
        return {}
    try:
        cfg = load_json(path)
    except RuntimeError as e:
        log.warning("ignoring %s: %s", path, e)
        return {}
    if not cfg.keys() & CONFIG_KEYS:
        log.warning("ignoring %s: it holds no embedder settings", path)
        return {}
    return cfg


def load_labels() -> dict:
    labels = load_json(LABELS_PATH)
    return labels or DEFAULT_LABELS


LABELS = load_labels()


def label_for(node_type: str, default: Optional[str] = None) -> str:
    return LABELS["mapping"].get(node_type, node_type if default is None else default)


def expand_path(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(path))


def project_data_dir(project: str, default: str = "data") -> str:
    """Index directory for a project: <embedding_store>/<project> when configured."""
    store_root = load_json(CONFIG_PATH).get("embedding_store")
    if store_root:
        return os.path.join(expand_path(store_root), project)
    return default


def resolve_data_dir(root: str, data_dir: Optional[str] = None, default: str = "data") -> str:
    """Resolve the index directory: explicit arg > config embedding_store/<project> > default."""
    if data_dir:
        return data_dir
    return project_data_dir(os.path.basename(os.path.abspath(root)), default)


@dataclass
class ModelConfig:
    """Embedding model settings resolved from config.json."""

    model_name: str = "all-MiniLM-L6-v2"
    device: Optional[str] = None
    query_prefix: Optional[str] = None
    passage_prefix: Optional[str] = None
    batch_size: int = 1024
    float_type: str = "fp32"
    cross_encoder_model: Optional[str] = None

    @classmethod
    def from_dict(cls, cfg: dict) -> "ModelConfig":
        defaults = cls()
        return cls(
            model_name=cfg.get("model_name", defaults.model_name),
            device=cfg.get("device"),
            query_prefix=cfg.get("query_prefix"),
            passage_prefix=cfg.get("passage_prefix"),
            batch_size=cfg.get("batch_size", defaults.batch_size),
            float_type=cfg.get("float_type", defaults.float_type),
            cross_encoder_model=cfg.get("cross_encoder_model"),
        )

    @classmethod
    def load(cls, root: Optional[str] = None) -> "ModelConfig":
        """Load settings from <root>/config.json, falling back to the embedder's own config."""
        if root:
            project_cfg = load_project_config(os.path.join(root, "config.json"))
            if project_cfg:
                return cls.from_dict(project_cfg)
        return cls.from_dict(load_json(CONFIG_PATH))


def resolve_device(device: Optional[str]) -> Optional[str]:
    """Fall back to CPU when a CUDA device is requested but unavailable."""
    if not device or not device.startswith("cuda"):
        return device
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
    except ImportError:
        return "cpu"
    return device


# --- tree-sitter node helpers ---


def node_text(node) -> str:
    return node.text.decode("utf8", errors="ignore")


def ts_base_classes(node) -> tuple[list[str], str]:
    """Base classes of a C++ class/struct node as (names, ': A, B') text."""
    bases = [
        node_text(cc)
        for child in node.children if child.type == "base_class_clause"
        for cc in child.children if cc.type == "type_identifier"
    ]
    if bases:
        return bases, ": " + ", ".join(bases)
    return [], ""


def ts_template_params(node) -> str:
    for child in node.children:
        if child.type == "template_parameter_list":
            return node_text(child)
    return ""


def _ts_qualified_parts(node) -> tuple[str, str]:
    """(innermost scope, name) of a possibly nested qualified_identifier.

    `Outer::Inner::run` nests right, so recurse to reach the class that actually
    owns the member: ('Inner', 'run').
    """
    scope = node.child_by_field_name("scope")
    name = node.child_by_field_name("name")
    scope_text = node_text(scope) if scope is not None else ""
    if name is not None and name.type == "qualified_identifier":
        inner_scope, inner_name = _ts_qualified_parts(name)
        return (inner_scope or scope_text), inner_name
    return scope_text, node_text(name) if name is not None else ""


def ts_declarator_name(node) -> tuple[str, str]:
    """(name, owning class) for a C++ declarator subtree.

    Handles out-of-line definitions (`int HashMap::find(int key) {...}`), which
    carry their name inside a qualified_identifier rather than a plain identifier
    and would otherwise be skipped entirely for having no name.
    """
    if node.type in ("identifier", "field_identifier", "operator_name"):
        return node_text(node), ""
    if node.type == "qualified_identifier":
        scope, name = _ts_qualified_parts(node)
        return name, scope
    for c in node.children:
        if c.type in ("identifier", "field_identifier", "operator_name"):
            return node_text(c), ""
        if c.type == "qualified_identifier":
            scope, name = _ts_qualified_parts(c)
            return name, scope
        if c.type in ("reference_declarator", "pointer_declarator", "declarator",
                      "function_declarator"):
            name, scope = ts_declarator_name(c)
            if name:
                return name, scope
    return "", ""


def _ts_method_entry(node, access: str) -> Optional[str]:
    name_node = node.child_by_field_name("name")
    params = node.child_by_field_name("parameters")
    decl = node.child_by_field_name("declarator")
    if decl is not None:
        if name_node is None:
            name_node = decl.child_by_field_name("name")
        if name_node is None:
            name_node = next((c for c in decl.children if c.type == "field_identifier"), None)
        if params is None:
            params = decl.child_by_field_name("parameters")
    if name_node is None:
        return None
    sig = node_text(params)[:40] if params is not None else ""
    return f"{access}: {node_text(name_node)}{sig}"


def ts_body_summary(node, max_methods: int = 8, max_fields: int = 6) -> str:
    """Summarize a C++ class/struct body as 'Methods: ... . Fields: ...'."""
    body = next((c for c in node.children if c.type == "field_declaration_list"), None)
    if body is None:
        return ""
    access = "public"
    methods: list[str] = []
    fields: list[str] = []
    for c in body.children:
        if c.type == "access_specifier":
            access = node_text(c).strip()
        elif c.type == "declaration":
            txt = node_text(c).strip()
            if txt.startswith("virtual") or "(" in txt:
                methods.append(f"{access}: {txt.split('(')[0].split()[-1]}(...)")
        elif c.type == "function_definition":
            entry = _ts_method_entry(c, access)
            if entry:
                methods.append(entry)
        elif c.type == "field_declaration":
            txt = node_text(c).strip()
            if "(" in txt and txt.split("(")[0].strip().split()[-1]:
                mname = txt.split("(")[0].strip().split()[-1]
                methods.append(f"{access}: {mname}(" + txt.split("(")[1][:40])
            else:
                parts = txt.split()
                if parts and parts[-1] not in ("override", "= 0", "final"):
                    fname = parts[-1].rstrip(";=,")
                    if fname and not fname.startswith("//"):
                        fields.append(fname)
    result = []
    if methods:
        result.append("Methods: " + ", ".join(methods[:max_methods]))
    if fields:
        result.append("Fields: " + ", ".join(fields[:max_fields]))
    return ". ".join(result) if result else ""


# --- chunk helpers ---


def format_chunk(node: dict, enriched: str) -> str:
    """Prefix an enriched node with '<kind> <file> <name>'."""
    prefix = f"{node.get('kind', '')} {node.get('file', '')} {node.get('name', '')}".strip()
    if not prefix:
        return enriched
    return f"{prefix} | {enriched}" if enriched else prefix


def enrich_chunks(file_chunks, strategy) -> list[str]:
    """Turn parser output (dicts and/or raw strings) into embeddable chunk texts."""
    chunks = []
    for c in file_chunks or []:
        if isinstance(c, str):
            if c.startswith(tuple(SKIP_PREFIXES)):
                continue
            chunks.append(c)
        else:
            chunks.append(format_chunk(c, strategy.enrich(c)))
    return chunks


def add_tree_context(chunks: list[str], data_dir: str, progress_steps: int = 0) -> list:
    """Append parent/children context to chunks in place; return per-chunk tree node ids."""
    node_ids: list = [None] * len(chunks)
    try:
        from tree_search import TreeIndex

        ti = TreeIndex(data_dir=data_dir)
        for i, text in enumerate(chunks):
            n = ti.match_node(text)
            if n is None:
                continue
            uid = n["_uid"]
            node_ids[i] = uid
            parent = ti.get_parent(uid)
            if parent:
                chunks[i] += f". Parent: {parent['name']}"
            kids = ti.get_children(uid)
            if kids:
                chunks[i] += ". Children: " + ", ".join(c["name"] for c in kids[:6])
            if progress_steps and ((i + 1) % max(1, len(chunks) // progress_steps) == 0
                                   or i == len(chunks) - 1):
                matched = sum(1 for n in node_ids[:i + 1] if n is not None)
                print(f"  enrich [{i+1}/{len(chunks)}] {matched} matched", flush=True)
        if not progress_steps:
            matched = sum(1 for n in node_ids if n is not None)
            print(f"  enriched {matched}/{len(chunks)} chunks with tree context")
    except Exception:
        log.warning("tree enrichment skipped for %s", data_dir, exc_info=True)
    return node_ids


def changed_files(root: str) -> list[str]:
    """Paths (relative to root) of files changed vs HEAD.

    Raises if git fails, so "not a git repository" cannot be mistaken for
    "nothing changed" and silently produce an empty delta index.
    """
    import subprocess

    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        capture_output=True, text=True, cwd=root, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git diff failed in {root} (exit {result.returncode}): {result.stderr.strip()}"
        )
    return [f.strip() for f in result.stdout.splitlines() if f.strip()]
