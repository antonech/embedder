import os, json, ast, argparse, logging, sys

import numpy as np
from typing import Optional
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer

from common import (
    DEFAULT_EXCLUDE, LABELS as _LABELS, ModelConfig,
    SKIP_DIRS as _SKIP_DIRS, SKIP_PREFIXES as _SKIP_PREFIXES,
    add_tree_context, changed_files, enrich_chunks, label_for,
    load_project_config, node_text, resolve_data_dir, ts_base_classes,
    ts_body_summary, ts_declarator_name, write_tree_index,
)

log = logging.getLogger(__name__)

# Enrichment applied to each AST node when config.json does not say otherwise.
# Deliberately excludes "statements": raw body lines are noise for retrieval.
DEFAULT_ENRICHMENT = ["signature", "structure", "content", "docstring"]

try:
    from tree_sitter import Language, Parser
    import tree_sitter_javascript, tree_sitter_go, tree_sitter_rust
    _TS_AVAILABLE = True
except ImportError:
    _TS_AVAILABLE = False

_TS_LANGUAGES = {
    ".js":  ("javascript", tree_sitter_javascript.language),
    ".jsx": ("javascript", tree_sitter_javascript.language),
    ".ts":  ("javascript", tree_sitter_javascript.language),  # TS uses JS grammar for basic parsing
    ".tsx": ("javascript", tree_sitter_javascript.language),
    ".mjs": ("javascript", tree_sitter_javascript.language),
    ".cjs": ("javascript", tree_sitter_javascript.language),
    ".go":  ("go", tree_sitter_go.language),
    ".rs":  ("rust", tree_sitter_rust.language),
} if _TS_AVAILABLE else {}

# --- Clang support (optional) ---
_CLANG_LIB_PATHS = ['/usr/lib/llvm-18/lib', '/usr/lib/llvm-15/lib', '/usr/lib/x86_64-linux-gnu']


def _configure_clang(ci_module) -> None:
    """Point libclang at the first library path that accepts it."""
    for libpath in _CLANG_LIB_PATHS:
        try:
            ci_module.Config.set_library_path(libpath)
            break
        except Exception as e:
            log.debug("clang library path %s rejected: %s", libpath, e)


try:
    import clang.cindex as ci
    _configure_clang(ci)
    _CLANG_AVAILABLE = True
except Exception as e:
    log.debug("clang bindings unavailable, C++ parsing falls back to tree-sitter: %s", e)
    _CLANG_AVAILABLE = False

_TS_SIGNIFICANT = {
    "javascript": {
        "class_declaration", "function_declaration",
        "method_definition", "arrow_function",
    },
    "go": {
        "function_declaration", "method_declaration",
        "type_declaration", "type_spec",
    },
    "rust": {
        "function_item", "struct_item", "impl_item",
        "trait_item", "enum_item",
    },
}


class EnrichmentStrategy(ABC):
    """Base class for AST node enrichment strategies."""

    @abstractmethod
    def enrich(self, node_info: dict) -> str:
        ...

    @classmethod
    @abstractmethod
    def key(cls) -> str:
        ...


class NameStrategy(EnrichmentStrategy):
    @classmethod
    def key(cls) -> str:
        return "name"

    def enrich(self, node: dict) -> str:
        return node.get("name", "")


class KindStrategy(EnrichmentStrategy):
    @classmethod
    def key(cls) -> str:
        return "kind"

    def enrich(self, node: dict) -> str:
        return node.get("kind", "")


class SignatureStrategy(EnrichmentStrategy):
    """Arguments and return type for functions/methods."""

    @classmethod
    def key(cls) -> str:
        return "signature"

    def enrich(self, node: dict) -> str:
        sig = node.get("signature")
        if sig and isinstance(sig, str):
            return sig
        args = node.get("args", [])
        if isinstance(args, list) and not args:
            return ""
        if isinstance(args, list):
            args_str = ", ".join(a.get("name", str(a)) if isinstance(a, dict) else str(a) for a in args)
        else:
            args_str = str(args)
        ret = node.get("returns", "")
        if ret:
            return f"({args_str}) -> {ret}"
        return f"({args_str})"


class DocstringStrategy(EnrichmentStrategy):
    @classmethod
    def key(cls) -> str:
        return "docstring"

    def enrich(self, node: dict) -> str:
        return node.get("docstring", node.get("doc", ""))


class StructureStrategy(EnrichmentStrategy):
    """Shape of a node: owning class, methods, fields, bases.

    Structural facts are what makes a chunk findable ("which class has a
    send_packet method?"), so this is part of the default enrichment.
    """

    @classmethod
    def key(cls) -> str:
        return "structure"

    def enrich(self, node: dict) -> str:
        parts = []
        in_class = node.get("in_class", "")
        if in_class:
            parts.append(f"In class: {in_class}")
        methods = node.get("methods", [])
        if methods:
            m_strs = [str(m)[:80] for m in methods[:8] if str(m).strip()]
            if m_strs:
                parts.append("Methods: " + ", ".join(m_strs))
        fields = node.get("fields", [])
        if fields:
            f_strs = [str(f)[:60] for f in fields[:6] if str(f).strip()]
            if f_strs:
                parts.append("Fields: " + ", ".join(f_strs))
        bases = node.get("bases", [])
        if bases:
            b_strs = [str(b)[:60] for b in bases[:4]]
            if b_strs:
                parts.append("Inherits: " + ", ".join(b_strs))
        return " | ".join(parts)


class ContentStrategy(EnrichmentStrategy):
    """Prose/summary text for nodes whose body is a single string.

    This is the only signal non-code files (.md, .json, unknown types) carry, and
    it also holds the C++ class summary, so it stays in the default enrichment.
    """

    MAX_CHARS = 200

    @classmethod
    def key(cls) -> str:
        return "content"

    def enrich(self, node: dict) -> str:
        body = node.get("body", "")
        if isinstance(body, str) and body.strip():
            return body[:self.MAX_CHARS]
        return ""


class StatementsStrategy(EnrichmentStrategy):
    """First few source statements of a function body.

    Opt-in: statements are mostly noise for retrieval (control flow and local
    variable names rarely describe what the function is for) and they crowd out
    the signature/docstring. Enable with "statements" in config enrichment.
    """

    MAX_LINES = 5
    MAX_CHARS = 100

    @classmethod
    def key(cls) -> str:
        return "statements"

    def enrich(self, node: dict) -> str:
        body = node.get("body", [])
        if not isinstance(body, list):
            return ""
        lines = [str(b)[:self.MAX_CHARS] for b in body if str(b).strip()]
        return " | ".join(lines[:self.MAX_LINES])


class BodyStrategy(EnrichmentStrategy):
    """Legacy key: structure + content + statements, in that order.

    Kept so existing configs asking for "body" keep working, since from_keys
    silently drops unknown keys.
    """

    MAX_LINES = StatementsStrategy.MAX_LINES

    @classmethod
    def key(cls) -> str:
        return "body"

    def enrich(self, node: dict) -> str:
        parts = [StructureStrategy().enrich(node),
                 ContentStrategy().enrich(node),
                 StatementsStrategy().enrich(node)]
        return " | ".join(p for p in parts if p)


class CompositeStrategy(EnrichmentStrategy):
    """Combine multiple strategies by config key order."""

    def __init__(self, strategies: list[EnrichmentStrategy], sep: str = " | "):
        self._strategies = strategies
        self._sep = sep

    @classmethod
    def key(cls) -> str:
        return "composite"

    def enrich(self, node: dict) -> str:
        parts = [s.enrich(node) for s in self._strategies]
        return self._sep.join(p for p in parts if p)

    @classmethod
    def from_keys(cls, keys: list[str], sep: str = " | ") -> "CompositeStrategy":
        registry = {s.key(): s for s in EnrichmentStrategy.__subclasses__()}
        strategies = []
        for k in keys:
            found = registry.get(k)
            if found:
                strategies.append(found() if isinstance(found, type) else found)
        return cls(strategies, sep=sep)

    @classmethod
    def default(cls) -> "CompositeStrategy":
        return cls.from_keys(["kind", "name", "signature", "docstring"])


class ASTParser:
    """Extract enriched chunks from source files for any language.

    Supports Python (native AST), C/C++ (regex-based), and falls back to
    line-based chunking for unknown languages. Add new languages by
    registering a handler: ASTParser.register('*.ext', handler_fn).
    """

    SKIP_DIRS = _SKIP_DIRS
    SKIP_PREFIXES = _SKIP_PREFIXES

    _handlers: dict[str, callable] = {}

    # --- Clang significant cursor kinds (if clang available) ---
    if _CLANG_AVAILABLE:
        _CLANG_SIGNIFICANT = {
            ci.CursorKind.CLASS_DECL,
            ci.CursorKind.STRUCT_DECL,
            ci.CursorKind.UNION_DECL,
            ci.CursorKind.FUNCTION_DECL,
            ci.CursorKind.ENUM_DECL,
            ci.CursorKind.TYPEDEF_DECL,
            ci.CursorKind.TYPE_ALIAS_DECL,
        }
    else:
        _CLANG_SIGNIFICANT = set()
    _use_clang = False  # class variable, set by scan_project from config

    @classmethod
    def register(cls, glob_pattern: str, handler: callable):
        """Register a handler for a file pattern (e.g. '*.cpp')."""
        cls._handlers[glob_pattern] = handler

    @classmethod
    def _handler_for(cls, filepath: str) -> callable:
        for pattern, handler in cls._handlers.items():
            if filepath.endswith(pattern.replace('*', '')):
                return handler
        return cls._parse_fallback

    @classmethod
    def parse_file(cls, filepath: str, path_hint: str = "") -> list[dict | str]:
        """Parse a single file into enriched chunks using the registered handler."""
        try:
            with open(filepath, errors='replace') as f:
                content = f.read()
        except OSError as e:
            log.warning("cannot read %s: %s", filepath, e)
            return []
        if not content.strip() or '\x00' in content[:2000]:
            return []
        return cls._handler_for(filepath)(content, path_hint or filepath)

    @classmethod
    def _load_strategy(cls, root: str, enrichment_keys: Optional[list[str]] = None) -> CompositeStrategy:
        cfg = load_project_config(os.path.join(root, "config.json"))
        cls._use_clang = cfg.get("use_clang", False)
        if enrichment_keys is None:
            enrichment_keys = cfg.get("enrichment")
        return CompositeStrategy.from_keys(enrichment_keys or DEFAULT_ENRICHMENT)

    @classmethod
    def scan_project(cls, root: str, enrichment_keys: Optional[list[str]] = None) -> list[str]:
        """Scan project and return enriched chunks for supported files.

        Args:
            root: Project root directory.
            enrichment_keys: Order of enrichment strategies. If None, read from
                root/config.json or use default.
        """
        chunks = []
        strategy = cls._load_strategy(root, enrichment_keys)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in cls.SKIP_DIRS]
            for fn in sorted(filenames):
                fp = os.path.join(dirpath, fn)
                name = os.path.relpath(fp, root)
                # Parse every file; unregistered types (XML, .sh, .md) use fallback
                try:
                    chunks.extend(enrich_chunks(cls.parse_file(fp, path_hint=name), strategy))
                except Exception:
                    log.warning("failed to parse %s, skipping", fp, exc_info=True)
        return chunks

    # --- Python handler ---

    class _ParentVisitor(ast.NodeTransformer):
        def visit(self, node):
            for child in ast.iter_child_nodes(node):
                child.parent = node
            return super().visit(node)

    @staticmethod
    def _python_body_lines(node, path_hint: str = "", limit: int = 2) -> list[str]:
        """First real statements of a function body, excluding its docstring.

        The docstring is a plain expression statement, so unparsing node.body
        blindly makes it body[0] -- duplicating DocstringStrategy and wasting a
        slot. ast.get_docstring already covers it.
        """
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            body = body[1:]
        try:
            return [ast.unparse(s)[:80] for s in body[:limit]]
        except Exception as e:
            log.debug("cannot unparse body of %s in %s: %s", node.name, path_hint, e)
            return []

    @classmethod
    def _parse_python(cls, source: str, path_hint: str = "") -> list[dict | str]:
        chunks: list[dict | str] = []
        try:
            tree = ast.parse(source)
            cls._ParentVisitor().visit(tree)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    doc = ast.get_docstring(node) or ""
                    bases = [b.id if isinstance(b, ast.Name) else (b.attr if isinstance(b, ast.Attribute) else str(b)) for b in node.bases]
                    methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                    chunks.append({
                        "kind": "Class",
                        "name": node.name,
                        "file": path_hint,
                        "docstring": doc,
                        "methods": methods,
                        "bases": bases,
                    })
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    doc = ast.get_docstring(node) or ""
                    args = [a.arg for a in node.args.args]
                    parent = getattr(node, "parent", None)
                    in_class = parent.name if isinstance(parent, ast.ClassDef) else ""
                    chunks.append({
                        # Methods get their own chunk: without one they are only
                        # reachable as a name in the class's "Methods:" list.
                        "kind": "Method" if in_class else "Function",
                        "name": node.name,
                        "file": path_hint,
                        "args": args,
                        "docstring": doc,
                        "in_class": in_class,
                        "body": cls._python_body_lines(node, path_hint),
                    })
        except SyntaxError as e:
            chunks.append(f"[file] {path_hint} (parse error: {e})")
        return chunks

    # --- Fallback: line-based chunking ---

    @classmethod
    def _parse_fallback(cls, source: str, path_hint: str = "", label: str = "") -> list[dict | str]:
        chunks: list[dict | str] = []
        blabel = label or _LABELS["default"].get("file", "File")
        lines = source.split('\n')
        block_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped:
                block_lines.append(stripped)
            else:
                if block_lines:
                    block_text = ' '.join(block_lines)
                    if len(block_text) >= 20:
                        chunks.append({
                            "kind": blabel,
                            "name": "",
                            "file": path_hint,
                            "body": block_text[:512],
                        })
                    block_lines = []
        if block_lines:
            block_text = ' '.join(block_lines)
            if len(block_text) >= 20:
                chunks.append({
                    "kind": blabel,
                    "name": "",
                    "file": path_hint,
                    "body": block_text[:512],
                })
        return chunks

    # --- C/C++ handler (tree-sitter) ---

    _CPP_SIGNIFICANT = {
        "class_specifier", "struct_specifier",
        "function_definition", "template_function",
        "template_method", "enum_specifier",
        "alias_declaration", "declaration",
    }

    _CPP_FUNC_TYPES = ("function_definition", "template_function", "template_method")

    @classmethod
    def _parse_cpp(cls, source: str, path_hint: str = "") -> list[dict | str]:
        """Clang when enabled, tree-sitter otherwise; line chunks as last resort."""
        if cls._use_clang and _CLANG_AVAILABLE:
            chunks = cls._parse_cpp_clang(source, path_hint)
            if chunks:
                return chunks
            # Clang needs the project's real include paths and flags to resolve a
            # translation unit; without them it errors on most files. Structured
            # tree-sitter output beats dumping raw source lines into the index.
            log.debug("clang yielded nothing for %s, using tree-sitter", path_hint)
        return cls._parse_cpp_treesitter(source, path_hint)

    @classmethod
    def _parse_cpp_treesitter(cls, source: str, path_hint: str = "") -> list[dict | str]:
        if not _TS_AVAILABLE:
            return cls._parse_fallback(source, path_hint, "Block")
        try:
            from tree_sitter import Language, Parser
            import tree_sitter_cpp
            lang = Language(tree_sitter_cpp.language())
        except Exception as e:
            log.warning(
                "tree-sitter C++ grammar unavailable (%s), using line-based chunking for %s",
                e, path_hint,
            )
            return cls._parse_fallback(source, path_hint, "Block")

        chunks: list[dict | str] = []
        try:
            parser = Parser(lang)
            tree = parser.parse(source.encode("utf8", errors="ignore"))
            root = tree.root_node

            def walk(node, class_name="", in_body=False):
                # A `declaration` inside a function body is a local variable, not an
                # API surface; indexing those is the noise we are trying to avoid.
                if node.type in cls._CPP_SIGNIFICANT and not (
                        in_body and node.type == "declaration"):
                    name_node = node.child_by_field_name("name")
                    name = node_text(name_node) if name_node else ""
                    qualifier = ""
                    if not name:
                        decl = node.child_by_field_name("declarator")
                        name, qualifier = ts_declarator_name(
                            decl if decl is not None else node)
                    doc = ""
                    for c in node.children:
                        if c.type == "comment":
                            doc = node_text(c)
                            break
                    if name:
                        params = node.child_by_field_name("parameters")
                        if not params and node.type in cls._CPP_FUNC_TYPES:
                            decl = node.child_by_field_name("declarator")
                            if decl:
                                params = decl.child_by_field_name("parameters")
                        sig_text = node_text(params) if params else ""

                        label = label_for(node.type)
                        body_summary = ts_body_summary(node)
                        bases_list = ts_base_classes(node)[0] if node.type in ("class_specifier", "struct_specifier") else []
                        # An out-of-line definition names its class in the
                        # qualifier (HashMap::find) rather than by nesting.
                        owner = class_name or qualifier
                        chunks.append({
                            "kind": label,
                            "name": name,
                            "file": path_hint,
                            "signature": sig_text,
                            "docstring": doc.strip() if doc else "",
                            "body": body_summary,
                            "bases": bases_list,
                            "in_class": owner if owner and node.type in cls._CPP_FUNC_TYPES else "",
                        })
                next_class = class_name
                if node.type in ("class_specifier", "struct_specifier"):
                    nname = node.child_by_field_name("name")
                    if nname:
                        next_class = node_text(nname)
                for c in node.children:
                    walk(c, next_class, in_body or node.type == "compound_statement")
            walk(root)
        except Exception as e:
            chunks.append(f"[file] {path_hint} (parse error: {e})")
        return chunks if chunks else cls._parse_fallback(source, path_hint, "Block")

    @classmethod
    def _parse_cpp_clang(cls, source: str, path_hint: str = "") -> list[str]:
        """Chunks from libclang, or [] so the caller can fall back to tree-sitter."""
        if not _CLANG_AVAILABLE:
            return []
        try:
            import clang.cindex as ci
            _configure_clang(ci)
            index = ci.Index.create()
            # Parse the source as an unsaved file
            unsaved_file = (path_hint, source)
            tu = index.parse(path_hint, unsaved_files=[unsaved_file], args=['-x', 'c++', '-std=c++17'])
            if tu.diagnostics:
                for diag in tu.diagnostics:
                    if diag.severity >= ci.Diagnostic.Error:
                        log.debug("clang reported errors for %s", path_hint)
                        return []
        except Exception as e:
            log.warning("clang parse of %s failed (%s)", path_hint, e)
            return []

        chunks: list[str] = []
        try:
            def walk(cursor):
                if cursor.kind in ASTParser._CLANG_SIGNIFICANT:
                    name = cursor.spelling
                    # Get comment: try to get raw comment
                    doc = cursor.raw_comment if hasattr(cursor, 'raw_comment') else ""
                    # Clean up comment delimiters if present
                    if doc:
                        # Remove common comment delimiters: /* ... */ or // ...
                        # Simple stripping: remove leading/trailing whitespace and common markers
                        doc = doc.strip()
                        if doc.startswith('/*') and doc.endswith('*/'):
                            doc = doc[2:-2].strip()
                        elif doc.startswith('//'):
                            doc = doc[2:].strip()
                    # Get signature for functions: parameter types only
                    signature = ""
                    if cursor.kind == ci.CursorKind.FUNCTION_DECL:
                        try:
                            func_type = cursor.type
                            arg_types = []
                            for arg_type in func_type.argument_types():
                                arg_types.append(arg_type.spelling)
                            signature = f"({', '.join(arg_types)})"
                        except Exception as e:
                            log.debug("cannot build signature in %s: %s", path_hint, e)
                            signature = ""
                    # Get base classes for class/struct
                    extra = []
                    if cursor.kind in (ci.CursorKind.CLASS_DECL, ci.CursorKind.STRUCT_DECL):
                        bases = []
                        for c in cursor.get_children():
                            if c.kind == ci.CursorKind.CXX_BASE_SPECIFIER:
                                try:
                                    bases.append(c.type.spelling)
                                except Exception as e:
                                    log.debug("cannot read base class in %s: %s", path_hint, e)
                        if bases:
                            extra.append(": public " + ", public ".join(bases))

                    # Get body summary: first few member names
                    if cursor.kind in (ci.CursorKind.CLASS_DECL, ci.CursorKind.STRUCT_DECL):
                        members = []
                        for c in cursor.get_children():
                            if c.kind in (ci.CursorKind.FIELD_DECL, ci.CursorKind.CXX_METHOD):
                                members.append(c.spelling)
                                if len(members) >= 5:
                                    break
                        if members:
                            extra.append(f"{{ {', '.join(members)} }}")

                    # Natural language text for embedding
                    kind_map = {"class_decl": "Class", "struct_decl": "Struct", "function_decl": "Function",
                                "enum_decl": "Enum", "typedef_decl": "Type", "type_alias_decl": "Alias"}
                    human_kind = kind_map.get(cursor.kind.name.lower(), cursor.kind.name.lower())
                    text = f"{human_kind} {name} in {path_hint}"
                    if extra:
                        text += ". " + ". ".join(extra)
                    if signature:
                        text += f". Signature: {signature}"
                    if doc:
                        text += f". Doc: {doc}"
                    chunks.append(text)
                # Recurse
                for child in cursor.get_children():
                    walk(child)
            walk(tu.cursor)
        except Exception as e:
            chunks.append(f"[file] {path_hint} (clang parse error: {e})")
        return chunks

    # --- Tree-sitter based handler (JS / Go / Rust) ---

    @classmethod
    def _parse_treesitter(cls, source: str, path_hint: str = "") -> list[dict | str]:
        if not _TS_AVAILABLE:
            return []

        ext = os.path.splitext(path_hint)[1].lower()
        lang_config = _TS_LANGUAGES.get(ext)
        if not lang_config:
            return []

        lang_name, lang_fn = lang_config
        significant = _TS_SIGNIFICANT.get(lang_name, set())

        chunks: list[dict | str] = []
        try:
            parser = Parser(Language(lang_fn()))
            tree = parser.parse(source.encode("utf8", errors="ignore"))
            root = tree.root_node

            def walk(node):
                if node.type in significant:
                    name_node = node.child_by_field_name("name")
                    name = node_text(name_node) if name_node else ""
                    doc = ""
                    for c in node.children:
                        if c.type == "comment":
                            doc = node_text(c)
                            break
                    if name:
                        params = node.child_by_field_name("parameters")
                        sig_text = node_text(params) if params else ""
                        label = label_for(node.type, f"[{node.type}]")
                        chunks.append({
                            "kind": label,
                            "name": name,
                            "file": path_hint,
                            "signature": sig_text,
                            "docstring": doc.strip() if doc else "",
                        })
                for c in node.children:
                    walk(c)
            walk(root)
        except Exception as e:
            chunks.append(f"[file] {path_hint} (parse error: {e})")
        return chunks


# Register default handlers
ASTParser.register('*.py', ASTParser._parse_python)
ASTParser.register('*.cpp', ASTParser._parse_cpp)
ASTParser.register('*.cc', ASTParser._parse_cpp)
ASTParser.register('*.cxx', ASTParser._parse_cpp)
ASTParser.register('*.h', ASTParser._parse_cpp)
ASTParser.register('*.hpp', ASTParser._parse_cpp)
ASTParser.register('*.js', ASTParser._parse_treesitter)
ASTParser.register('*.jsx', ASTParser._parse_treesitter)
ASTParser.register('*.ts', ASTParser._parse_treesitter)
ASTParser.register('*.tsx', ASTParser._parse_treesitter)
ASTParser.register('*.mjs', ASTParser._parse_treesitter)
ASTParser.register('*.cjs', ASTParser._parse_treesitter)
ASTParser.register('*.go', ASTParser._parse_treesitter)
ASTParser.register('*.rs', ASTParser._parse_treesitter)


class EmbeddingModel:
    """Wraps a SentenceTransformer model for producing embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None,
                 query_prefix: str | None = None, passage_prefix: str | None = None,
                 float_type: str = "fp32"):
        self.model = SentenceTransformer(model_name, device=device)
        if float_type == "fp16":
            self.model.half()
        self.query_prefix = query_prefix if query_prefix is not None else self._detect_query_prefix(model_name)
        self.passage_prefix = passage_prefix if passage_prefix is not None else self._detect_passage_prefix(model_name)

        target = 512
        try:
            arch_max = self.model._first_module().auto_model.config.max_position_embeddings
            target = min(target, arch_max)
        except Exception as e:
            log.debug(
                "cannot read max_position_embeddings for %s, using %d: %s",
                model_name, target, e,
            )

        self.model.max_seq_length = target
        self.model.tokenizer.model_max_length = target
        self.dim = self.model.get_embedding_dimension()

    @staticmethod
    def _detect_query_prefix(model_name: str) -> str:
        name = model_name.lower()
        if 'e5' in name:
            return 'query: '
        if 'bge' in name or 'bce' in name:
            return 'Represent this sentence for searching relevant passages: '
        return ''

    @staticmethod
    def _detect_passage_prefix(model_name: str) -> str:
        name = model_name.lower()
        if 'e5' in name:
            return 'passage: '
        return ''

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)

    def embed_many(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)

    def as_passages(self, texts: list[str]) -> list[str]:
        """Apply the model's passage instruction prefix to indexable texts."""
        return [self.passage_prefix + t for t in texts] if self.passage_prefix else texts

    def embed_passage(self, text: str) -> np.ndarray:
        return self.embed(self.passage_prefix + text if self.passage_prefix else text)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed(self.query_prefix + query if self.query_prefix else query)

    @classmethod
    def from_config(cls, cfg: "ModelConfig", device: Optional[str] = None) -> "EmbeddingModel":
        return cls(cfg.model_name, device=cfg.device if device is None else device,
                   query_prefix=cfg.query_prefix, passage_prefix=cfg.passage_prefix,
                   float_type=cfg.float_type)


_EMPTY_MATRIX = np.empty((0, 0), dtype=np.float32)


def _as_matrix(vecs) -> np.ndarray:
    """Coerce vectors (2-D array, single row, or sequence of rows) to one (N, dim) array."""
    if not isinstance(vecs, np.ndarray):
        vecs = np.stack(vecs) if len(vecs) else _EMPTY_MATRIX
    if vecs.ndim == 1:
        return vecs.reshape(1, -1) if vecs.size else _EMPTY_MATRIX
    return vecs if vecs.size else _EMPTY_MATRIX


class VectorStore:
    """In-memory vector store with cosine similarity search.

    Vectors live in one (N, dim) matrix rather than a list of per-row arrays with
    a stacked cache beside it: that layout held the same data twice, so a 375k
    chunk index needed ~1.9 GB resident for a 549 MiB matrix. Appends collect in
    `_pending` and are concatenated on the next read.
    """

    def __init__(self):
        self._matrix: np.ndarray = _EMPTY_MATRIX
        self._pending: list[np.ndarray] = []
        self.texts: list[str] = []
        self.node_ids: list[int | None] = []

    @property
    def vectors(self) -> np.ndarray:
        """Every vector as a single (N, dim) array."""
        return self._get_array()

    @vectors.setter
    def vectors(self, value) -> None:
        self._matrix = _as_matrix(value)
        self._pending = []

    @property
    def dim(self) -> int:
        source = self._matrix if len(self._matrix) else next(iter(self._pending), _EMPTY_MATRIX)
        return int(source.shape[1]) if len(source) else 0

    def _get_array(self) -> np.ndarray:
        if self._pending:
            parts = ([self._matrix] if len(self._matrix) else []) + self._pending
            self._matrix = parts[0] if len(parts) == 1 else np.concatenate(parts)
            self._pending = []
        return self._matrix

    def add(self, vec: np.ndarray, text: str, node_id: int | None = None) -> None:
        self._pending.append(_as_matrix(vec))
        self.texts.append(text)
        self.node_ids.append(node_id)

    def add_many(self, vecs: np.ndarray, texts: list[str], node_ids: list[int | None] | None = None) -> None:
        block = _as_matrix(vecs)
        if len(block):
            self._pending.append(block)
        self.texts.extend(texts)
        self.node_ids.extend(node_ids if node_ids is not None else [None] * len(texts))

    def truncate(self, size: int) -> None:
        """Keep the first `size` entries, dropping the tail (used to clear a delta)."""
        self._matrix = self._get_array()[:size]
        self.texts = self.texts[:size]
        self.node_ids = self.node_ids[:size]

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[dict]:
        array = self._get_array()
        if array.size == 0:
            return []
        scores = np.dot(array, query_vec)
        top_idxs = np.argsort(scores)[-top_k:][::-1]
        return [
            {"text": self.texts[i], "score": float(scores[i]), "idx": i,
             "node_id": self.node_ids[i] if i < len(self.node_ids) else None,
             "method": "embed"}
            for i in top_idxs
        ]

    def __len__(self) -> int:
        return len(self._matrix) + sum(len(block) for block in self._pending)


class StorageIO:
    """Save/load vectors, texts, node_ids and dimension to/from .npz files.

    Texts are stored as a length-prefixed UTF-8 blob: smaller and faster than a
    numpy object array, and it loads without pickle.
    """

    @staticmethod
    def save(path: str, vectors: np.ndarray | list[np.ndarray], texts: list[str], dim: int,
             node_ids: list[int | None] | None = None) -> None:
        if isinstance(vectors, list):
            vecs_array = np.stack(vectors) if vectors else np.array([])
        else:
            vecs_array = np.asarray(vectors)
        offsets = np.empty(len(texts) + 1, dtype=np.int64)
        offsets[0] = 0
        blob = bytearray()
        for i, text in enumerate(texts):
            blob.extend(text.encode("utf-8"))
            offsets[i + 1] = len(blob)
        data = {
            "dim": np.array(dim),
            "vectors": vecs_array,
            "texts_blob": np.frombuffer(blob, dtype=np.uint8),
            "texts_offsets": offsets,
        }
        if node_ids is not None:
            data["node_ids"] = np.array(
                [nid if nid is not None else -1 for nid in node_ids], dtype=np.int32
            )
        np.savez_compressed(path, **data)

    @staticmethod
    def load(path: str) -> tuple:
        """Load vectors, texts, node_ids and dimension from a .npz file."""
        try:
            data = np.load(path, allow_pickle=False)
        except ValueError as e:
            log.error("cannot load index %s: %s", path, e)
            raise
        # Return the (N, dim) array as loaded: splitting it into per-row views kept
        # the whole matrix alive behind 375k small objects, and the caller stacked a
        # second copy of it anyway.
        vectors = data["vectors"]
        if "texts_blob" in data:
            blob = data["texts_blob"].tobytes()
            offsets = data["texts_offsets"]
            texts = [
                blob[offsets[i]:offsets[i + 1]].decode("utf-8")
                for i in range(len(offsets) - 1)
            ]
        else:
            # Indices written before the blob format stored texts as an object array.
            texts = [str(t) for t in np.load(path, allow_pickle=True)["texts"]]
        dim = int(data["dim"])
        node_ids = None
        if "node_ids" in data:
            raw = data["node_ids"]
            node_ids = [int(x) if x >= 0 else None for x in raw]
        return vectors, texts, dim, node_ids


def build_flat_index(root: str, data_dir: str | None = None, delta: bool = False) -> None:
    """Build flat vector index for a project.

    Args:
        root: Project root directory.
        data_dir: Directory to save the index, used as given (relative paths are
                  relative to the cwd, matching build_all and tree_ast_parser).
                  If not given, computed from config embedding_store + project name.
        delta: If True, build delta index (only changed files).
    """
    data_dir = resolve_data_dir(root, data_dir)
    cfg = ModelConfig.load(root)
    enc = EmbeddingModel.from_config(cfg)

    os.makedirs(data_dir, exist_ok=True)
    project = root
    if delta:
        # --- Delta mode: only changed files ---
        changed = changed_files(project)
        delta_vec_path = os.path.join(data_dir, 'delta.npz')
        delta_texts_path = os.path.join(data_dir, 'delta_texts.json')

        def _save_empty_delta(files: list[str]) -> None:
            StorageIO.save(delta_vec_path, [], [], enc.dim)
            with open(delta_texts_path, 'w') as f:
                json.dump({"files": files, "texts": [], "model": cfg.model_name}, f)

        if not changed:
            print("Delta: no changed files")
            _save_empty_delta([])
            return

        print(f"Delta: {len(changed)} changed files")
        chunks: list[str] = []
        strategy = ASTParser._load_strategy(project)
        for fp in changed:
            abspath = os.path.join(project, fp)
            if not os.path.isfile(abspath):
                continue
            try:
                chunks.extend(enrich_chunks(ASTParser.parse_file(abspath, path_hint=fp), strategy))
            except Exception as e:
                chunks.append(f"[file] {fp} (read error: {e})")

        if not chunks:
            print("Delta: no parseable chunks")
            _save_empty_delta(changed)
            return

        vecs = enc.embed_many(enc.as_passages(chunks))

        StorageIO.save(delta_vec_path, vecs, chunks, enc.dim)

        with open(delta_texts_path, 'w') as f:
            json.dump({"files": changed, "texts": chunks, "model": cfg.model_name}, f,
                      ensure_ascii=False)

        print(f"Delta index: {len(chunks)} chunks from {len(changed)} files -> {delta_vec_path}")
    else:
        # --- Full rebuild ---
        chunks = ASTParser.scan_project(project)

        node_ids = [None] * len(chunks)
        if os.path.exists(os.path.join(data_dir, "tree_index.json")):
            node_ids = add_tree_context(chunks, data_dir)

        vecs = enc.embed_many(enc.as_passages(chunks))

        out = os.path.join(data_dir, 'enriched_vectors.npz')
        StorageIO.save(out, vecs, chunks, enc.dim, node_ids=node_ids)
        print(f"Flat index: {len(chunks)} chunks -> {out}")


def _parse_file_worker(args):
    """Worker function for parallel file parsing (CPU only, no GPU).

    Returns (tree_nodes, flat_chunks, rel, errors); errors describe per-file
    failures so the parent process can report them instead of losing them.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    fp, rel, root, tree_exts, exclude = args
    errors: list[str] = []
    if any(x in fp for x in exclude):
        return [], [], rel, errors

    tree_nodes = []
    flat_chunks = []
    strategy = ASTParser._load_strategy(root)

    # Flat chunks (all supported files)
    try:
        flat_chunks = enrich_chunks(ASTParser.parse_file(fp, path_hint=rel), strategy)
    except Exception as e:
        errors.append(f"flat parse of {rel} failed: {e!r}")

    # Tree nodes (code languages only)
    if fp.endswith(tree_exts):
        try:
            from tree_ast_parser import parse_file as tree_parse_file
            nodes = tree_parse_file(fp, root=root)
            tree_nodes = nodes
        except Exception as e:
            errors.append(f"tree parse of {rel} failed: {e!r}")

    return tree_nodes, flat_chunks, rel, errors


def _parse_files(root: str, num_workers: int | None = None,
                 exclude=DEFAULT_EXCLUDE) -> tuple:
    """Phase 1: parse source files (CPU only, no GPU model)."""
    from tree_ast_parser import LANGUAGES as TREE_LANGS
    tree_exts = tuple(TREE_LANGS.keys())
    file_list = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ASTParser.SKIP_DIRS]
        for fn in sorted(filenames):
            fp = os.path.join(dirpath, fn)
            rel = os.path.relpath(fp, root)
            if any(x in fp for x in exclude):
                continue
            file_list.append((fp, rel, root, tree_exts, exclude))

    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 12) - 1)
    print(f"Parsing {len(file_list)} files with {num_workers} workers...", flush=True)
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    from concurrent.futures import ProcessPoolExecutor, as_completed
    all_tree_nodes = []
    chunks = []
    tree_node_count = 0
    failed_files = 0
    parse_errors = 0
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
        futures = {pool.submit(_parse_file_worker, f): f[0] for f in file_list}
        for i, f in enumerate(as_completed(futures)):
            try:
                tree_nodes, flat_chunks, _, errors = f.result()
            except Exception:
                failed_files += 1
                log.warning("worker crashed on %s", futures[f], exc_info=True)
                continue
            for err in errors:
                parse_errors += 1
                log.warning("%s", err)
            if tree_nodes:
                all_tree_nodes.append(tree_nodes)
                tree_node_count += len(tree_nodes)
            if flat_chunks:
                chunks.extend(flat_chunks)
            if (i + 1) % max(1, len(file_list) // 80) == 0 or i == len(file_list) - 1:
                print(f"  [{i+1}/{len(file_list)}] files, {tree_node_count} tree nodes, {len(chunks)} chunks", flush=True)

    # Renumber tree nodes: local IDs -> global IDs
    tree_texts = []
    global_id = 0
    for file_nodes in all_tree_nodes:
        old_to_new = {}
        for n in file_nodes:
            old_to_new[n["id"]] = global_id
            n["id"] = global_id
            global_id += 1
        for n in file_nodes:
            pid = n.get("parent_id", -1)
            n["parent_id"] = old_to_new.get(pid, -1)
        tree_texts.extend(n["text"] for n in file_nodes)
    all_tree_nodes = [n for nodes in all_tree_nodes for n in nodes]
    if failed_files or parse_errors:
        print(f"  WARNING: {failed_files} files crashed a worker, {parse_errors} parse errors", flush=True)
    if file_list and failed_files == len(file_list):
        raise RuntimeError(
            f"all {len(file_list)} files failed to parse; see log for details"
        )
    print(f"  parsed {len(all_tree_nodes)} tree nodes, {len(chunks)} flat chunks", flush=True)
    return all_tree_nodes, tree_texts, chunks


def build_all(root: str, data_dir: str | None = None, num_workers: int | None = None,
              embed_mode: str = "multi",
              exclude=DEFAULT_EXCLUDE) -> None:
    """Build both tree and flat indices.

    Phase 1: parse source files (CPU only, no GPU).
    Phase 2: load model, embed, save indices.
    """
    data_dir = resolve_data_dir(root, data_dir)
    cfg = ModelConfig.load(root)

    # Infer mode from device
    if embed_mode is None:
        if cfg.device and cfg.device.startswith("cuda"):
            embed_mode = "gpu"
        elif cfg.device == "cpu":
            embed_mode = "cpu"
        else:
            import torch
            embed_mode = "multi" if torch.cuda.is_available() else "cpu"

    os.makedirs(data_dir, exist_ok=True)
    tree_json_path = os.path.join(data_dir, "tree_index.json")
    tree_exists = os.path.exists(tree_json_path)
    BATCH = cfg.batch_size

    # Phase 1: parse files (no GPU) — always needed for flat chunks
    all_tree_nodes, tree_texts, chunks = _parse_files(root, num_workers, exclude)

    # Phase 2: embed and save
    print(f"Loading models (mode={embed_mode})...", flush=True)
    enc_gpu = None
    enc_cpu = None
    if embed_mode in ("multi", "gpu"):
        enc_gpu = EmbeddingModel.from_config(cfg, device=cfg.device or "cuda")
    if embed_mode in ("multi", "cpu"):
        enc_cpu = EmbeddingModel.from_config(cfg, device="cpu")

    if tree_exists and os.path.exists(os.path.join(data_dir, "tree_vectors.npz")):
        print("Tree index exists, skipping tree embedding", flush=True)
    else:
        enc_tree = enc_gpu or enc_cpu
        # Batch-embed tree texts (pre-allocated to avoid list-of-arrays)
        tree_vecs = np.array([])
        if tree_texts and enc_tree is not None:
            embed_texts = enc_tree.as_passages(tree_texts)
            dim = enc_tree.dim
            n = len(embed_texts)
            tree_vecs = np.empty((n, dim), dtype=np.float32)
            for i in range(0, n, BATCH):
                batch = embed_texts[i:i + BATCH]
                tree_vecs[i:i + len(batch)] = enc_tree.embed_many(batch)

        # Save tree index
        tree_vec_path = os.path.join(data_dir, "tree_vectors.npz")
        if tree_vecs.size and enc_tree is not None:
            StorageIO.save(tree_vec_path, tree_vecs, tree_texts, enc_tree.dim)
            write_tree_index(tree_json_path, all_tree_nodes, tree_texts)
            print(f"Tree index: {len(all_tree_nodes)} nodes -> {tree_vec_path} + {tree_json_path}", flush=True)
        else:
            print("No tree nodes found")

    # Enrich flat chunks with tree context and build node_ids
    node_ids = add_tree_context(chunks, data_dir, progress_steps=40)

    # Flat chunk embedding
    print(f"  embedding {len(chunks)} flat chunks (mode={embed_mode})...", flush=True)

    def _embed_sequential(model: EmbeddingModel, texts: list[str]):
        dim = model.dim
        n = len(texts)
        out = np.empty((n, dim), dtype=np.float32)
        for i in range(0, n, BATCH):
            batch = texts[i:i + BATCH]
            out[i:i + len(batch)] = model.embed_many(batch)
            print(f"    [{min(i+BATCH, n)}/{n}]", flush=True)
        return out

    embed_chunks = (enc_gpu or enc_cpu).as_passages(chunks)

    if embed_mode == "multi":
        from concurrent.futures import ThreadPoolExecutor
        import threading
        import queue
        chunk_queue = queue.Queue()
        for i in range(0, len(embed_chunks), BATCH):
            chunk_queue.put((i, embed_chunks[i:i + BATCH]))
        results = []
        results_lock = threading.Lock()
        _done = 0

        def _embed_worker(model: EmbeddingModel, name: str):
            nonlocal _done
            import torch
            if name == "CPU":
                torch.set_num_threads(10)
            while True:
                try:
                    idx, batch = chunk_queue.get_nowait()
                except queue.Empty:
                    return
                vecs = model.embed_many(batch)
                with results_lock:
                    results.append((idx, vecs))
                    _done += len(batch)
                    print(f"    {name}: {_done}/{len(embed_chunks)}", flush=True)
                chunk_queue.task_done()

        pool = ThreadPoolExecutor(max_workers=2)
        worker_futures = []
        if enc_gpu is not None:
            worker_futures.append(pool.submit(_embed_worker, enc_gpu, "GPU"))
        if enc_cpu is not None:
            worker_futures.append(pool.submit(_embed_worker, enc_cpu, "CPU"))
        try:
            pool.shutdown()
        except KeyboardInterrupt:
            print("\nInterrupted, exiting...", flush=True)
            os._exit(130)
        # Surface worker exceptions instead of saving a partial index.
        for fut in worker_futures:
            fut.result()
        results.sort(key=lambda x: x[0])
        flat_dim = (enc_gpu or enc_cpu).dim
        vecs = np.empty((len(embed_chunks), flat_dim), dtype=np.float32)
        pos = 0
        for _, batch_vecs in results:
            vecs[pos:pos + len(batch_vecs)] = batch_vecs
            pos += len(batch_vecs)
        if pos != len(embed_chunks):
            raise RuntimeError(
                f"embedding incomplete: {pos}/{len(embed_chunks)} chunks embedded"
            )
    elif embed_mode == "gpu":
        vecs = _embed_sequential(enc_gpu, embed_chunks)
    else:
        vecs = _embed_sequential(enc_cpu, embed_chunks)

    out = os.path.join(data_dir, "enriched_vectors.npz")
    flat_dim = (enc_gpu or enc_cpu).dim
    StorageIO.save(out, vecs, chunks, flat_dim, node_ids=node_ids)
    print(f"Flat index: {len(chunks)} chunks -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build vector index for a project.')
    parser.add_argument('--build-flat', action='store_true', help='Build the flat index')
    parser.add_argument('--build-all', action='store_true', help='Build tree + flat index in one pass')
    parser.add_argument('--delta', action='store_true', help='Build delta index (only changed files)')
    parser.add_argument('--data-dir', default=None, help='Directory to save the index (default: from config)')
    parser.add_argument('--root', default='.', help='Project root directory')
    parser.add_argument('--workers', type=int, default=10, help='Number of worker processes for file parsing (default: 10)')
    parser.add_argument('--embed-mode', choices=['multi', 'gpu', 'cpu'], default=None,
                        help='Embedding mode: multi (GPU+CPU, default), gpu-only, cpu-only')
    args = parser.parse_args()

    logging.basicConfig(
        level=os.environ.get("EMBEDDER_LOG_LEVEL", "WARNING").upper(), stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.build_all:
        build_all(args.root, args.data_dir, num_workers=args.workers, embed_mode=args.embed_mode)
    elif args.build_flat:
        build_flat_index(args.root, args.data_dir, args.delta)
    else:
        parser.error('Please specify --build-flat')
