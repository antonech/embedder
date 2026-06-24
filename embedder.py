import os, json, ast, re as _re, subprocess, argparse, sys

import numpy as np
from typing import Optional
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer

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
_CLANG_AVAILABLE = False
_USE_CLANG = False  # can be enabled via config.json
try:
    import clang.cindex as ci
    # Try to set the library path to common locations
    for libpath in ['/usr/lib/llvm-18/lib', '/usr/lib/llvm-15/lib', '/usr/lib/x86_64-linux-gnu']:
        try:
            ci.Config.set_library_path(libpath)
            break
        except:
            pass
    _CLANG_AVAILABLE = True
    # Define significant cursor kinds for clang (if available)
    _CLANG_SIGNIFICANT = {
        ci.CursorKind.CLASS_DECL,
        ci.CursorKind.STRUCT_DECL,
        ci.CursorKind.UNION_DECL,
        ci.CursorKind.CXX_METHOD,
        ci.CursorKind.CONSTRUCTOR,
        ci.CursorKind.FUNCTION_DECL,
        ci.CursorKind.FUNCTION_TEMPLATE,
        ci.CursorKind.ENUM_DECL,
        ci.CursorKind.TYPEDEF_DECL,
        ci.CursorKind.TYPE_ALIAS_DECL,
    }
except Exception:
    _CLANG_AVAILABLE = False
    _CLANG_SIGNIFICANT = set()

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


class BodyStrategy(EnrichmentStrategy):
    """Methods, fields, bases, and body lines as summary."""

    MAX_LINES = 5

    @classmethod
    def key(cls) -> str:
        return "body"

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
        body = node.get("body", [])
        if isinstance(body, list):
            lines = [str(b)[:100] for b in body if str(b).strip()]
            for line in lines[:self.MAX_LINES]:
                parts.append(line)
        elif isinstance(body, str) and body.strip():
            parts.append(body[:200])
        return " | ".join(parts)


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


_LABELS_PATH = os.path.join(os.path.dirname(__file__), "labels.json")
if os.path.exists(_LABELS_PATH):
    with open(_LABELS_PATH) as f:
        _LABELS = json.load(f)
else:
    _LABELS = {"default": {"file": "[file]", "line": "[line]", "fallback": "[chank]"}, "mapping": {}}

class ASTParser:
    """Extract enriched chunks from source files for any language.

    Supports Python (native AST), C/C++ (regex-based), and falls back to
    line-based chunking for unknown languages. Add new languages by
    registering a handler: ASTParser.register('*.ext', handler_fn).
    """

    SKIP_DIRS = {'venv', '.git', '__pycache__', 'node_modules'}
    SKIP_PREFIXES = {'[line]', '[file]'}

    _handlers: dict[str, callable] = {}
    _use_clang = False  # class variable, set by scan_project from config

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
        except Exception:
            return []
        if not content.strip() or '\x00' in content[:2000]:
            return []
        return cls._handler_for(filepath)(content, path_hint or filepath)

    @classmethod
    def _load_strategy(cls, root: str, enrichment_keys: Optional[list[str]] = None) -> CompositeStrategy:
        if enrichment_keys is None:
            cfg_path = os.path.join(root, "config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    cfg = json.load(f)
                enrichment_keys = cfg.get("enrichment")
                # Set the use_clang flag from config
                cls._use_clang = cfg.get("use_clang", False)
            else:
                cls._use_clang = False
        else:
            # If enrichment_keys is provided, we still set the flag from config if available
            cfg_path = os.path.join(root, "config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    cfg = json.load(f)
                cls._use_clang = cfg.get("use_clang", False)
            else:
                cls._use_clang = False
        return CompositeStrategy.from_keys(enrichment_keys or ["signature", "body", "docstring"])

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
        file_label = _LABELS["default"].get("file", "[file]")
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in cls.SKIP_DIRS]
            for fn in sorted(filenames):
                fp = os.path.join(dirpath, fn)
                name = os.path.relpath(fp, root)
                # Parse every file; unregistered types (XML, .sh, .md) use fallback
                try:
                    file_chunks = cls.parse_file(fp, path_hint=name)
                    if file_chunks:
                        for c in file_chunks:
                            if isinstance(c, str):
                                if c.startswith(tuple(cls.SKIP_PREFIXES)):
                                    continue
                                chunks.append(c)
                            else:
                                enriched = strategy.enrich(c)
                                kind = c.get("kind", "")
                                name = c.get("name", "")
                                f = c.get("file", "")
                                prefix = f"{kind} {f} {name}".strip()
                                if prefix:
                                    enriched = f"{prefix} | {enriched}" if enriched else prefix
                                chunks.append(enriched)
                except Exception as e:
                    pass
        return chunks

    # --- Python handler ---

    class _ParentVisitor(ast.NodeTransformer):
        def visit(self, node):
            for child in ast.iter_child_nodes(node):
                child.parent = node
            return super().visit(node)

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
                    if hasattr(node, 'parent') and isinstance(node.parent, ast.ClassDef):
                        continue
                    doc = ast.get_docstring(node) or ""
                    args = [a.arg for a in node.args.args]
                    try:
                        body_lines = [ast.unparse(s)[:80] for s in node.body[:2]]
                    except Exception:
                        body_lines = []
                    chunks.append({
                        "kind": "Function",
                        "name": node.name,
                        "file": path_hint,
                        "args": args,
                        "docstring": doc,
                        "body": body_lines,
                    })
        except SyntaxError as e:
            chunks.append(f"[file] {path_hint} (parse error: {e})")
        return chunks

    # --- Fallback: line-based chunking ---

    @classmethod
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

    @classmethod
    def _parse_cpp(cls, source: str, path_hint: str = "") -> list[str]:
        # Use clang if enabled and available
        if cls._use_clang and _CLANG_AVAILABLE:
            return cls._parse_cpp_clang(source, path_hint)
        # Otherwise use tree-sitter
        if not _TS_AVAILABLE:
            return cls._parse_fallback(source, path_hint, "Block")
        try:
            from tree_sitter import Language, Parser
            import tree_sitter_cpp
            lang = Language(tree_sitter_cpp.language())
        except Exception:
            return cls._parse_fallback(source, path_hint, "Block")

        chunks: list[str] = []
        try:
            parser = Parser(lang)
            tree = parser.parse(source.encode("utf8", errors="ignore"))
            root = tree.root_node

            def _ts_base_classes(n):
                bases = []
                for c in n.children:
                    if c.type == "base_class_clause":
                        for cc in c.children:
                            if cc.type == "type_identifier":
                                bases.append(cc.text.decode("utf8", errors="ignore"))
                if bases:
                    return ": " + ", ".join(bases)
                return ""

            def _ts_body_summary(n):
                body = None
                for c in n.children:
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
                        decl = c.child_by_field_name("declarator")
                        if decl:
                            fid = decl.child_by_field_name("name")
                            if not fid:
                                for gc in decl.children:
                                    if gc.type == "field_identifier":
                                        fid = gc
                                        break
                            if fid:
                                mname = fid.text.decode()
                                param_list = decl.child_by_field_name("parameters")
                                sig = param_list.text.decode("utf8", errors="ignore")[:40] if param_list else ""
                                methods.append(f"{access}: {mname}{sig}")
                    elif c.type == "field_declaration":
                        txt = c.text.decode("utf8", errors="ignore").strip()
                        # Check if it's a method declaration (has parentheses)
                        if "(" in txt and txt.split("(")[0].strip().split()[-1]:
                            mname = txt.split("(")[0].strip().split()[-1]
                            sig = "(" + txt.split("(")[1][:40]
                            methods.append(f"{access}: {mname}{sig}")
                        else:
                            # Member variable
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

            def _ts_nl_description(name, node, bases_str):
                if node.type not in ("class_specifier", "struct_specifier"):
                    return ""
                body = None
                for c in node.children:
                    if c.type == "field_declaration_list":
                        body = c
                        break
                if not body:
                    return ""

                # Extract meaningful subwords from class name for enrichment
                # Split class name into subwords (handle CamelCase and snake_case)
                import re
                subwords = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', name)
                # Also split by underscore if present
                if '_' in name:
                    subwords = [word for subword in subwords for word in subword.split('_')]
                subwords = [sw.lower() for sw in subwords if sw]

                # Build fluent description using subwords from class name
                base_name = ""
                if bases_str:
                    base_name = bases_str.replace(": ", "").replace("public ", "").strip()
                    # Take first base only
                    base_name = base_name.split(",")[0].strip().split()[-1] if base_name.split() else ""

                desc_parts = []
                
                # Add class name with its meaningful subwords for enrichment
                if subwords:
                    # Filter out very generic words that don't add much meaning
                    meaningful_subwords = [sw for sw in subwords if sw not in {'base', 'abstract', 'interface', 'impl', 'default', 'simple', 'basic'}]
                    if meaningful_subwords:
                        desc_parts.append(f"{name} ({' '.join(meaningful_subwords)})")
                    else:
                        desc_parts.append(f"{name}")
                else:
                    desc_parts.append(f"{name}")

                # Add inheritance info if available
                if base_name:
                    desc_parts.append(f"extending {base_name}")

                desc = " ".join(desc_parts) + "."
                return f"Description: {desc}"

            def _ts_template_params(n):
                for c in n.children:
                    if c.type == "template_parameter_list":
                        return c.text.decode("utf8", errors="ignore")
                return ""

            def _ts_decl_name(n):
                for c in n.children:
                    if c.type == "identifier":
                        return c.text.decode("utf8", errors="ignore")
                    if c.type in ("reference_declarator", "pointer_declarator", "declarator", "function_declarator"):
                        result = _ts_decl_name(c)
                        if result:
                            return result
                return ""

            def _ts_func_name(n):
                decl = n.child_by_field_name("declarator")
                if not decl:
                    return _ts_decl_name(n)
                return _ts_decl_name(decl)

            def walk(node, class_name=""):
                if node.type in cls._CPP_SIGNIFICANT:
                    name_node = node.child_by_field_name("name")
                    name = name_node.text.decode("utf8", errors="ignore") if name_node else ""
                    if not name:
                        if node.type == "declaration":
                            name = _ts_decl_name(node)
                        elif node.type in ("function_definition", "template_function", "template_method"):
                            name = _ts_func_name(node)
                    doc = ""
                    for c in node.children:
                        if c.type == "comment":
                            doc = c.text.decode("utf8", errors="ignore")
                            break
                    if name:
                        params = node.child_by_field_name("parameters")
                        if not params and node.type in ("function_definition", "template_function", "template_method"):
                            decl = node.child_by_field_name("declarator")
                            if decl:
                                params = decl.child_by_field_name("parameters")
                        sig_text = params.text.decode("utf8", errors="ignore") if params else ""

                        label = _LABELS["mapping"].get(node.type, node.type)
                        body_summary = _ts_body_summary(node)
                        bases_text = _ts_base_classes(node) if node.type in ("class_specifier", "struct_specifier") else ""
                        bases_list = [b.strip() for b in bases_text.replace(": ", "").split(",") if b.strip()] if bases_text else []
                        chunks.append({
                            "kind": label,
                            "name": name,
                            "file": path_hint,
                            "signature": sig_text,
                            "docstring": doc.strip() if doc else "",
                            "body": body_summary,
                            "bases": bases_list,
                            "in_class": class_name if class_name and node.type in ("function_definition", "template_function", "template_method") else "",
                        })
                next_class = class_name
                if node.type in ("class_specifier", "struct_specifier"):
                    nname = node.child_by_field_name("name")
                    if nname:
                        next_class = nname.text.decode("utf8", errors="ignore")
                for c in node.children:
                    walk(c, next_class)
            walk(root)
        except Exception as e:
            chunks.append(f"[file] {path_hint} (parse error: {e})")
        return chunks if chunks else cls._parse_fallback(source, path_hint, "Block")

    @classmethod
    def _parse_cpp_clang(cls, source: str, path_hint: str = "") -> list[str]:
        if not _CLANG_AVAILABLE:
            return cls._parse_fallback(source, path_hint, "Block")
        try:
            import clang.cindex as ci
            # Try to set the library path to common locations
            for libpath in ['/usr/lib/llvm-18/lib', '/usr/lib/llvm-15/lib', '/usr/lib/x86_64-linux-gnu']:
                try:
                    ci.Config.set_library_path(libpath)
                    break
                except:
                    pass
            index = ci.Index.create()
            # Parse the source as an unsaved file
            unsaved_file = (path_hint, source)
            tu = index.parse(path_hint, unsaved_files=[unsaved_file], args=['-x', 'c++', '-std=c++17'])
            if tu.diagnostics:
                for diag in tu.diagnostics:
                    if diag.severity >= ci.Diagnostic.Error:
                        return cls._parse_fallback(source, path_hint, "Block")
        except Exception:
            return cls._parse_fallback(source, path_hint, "Block")

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
                        except Exception:
                            signature = ""
                    # Get base classes for class/struct
                    extra = []
                    if cursor.kind in (ci.CursorKind.CLASS_DECL, ci.CursorKind.STRUCT_DECL):
                        bases = []
                        for c in cursor.get_children():
                            if c.kind == ci.CursorKind.CXX_BASE_SPECIFIER:
                                try:
                                    bases.append(c.type.spelling)
                                except Exception:
                                    pass
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
                    type_key = cursor.kind.name.lower().replace('_decl', '')
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
                    name = name_node.text.decode("utf8", errors="ignore") if name_node else ""
                    doc = ""
                    for c in node.children:
                        if c.type == "comment":
                            doc = c.text.decode("utf8", errors="ignore")
                            break
                    if name:
                        params = node.child_by_field_name("parameters")
                        sig_text = params.text.decode("utf8", errors="ignore") if params else ""
                        label = _LABELS["mapping"].get(node.type, f"[{node.type}]")
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
        except Exception:
            pass

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


class VectorStore:
    """In-memory vector store with cosine similarity search."""

    def __init__(self):
        self.vectors: list[np.ndarray] = []
        self._array_cache: np.ndarray | None = None
        self._cached_len = 0
        self.texts: list[str] = []
        self.node_ids: list[int | None] = []

    def _get_array(self) -> np.ndarray:
        if len(self.vectors) != self._cached_len or self._array_cache is None:
            self._array_cache = np.stack(self.vectors) if self.vectors else np.array([])
            self._cached_len = len(self.vectors)
        return self._array_cache

    def add(self, vec: np.ndarray, text: str, node_id: int | None = None) -> None:
        self.vectors.append(vec)
        self.texts.append(text)
        self.node_ids.append(node_id)
        self._array_cache = None

    def add_many(self, vecs: np.ndarray, texts: list[str], node_ids: list[int | None] | None = None) -> None:
        self.vectors.extend(vecs)
        self.texts.extend(texts)
        if node_ids is not None:
            self.node_ids.extend(node_ids)
        else:
            self.node_ids.extend([None] * len(texts))
        self._array_cache = None

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
        return len(self.vectors)


class StorageIO:
    """Save/load vectors, texts, node_ids and dimension to/from .npz files."""

    @staticmethod
    def save(path: str, vectors: np.ndarray | list[np.ndarray], texts: list[str], dim: int,
             node_ids: list[int | None] | None = None) -> None:
        if isinstance(vectors, list):
            vecs_array = np.stack(vectors) if vectors else np.array([])
        else:
            vecs_array = np.asarray(vectors)
        data = {
            "dim": np.array(dim),
            "vectors": vecs_array,
            "texts": np.array(texts, dtype=object),
        }
        if node_ids is not None:
            data["node_ids"] = np.array(
                [nid if nid is not None else -1 for nid in node_ids], dtype=np.int32
            )
        np.savez_compressed(path, **data)

    @staticmethod
    def load(path: str) -> tuple:
        """Load vectors, texts, node_ids and dimension from a .npz file."""
        data = np.load(path, allow_pickle=True)
        vecs = data["vectors"]
        vectors = [vecs[i] for i in range(len(vecs))]
        texts = list(data["texts"])
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
        data_dir: Directory to save the index (relative to root).
                  If not given, computed from config embedding_store + project name.
        delta: If True, build delta index (only changed files).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    embedder_cfg_path = os.path.join(script_dir, "config.json")

    # Compute data_dir from embedder config if not explicitly given
    if data_dir is None:
        if os.path.exists(embedder_cfg_path):
            with open(embedder_cfg_path) as f:
                ecfg = json.load(f)
            store_root = ecfg.get("embedding_store")
            if store_root:
                store_root = os.path.expandvars(os.path.expanduser(store_root))
                data_dir = os.path.join(store_root, os.path.basename(root))
    if not data_dir:
        data_dir = "data"

    # Load model configuration — project root overrides, else fall back to embedder config
    model_name = "all-MiniLM-L6-v2"
    device = None
    query_prefix = None
    passage_prefix = None
    batch_size = 1024
    float_type = "fp32"
    cfg_path = os.path.join(root, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        model_name = cfg.get("model_name", model_name)
        device = cfg.get("device")
        query_prefix = cfg.get("query_prefix")
        passage_prefix = cfg.get("passage_prefix")
        batch_size = cfg.get("batch_size", batch_size)
        float_type = cfg.get("float_type", float_type)
    elif os.path.exists(embedder_cfg_path):
        with open(embedder_cfg_path) as f:
            ecfg = json.load(f)
        model_name = ecfg.get("model_name", model_name)
        device = ecfg.get("device")
        query_prefix = ecfg.get("query_prefix")
        passage_prefix = ecfg.get("passage_prefix")
        batch_size = ecfg.get("batch_size", batch_size)
        float_type = ecfg.get("float_type", float_type)

    # Initialize model
    enc = EmbeddingModel(model_name, device=device,
                         query_prefix=query_prefix, passage_prefix=passage_prefix,
                         float_type=float_type)

    project = root
    if delta:
        # --- Delta mode: only changed files ---
        os.chdir(project)
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True, text=True, timeout=30
        )
        changed = [f.strip() for f in result.stdout.split('\n') if f.strip()]
        if not changed:
            print("Delta: no changed files")
            StorageIO.save(os.path.join(project, data_dir, 'delta.npz'), [], [], enc.dim)
            with open(os.path.join(project, data_dir, 'delta_texts.json'), 'w') as f:
                json.dump({"files": [], "texts": [], "model": model_name}, f)
            return

        print(f"Delta: {len(changed)} changed files")
        chunks: list[str] = []
        strategy = ASTParser._load_strategy(project)
        for fp in changed:
            abspath = os.path.join(project, fp)
            if not os.path.isfile(abspath):
                continue
            try:
                file_chunks = ASTParser.parse_file(abspath, path_hint=fp)
                if file_chunks:
                    for c in file_chunks:
                        if isinstance(c, str):
                            if c.startswith(tuple(ASTParser.SKIP_PREFIXES)):
                                continue
                            chunks.append(c)
                        else:
                            enriched = strategy.enrich(c)
                            kind = c.get("kind", "")
                            name = c.get("name", "")
                            f = c.get("file", "")
                            prefix = f"{kind} {f} {name}".strip()
                            if prefix:
                                enriched = f"{prefix} | {enriched}" if enriched else prefix
                            chunks.append(enriched)
            except Exception as e:
                chunks.append(f"[file] {fp} (read error: {e})")

        if not chunks:
            print("Delta: no parseable chunks")
            StorageIO.save(os.path.join(project, data_dir, 'delta.npz'), [], [], enc.dim)
            with open(os.path.join(project, data_dir, 'delta_texts.json'), 'w') as f:
                json.dump({"files": changed, "texts": [], "model": model_name}, f)
            return

        embed_texts = [enc.passage_prefix + c for c in chunks] if enc.passage_prefix else chunks
        vecs = enc.embed_many(embed_texts)

        out_vec = os.path.join(project, data_dir, 'delta.npz')
        os.makedirs(os.path.dirname(out_vec), exist_ok=True)
        StorageIO.save(out_vec, vecs, chunks, enc.dim)

        delta_data = {
            "files": changed,
            "texts": chunks,
            "model": model_name,
        }
        with open(os.path.join(project, data_dir, 'delta_texts.json'), 'w') as f:
            json.dump(delta_data, f, ensure_ascii=False)

        print(f"Delta index: {len(chunks)} chunks from {len(changed)} files -> {out_vec}")
    else:
        # --- Full rebuild ---
        chunks = ASTParser.scan_project(project)

        # Enrich chunks with tree context and build node_ids mapping
        node_ids = [None] * len(chunks)
        tree_index_path = os.path.join(data_dir, "tree_index.json")
        if os.path.exists(tree_index_path):
            try:
                from tree_search import TreeIndex
                ti = TreeIndex(data_dir=data_dir)
                for i, text in enumerate(chunks):
                    n = ti.match_node(text)
                    if n is None:
                        continue
                    uid = n["_uid"]
                    node_ids[i] = uid
                    # Append parent context to chunk text
                    parent = ti.get_parent(uid)
                    if parent:
                        chunks[i] += f". Parent: {parent['name']}"
                    # Append children context
                    kids = ti.get_children(uid)
                    if kids:
                        child_names = [c["name"] for c in kids[:6]]
                        chunks[i] += f". Children: {', '.join(child_names)}"
                matched = sum(1 for n in node_ids if n is not None)
                print(f"  enriched {matched}/{len(chunks)} chunks with tree context")
            except Exception as e:
                print(f"  tree enrichment skipped: {e}")

        embed_texts = [enc.passage_prefix + c for c in chunks] if enc.passage_prefix else chunks
        vecs = enc.embed_many(embed_texts)

        out = os.path.join(project, data_dir, 'enriched_vectors.npz')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        StorageIO.save(out, vecs, chunks, enc.dim, node_ids=node_ids)
        print(f"Flat index: {len(chunks)} chunks -> {out}")


def _parse_file_worker(args):
    """Worker function for parallel file parsing (CPU only, no GPU)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    fp, rel, root, tree_exts, exclude = args
    if any(x in fp for x in exclude):
        return [], [], rel

    tree_nodes = []
    flat_chunks = []
    strategy = ASTParser._load_strategy(root)

    # Flat chunks (all supported files)
    try:
        file_chunks = ASTParser.parse_file(fp, path_hint=rel)
        if file_chunks:
            for c in file_chunks:
                if isinstance(c, str):
                    if c.startswith(tuple(ASTParser.SKIP_PREFIXES)):
                        continue
                    flat_chunks.append(c)
                else:
                    enriched = strategy.enrich(c)
                    kind = c.get("kind", "")
                    name = c.get("name", "")
                    f = c.get("file", "")
                    prefix = f"{kind} {f} {name}".strip()
                    if prefix:
                        enriched = f"{prefix} | {enriched}" if enriched else prefix
                    flat_chunks.append(enriched)
    except Exception:
        pass

    # Tree nodes (code languages only)
    if fp.endswith(tree_exts):
        try:
            from tree_ast_parser import parse_file as tree_parse_file
            nodes = tree_parse_file(fp, root=root)
            tree_nodes = nodes
        except Exception:
            pass

    return tree_nodes, flat_chunks, rel


def _parse_files(root: str, num_workers: int | None = None,
                 exclude={"/venv/", "/__pycache__/", "/.", "/node_modules/", "/.git/"}) -> tuple:
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
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
        futures = {pool.submit(_parse_file_worker, f): f[0] for f in file_list}
        for i, f in enumerate(as_completed(futures)):
            try:
                tree_nodes, flat_chunks, _ = f.result()
            except Exception as e:
                continue
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
    print(f"  parsed {len(all_tree_nodes)} tree nodes, {len(chunks)} flat chunks", flush=True)
    return all_tree_nodes, tree_texts, chunks


def build_all(root: str, data_dir: str | None = None, num_workers: int | None = None,
              embed_mode: str = "multi",
              exclude={"/venv/", "/__pycache__/", "/.", "/node_modules/", "/.git/"}) -> None:
    """Build both tree and flat indices.

    Phase 1: parse source files (CPU only, no GPU).
    Phase 2: load model, embed, save indices.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    embedder_cfg_path = os.path.join(script_dir, "config.json")

    # Resolve data_dir
    if data_dir is None:
        if os.path.exists(embedder_cfg_path):
            with open(embedder_cfg_path) as f:
                ecfg = json.load(f)
            store_root = ecfg.get("embedding_store")
            if store_root:
                store_root = os.path.expandvars(os.path.expanduser(store_root))
                data_dir = os.path.join(store_root, os.path.basename(root))
    if not data_dir:
        data_dir = "data"

    # Load model config
    model_name = "all-MiniLM-L6-v2"
    device = None
    query_prefix = None
    passage_prefix = None
    batch_size = 1024
    float_type = "fp32"
    cfg_path = os.path.join(root, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        model_name = cfg.get("model_name", model_name)
        device = cfg.get("device")
        query_prefix = cfg.get("query_prefix")
        passage_prefix = cfg.get("passage_prefix")
        batch_size = cfg.get("batch_size", batch_size)
        float_type = cfg.get("float_type", float_type)
    elif os.path.exists(embedder_cfg_path):
        with open(embedder_cfg_path) as f:
            ecfg = json.load(f)
        model_name = ecfg.get("model_name", model_name)
        device = ecfg.get("device")
        query_prefix = ecfg.get("query_prefix")
        passage_prefix = ecfg.get("passage_prefix")
        batch_size = ecfg.get("batch_size", batch_size)
        float_type = ecfg.get("float_type", float_type)

    # Infer mode from device
    if embed_mode is None:
        if device and device.startswith("cuda"):
            embed_mode = "gpu"
        elif device == "cpu":
            embed_mode = "cpu"
        else:
            import torch
            embed_mode = "multi" if torch.cuda.is_available() else "cpu"

    os.makedirs(data_dir, exist_ok=True)
    tree_json_path = os.path.join(data_dir, "tree_index.json")
    tree_exists = os.path.exists(tree_json_path)
    BATCH = batch_size

    # Phase 1: parse files (no GPU) — always needed for flat chunks
    all_tree_nodes, tree_texts, chunks = _parse_files(root, num_workers, exclude)

    # Phase 2: embed and save
    print(f"Loading models (mode={embed_mode})...", flush=True)
    enc_gpu = None
    enc_cpu = None
    if embed_mode in ("multi", "gpu"):
        enc_gpu = EmbeddingModel(model_name, device=device or "cuda",
                                 query_prefix=query_prefix, passage_prefix=passage_prefix,
                                 float_type=float_type)
    if embed_mode in ("multi", "cpu"):
        enc_cpu = EmbeddingModel(model_name, device="cpu",
                                 query_prefix=query_prefix, passage_prefix=passage_prefix,
                                 float_type=float_type)

    if tree_exists and os.path.exists(os.path.join(data_dir, "tree_vectors.npz")):
        print("Tree index exists, skipping tree embedding", flush=True)
    else:
        enc_tree = enc_gpu or enc_cpu
        # Batch-embed tree texts (pre-allocated to avoid list-of-arrays)
        tree_vecs = np.array([])
        if tree_texts and enc_tree is not None:
            prefix = enc_tree.passage_prefix
            embed_texts = [prefix + t for t in tree_texts] if prefix else tree_texts
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
            tree_data = {"nodes": all_tree_nodes, "texts": tree_texts}
            with open(tree_json_path, "w", encoding="utf8") as f:
                json.dump(tree_data, f, ensure_ascii=False)
            print(f"Tree index: {len(all_tree_nodes)} nodes -> {tree_vec_path} + {tree_json_path}", flush=True)
        else:
            print("No tree nodes found")

    # Enrich flat chunks with tree context and build node_ids
    node_ids = [None] * len(chunks)
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
                child_names = [c["name"] for c in kids[:6]]
                chunks[i] += f". Children: {', '.join(child_names)}"
            if (i + 1) % max(1, len(chunks) // 40) == 0 or i == len(chunks) - 1:
                matched = sum(1 for n in node_ids[:i+1] if n is not None)
                print(f"  enrich [{i+1}/{len(chunks)}] {matched} matched", flush=True)
    except Exception as e:
        print(f"  tree enrichment skipped: {e}")

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

    prefix = (enc_gpu or enc_cpu).passage_prefix
    embed_chunks = [prefix + c for c in chunks] if prefix else chunks

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

        with ThreadPoolExecutor(max_workers=2) as pool:
            if enc_gpu is not None:
                pool.submit(_embed_worker, enc_gpu, "GPU")
            if enc_cpu is not None:
                pool.submit(_embed_worker, enc_cpu, "CPU")
        results.sort(key=lambda x: x[0])
        flat_dim = (enc_gpu or enc_cpu).dim
        vecs = np.empty((len(embed_chunks), flat_dim), dtype=np.float32)
        pos = 0
        for _, batch_vecs in results:
            vecs[pos:pos + len(batch_vecs)] = batch_vecs
            pos += len(batch_vecs)
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

    if args.build_all:
        build_all(args.root, args.data_dir, num_workers=args.workers, embed_mode=args.embed_mode)
    elif args.build_flat:
        build_flat_index(args.root, args.data_dir, args.delta)
    else:
        parser.error('Please specify --build-flat')
