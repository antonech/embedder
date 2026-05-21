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
        args = node.get("args", [])
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
    """First N non-empty lines of body as summary."""

    MAX_LINES = 5

    @classmethod
    def key(cls) -> str:
        return "body"

    def enrich(self, node: dict) -> str:
        body = node.get("body", node.get("methods", []))
        if isinstance(body, list):
            lines = [str(b)[:100] for b in body if str(b).strip()]
            return " | ".join(lines[:self.MAX_LINES])
        return str(body)[:200]


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

    SKIP_DIRS = {'venv', '.git', '__pycache__', 'node_modules'}

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
        return CompositeStrategy.from_keys(enrichment_keys or ["kind", "name", "signature", "docstring"])

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
                try:
                    file_chunks = cls.parse_file(fp, path_hint=name)
                    if file_chunks:
                        chunks.append(f"[FILE] {name}")
                        for c in file_chunks:
                            if isinstance(c, str):
                                chunks.append(c)
                            else:
                                enriched = strategy.enrich(c)
                                f = c.get("file")
                                if f:
                                    enriched += f" ({f})"
                                chunks.append(enriched)
                except Exception as e:
                    chunks.append(f"[FILE] {name} (read error: {e})")
        return chunks

    # --- Python handler ---

    class _ParentVisitor(ast.NodeTransformer):
        def visit(self, node):
            for child in ast.iter_child_nodes(node):
                child.parent = node
            return super().visit(node)

    @classmethod
    def _parse_python(cls, source: str, path_hint: str = "") -> list[str]:
        chunks: list[str] = []
        try:
            tree = ast.parse(source)
            cls._ParentVisitor().visit(tree)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    doc = ast.get_docstring(node) or ""
                    chunk = f"{path_hint} class_definition {node.name}"
                    if doc:
                        chunk += f" | {doc}"
                    chunks.append(chunk)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if hasattr(node, 'parent') and isinstance(node.parent, ast.ClassDef):
                        continue
                    doc = ast.get_docstring(node) or ""
                    args = [a.arg for a in node.args.args]
                    chunk = f"{path_hint} function_definition {node.name}({', '.join(args)})"
                    if doc:
                        chunk += f" | {doc}"
                    chunks.append(chunk)
        except SyntaxError as e:
            chunks.append(f"[FILE] {path_hint} (parse error: {e})")
        return chunks if chunks else cls._parse_fallback(source, path_hint)

    # --- Fallback: line-based chunking ---

    @classmethod
    def _parse_fallback(cls, source: str, path_hint: str = "") -> list[str]:
        chunks = []
        for line in source.split('\n'):
            stripped = line.strip()
            if len(stripped) >= 15:
                chunks.append(f"[LINE] {stripped[:120]}")
        return chunks

    # --- C/C++ handler (tree-sitter) ---

    _CPP_SIGNIFICANT = {
        "class_specifier", "struct_specifier",
        "function_definition", "template_function",
        "template_method", "enum_specifier",
        "alias_declaration",
    }

    @classmethod
    def _parse_cpp(cls, source: str, path_hint: str = "") -> list[str]:
        # Use clang if enabled and available
        if cls._use_clang and _CLANG_AVAILABLE:
            return cls._parse_cpp_clang(source, path_hint)
        # Otherwise use tree-sitter
        if not _TS_AVAILABLE:
            return cls._parse_fallback(source, path_hint)
        try:
            from tree_sitter import Language, Parser
            import tree_sitter_cpp
            lang = Language(tree_sitter_cpp.language())
        except Exception:
            return cls._parse_fallback(source, path_hint)

        chunks: list[str] = []
        try:
            parser = Parser(lang)
            tree = parser.parse(source.encode("utf8", errors="ignore"))
            root = tree.root_node

            def walk(node):
                if node.type in cls._CPP_SIGNIFICANT:
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
                        # Build text in the same format as tree parser
                        text = f"{path_hint} {node.type} {name}"
                        if sig_text:
                            text += f" ({sig_text})"
                        if doc:
                            text += f" | {doc.strip()}"
                        chunks.append(text)
                for c in node.children:
                    walk(c)
            walk(root)
        except Exception as e:
            chunks.append(f"[FILE] {path_hint} (parse error: {e})")
        return chunks if chunks else cls._parse_fallback(source, path_hint)

    @classmethod
    def _parse_cpp_clang(cls, source: str, path_hint: str = "") -> list[str]:
        if not _CLANG_AVAILABLE:
            return cls._parse_fallback(source, path_hint)
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
                        return cls._parse_fallback(source, path_hint)
        except Exception:
            return cls._parse_fallback(source, path_hint)

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
                    # Build text in the same format as tree parser
                    text = f"{path_hint} {cursor.kind.name.lower().replace('_decl', '')} {name}"
                    if signature:
                        text += f" ({signature})"
                    if doc:
                        text += f" | {doc}"
                    chunks.append(text)
                # Recurse
                for child in cursor.get_children():
                    walk(child)
            walk(tu.cursor)
        except Exception as e:
            chunks.append(f"[FILE] {path_hint} (clang parse error: {e})")
        return chunks if chunks else cls._parse_fallback(source, path_hint)

    # --- Tree-sitter based handler (JS / Go / Rust) ---

    @classmethod
    def _parse_treesitter(cls, source: str, path_hint: str = "") -> list[str]:
        if not _TS_AVAILABLE:
            return cls._parse_fallback(source, path_hint)

        ext = os.path.splitext(path_hint)[1].lower()
        lang_config = _TS_LANGUAGES.get(ext)
        if not lang_config:
            return cls._parse_fallback(source, path_hint)

        lang_name, lang_fn = lang_config
        significant = _TS_SIGNIFICANT.get(lang_name, set())

        chunks: list[str] = []
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
                        # Build text in the same format as tree parser
                        text = f"{path_hint} {node.type} {name}"
                        if sig_text:
                            text += f" ({sig_text})"
                        if doc:
                            text += f" | {doc}"
                        chunks.append(text)
                for c in node.children:
                    walk(c)
            walk(root)
        except Exception as e:
            chunks.append(f"[FILE] {path_hint} (parse error: {e})")
        return chunks if chunks else cls._parse_fallback(source, path_hint)


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

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.model.encode(text, normalize_embeddings=True)

    def embed_many(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts."""
        return self.model.encode(texts, normalize_embeddings=True)


class VectorStore:
    """In-memory vector store with cosine similarity search."""

    def __init__(self):
        self.vectors: list[np.ndarray] = []
        self.texts: list[str] = []

    def add(self, vec: np.ndarray, text: str) -> None:
        """Add one vector-text pair."""
        self.vectors.append(vec)
        self.texts.append(text)

    def add_many(self, vecs: np.ndarray, texts: list[str]) -> None:
        """Add multiple vector-text pairs."""
        self.vectors.extend(vecs)
        self.texts.extend(texts)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[dict]:
        """Return top_k nearest items as [{text, score}]."""
        if not self.vectors:
            return []
        scores = np.dot(np.stack(self.vectors), query_vec)
        top_idxs = np.argsort(scores)[-top_k:][::-1]
        return [
            {"text": self.texts[i], "score": float(scores[i])}
            for i in top_idxs
        ]

    def __len__(self) -> int:
        return len(self.vectors)


class StorageIO:
    """Save/load vectors, texts and dimension to/from .npz files."""

    @staticmethod
    def save(path: str, vectors: list[np.ndarray], texts: list[str], dim: int) -> None:
        """Persist vectors, texts and dimension to a compressed .npz file."""
        np.savez_compressed(
            path,
            dim=np.array(dim),
            vectors=np.stack(vectors) if vectors else np.array([]),
            texts=np.array(texts, dtype=object),
        )

    @staticmethod
    def load(path: str) -> tuple[list[np.ndarray], list[str], int]:
        """Load vectors, texts and dimension from a .npz file."""
        data = np.load(path, allow_pickle=True)
        vecs = data["vectors"]
        vectors = [vecs[i] for i in range(len(vecs))]
        texts = list(data["texts"])
        dim = int(data["dim"])
        return vectors, texts, dim


def build_flat_index(root: str, data_dir: str = "data", delta: bool = False) -> None:
    """Build flat vector index for a project.

    Args:
        root: Project root directory.
        data_dir: Directory to save the index (relative to root).
        delta: If True, build delta index (only changed files).
    """
    # Load configuration
    cfg_path = os.path.join(root, "config.json")
    model_name = "all-MiniLM-L6-v2"
    device = None
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        model_name = cfg.get("model_name", model_name)
        device = cfg.get("device")

    # Initialize model and store
    enc = EmbeddingModel(model_name, device=device)
    store = VectorStore()

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
            StorageIO.save(f'{project}/data/delta.npz', [], [], enc.dim)
            with open(f'{project}/data/delta_texts.json', 'w') as f:
                json.dump({"files": [], "texts": [], "model": model_name}, f)
            return

        print(f"Delta: {len(changed)} changed files")
        chunks = []
        for fp in changed:
            abspath = os.path.join(project, fp)
            if not os.path.isfile(abspath):
                continue
            try:
                file_chunks = ASTParser.parse_file(abspath, path_hint=fp)
                if file_chunks:
                    chunks.append(f"[FILE] {fp}")
                    chunks.extend(file_chunks)
            except Exception as e:
                chunks.append(f"[FILE] {fp} (read error: {e})")

        if not chunks:
            print("Delta: no parseable chunks")
            StorageIO.save(f'{project}/data/delta.npz', [], [], enc.dim)
            with open(f'{project}/data/delta_texts.json', 'w') as f:
                json.dump({"files": changed, "texts": [], "model": model_name}, f)
            return

        vecs = enc.embed_many(chunks)
        store.add_many(vecs, chunks)

        out_vec = f'{project}/data/delta.npz'
        os.makedirs(os.path.dirname(out_vec), exist_ok=True)
        StorageIO.save(out_vec, store.vectors, store.texts, enc.dim)

        delta_data = {
            "files": changed,
            "texts": chunks,
            "model": model_name,
        }
        with open(f'{project}/data/delta_texts.json', 'w') as f:
            json.dump(delta_data, f, ensure_ascii=False)

        print(f"Delta index: {len(chunks)} chunks from {len(changed)} files -> {out_vec}")
    else:
        # --- Full rebuild ---
        chunks = ASTParser.scan_project(project)
        vecs = enc.embed_many(chunks)
        store.add_many(vecs, chunks)

        out = f'{project}/data/enriched_vectors.npz'
        os.makedirs(os.path.dirname(out), exist_ok=True)
        StorageIO.save(out, store.vectors, store.texts, enc.dim)
        print(f"Flat index: {len(chunks)} chunks -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build flat vector index for a project.')
    parser.add_argument('--build-flat', action='store_true', help='Build the flat index')
    parser.add_argument('--delta', action='store_true', help='Build delta index (only changed files)')
    parser.add_argument('--data-dir', default='data', help='Directory to save the index')
    parser.add_argument('--root', default='.', help='Project root directory')
    args = parser.parse_args()

    if args.build_flat:
        build_flat_index(args.root, args.data_dir, args.delta)
    else:
        parser.error('Please specify --build-flat')
