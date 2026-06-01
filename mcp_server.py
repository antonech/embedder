import os
import json
import asyncio
import argparse
import re
from mcp.server.fastmcp import FastMCP
from embedder import EmbeddingModel, StorageIO, VectorStore
from tree_search import TreeIndex


class EmbedderApp:
    def __init__(self, project_name: str, model_name: str = "all-MiniLM-L6-v2", data_dir: str = "data", device: str | None = None):
        self.project_name = project_name
        self.model_name = model_name
        # Handle CUDA fallback
        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    device = "cpu"
            except ImportError:
                device = "cpu"
        self.device = device
        self.data_dir = data_dir
        self.encoder = EmbeddingModel(model_name, device=device)
        self.store = VectorStore()
        self._bm25 = None

    def _tokenize(self, text: str) -> list[str]:
        words = re.split(r'[^a-zA-Z0-9]+', text)
        result = []
        for w in words:
            if not w:
                continue
            parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)|\d+', w)
            for p in parts:
                p = p.lower()
                result.append(p)
        return result

    def _build_bm25(self):
        if len(self.store) == 0:
            self._bm25 = None
            return
        from rank_bm25 import BM25Okapi
        corpus = [self._tokenize(t) for t in self.store.texts]
        self._bm25 = BM25Okapi(corpus)

    def init(self, data_path: str) -> str:
        vecs, texts, dim, node_ids = StorageIO.load(data_path)
        self.store = VectorStore()
        self.store.vectors = vecs
        self.store.texts = texts
        self.store.node_ids = node_ids if node_ids is not None else [None] * len(texts)
        self._delta_count = 0
        self.data_dir = os.path.dirname(data_path)
        if hasattr(self, "_tree"):
            del self._tree
        self._build_bm25()
        return f"Loaded {len(self.store)} vectors, dim={dim}"

    def load_delta(self, data_path: str) -> str:
        if not os.path.exists(data_path):
            return "No delta file"
        vecs, texts, dim = StorageIO.load(data_path)
        self.store.vectors.extend(vecs)
        self.store.texts.extend(texts)
        self._delta_count = len(vecs)
        return f"Loaded {len(vecs)} delta vectors"

    def clear_delta(self) -> str:
        if self._delta_count < 1 or not hasattr(self, '_delta_count'):
            return "No delta to clear"
        self.store.vectors = self.store.vectors[:-self._delta_count]
        self.store.texts = self.store.texts[:-self._delta_count]
        self._delta_count = 0
        return "Delta cleared"

    def _get_tree(self):
        if not hasattr(self, "_tree"):
            self._tree = TreeIndex(data_dir=self.data_dir)
        return self._tree

    def _get_tree_store(self):
        if not hasattr(self, "_tree_store"):
            tree_vec_path = os.path.join(self.data_dir, "tree_vectors.npz")
            if os.path.exists(tree_vec_path):
                tvecs, ttexts, _, _ = StorageIO.load(tree_vec_path)
                self._tree_store = VectorStore()
                self._tree_store.add_many(tvecs, ttexts)
                # Build reverse map: tree_uid → flat idx
                self._tree_to_flat = {}
                for flat_idx, nid in enumerate(self.store.node_ids):
                    if nid is not None:
                        self._tree_to_flat[nid] = flat_idx
            else:
                self._tree_store = None
        return self._tree_store

    def _fuse_with_tree(self, flat_hits: list[dict], qv, top_k: int, beta: float = 0.3) -> list[dict]:
        """Fuse flat search results with tree vector search results.

        Tree hits that map to a flat chunk (via _tree_to_flat) boost that chunk's score.
        Unmapped tree hits are added as standalone results.
        Final results are sorted by score and deduplicated by text.
        """
        tree_store = self._get_tree_store()
        if tree_store is None or len(tree_store) == 0:
            return flat_hits

        tree_hits = tree_store.search(qv, top_k=top_k * 3)

        # Normalize tree scores
        tree_scores = [h["score"] for h in tree_hits]
        t_min, t_max = min(tree_scores), max(tree_scores)

        # Build flat result lookup by flat idx
        flat_by_idx = {h["idx"]: h for h in flat_hits}

        # Build standalone tree results (not mapped to flat)
        standalone = []
        seen_texts = set(h["text"] for h in flat_hits)

        for th in tree_hits:
            ts = (th["score"] - t_min) / (t_max - t_min) if t_max > t_min else 0.5
            th_nid = th["idx"]
            flat_idx = self._tree_to_flat.get(th_nid)

            if flat_idx is not None:
                if flat_idx in flat_by_idx:
                    # Boost existing flat result
                    h = flat_by_idx[flat_idx]
                    h["score"] = beta * ts + (1 - beta) * h["score"]
                    h["_tree_boosted"] = True
                elif th["text"] not in seen_texts:
                    # Tree result maps to a flat chunk not in top-k; add it
                    flat_by_idx[flat_idx] = {
                        "text": th["text"],
                        "score": beta * ts + (1 - beta) * 0.0,
                        "idx": flat_idx,
                        "node_id": th_nid,
                        "method": "tree_boosted",
                    }
                    seen_texts.add(th["text"])
            elif th["text"] not in seen_texts:
                # Tree node not mapped to any flat chunk
                standalone.append({
                    "text": th["text"],
                    "score": beta * ts,
                    "idx": -1,
                    "node_id": th_nid,
                    "method": "tree",
                })
                seen_texts.add(th["text"])

        merged = sorted(
            list(flat_by_idx.values()) + standalone,
            key=lambda h: -h["score"],
        )
        return merged[:top_k]

    def _annotate(self, hits: list[dict]) -> list[dict]:
        tree = self._get_tree()
        for h in hits:
            nid = h.get("node_id")
            if nid is not None:
                n = tree.get_node(nid)
            else:
                n = tree.match_node(h.get("text", ""))
            if not n:
                continue
            uid = n["_uid"]
            h["context"] = {
                "children": [
                    {"name": c["name"], "type": c["type"], "file": c["file"],
                     "lines": f"{c['start_line']}-{c['end_line']}"}
                    for c in tree.get_children(uid)
                ],
                "parent": None,
                "siblings": [
                    {"name": s["name"], "type": s["type"]}
                    for s in tree.get_siblings(uid)
                ],
            }
            p = tree.get_parent(uid)
            if p:
                h["context"]["parent"] = {"name": p["name"], "type": p["type"]}
        return hits

    @staticmethod
    def _format(hits: list[dict], fmt: str = "text") -> str:
        if fmt == "json":
            return json.dumps(hits, ensure_ascii=False, default=str)
        lines = []
        for i, h in enumerate(hits, 1):
            if fmt == "markdown":
                ctx = h.get("context")
                if ctx:
                    parent = ctx.get("parent")
                    parent_str = f"{parent['name']} ({parent['type']})" if parent else "—"
                    children_str = ", ".join(c["name"] for c in ctx.get("children", [])[:5])
                    if len(ctx.get("children", [])) > 5:
                        children_str += "..."
                    siblings_str = ", ".join(s["name"] for s in ctx.get("siblings", [])[:5])
                    if len(ctx.get("siblings", [])) > 5:
                        siblings_str += "..."
                lines.append(f"### {i}. [{h['score']:.3f}] {h['text'][:200]}")
                if ctx:
                    lines.append(f"  **Parent:** {parent_str}")
                    if children_str:
                        lines.append(f"  **Children:** {children_str}")
                    if siblings_str:
                        lines.append(f"  **Siblings:** {siblings_str}")
            else:
                lines.append(f"{i}. [{h['score']:.3f}] {h['text'][:200]}")
            lines.append("")
        return "\n".join(lines).strip()

    def search(self, query: str, top_k: int = 5, fmt: str = "text") -> str:
        qv = self.encoder.embed(query)
        hits = self.store.search(qv, top_k=top_k)
        hits = self._fuse_with_tree(hits, qv, top_k=top_k)
        return self._format(self._annotate(hits), fmt)

    def _expand_query(self, query: str, qv, top_k: int = 5, max_terms: int = 5) -> list[str]:
        """Expand BM25 query via pseudo-relevance feedback from embedding hits."""
        if not self._bm25 or len(self.store) == 0:
            return self._tokenize(query)
        hits = self.store.search(qv, top_k=top_k)
        if not hits:
            return self._tokenize(query)

        original_tokens = set(self._tokenize(query))
        # Collect terms from top hits, deduplicated per document
        term_doc_count = {}
        for h in hits:
            seen = set()
            for t in self._tokenize(h["text"]):
                if t not in original_tokens and t not in seen:
                    seen.add(t)
                    term_doc_count[t] = term_doc_count.get(t, 0) + 1

        # Score by doc frequency * idf
        scored = []
        for term, df in term_doc_count.items():
            idf = self._bm25.idf.get(term, 0.0)
            scored.append((idf * df, term))

        scored.sort(reverse=True)
        expanded = list(self._tokenize(query))
        for _, term in scored[:max_terms]:
            expanded.append(term)
        return expanded

    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.7, fmt: str = "text") -> str:
        if self._bm25 is None:
            return self.search(query, top_k=top_k, fmt=fmt)
        qv = self.encoder.embed(query)
        emb_hits = self.store.search(qv, top_k=top_k * 3)
        tokenized = self._expand_query(query, qv, top_k=top_k)
        n = len(self.store)
        bm25_raw = self._bm25.get_scores(tokenized)
        bm25_top = sorted(range(n), key=lambda i: bm25_raw[i], reverse=True)[:top_k * 3]

        if alpha >= 1.0:
            hits = [dict(**h, method="embed") for h in self.store.search(qv, top_k=top_k)]
            hits = self._fuse_with_tree(hits, qv, top_k=top_k)
            return self._format(self._annotate(hits), fmt)
        if alpha <= 0.0:
            hits = [{"text": self.store.texts[i], "score": float(bm25_raw[i]), "idx": i, "method": "bm25"}
                    for i in bm25_top[:top_k]]
            hits = self._fuse_with_tree(hits, qv, top_k=top_k)
            return self._format(self._annotate(hits), fmt)

        candidates = set(h["idx"] for h in emb_hits) | set(bm25_top)

        emb_ranks = {h["idx"]: r for r, h in enumerate(emb_hits)}
        bm25_ranks = {idx: r for r, idx in enumerate(bm25_top)}
        cand_bm25 = {idx: bm25_raw[idx] for idx in candidates}
        min_b, max_b = min(cand_bm25.values()), max(cand_bm25.values())

        K = 5
        def blend(doc_id):
            rrf = 0.0
            if doc_id in emb_ranks:
                rrf += 1.0 / (K + emb_ranks[doc_id])
            if doc_id in bm25_ranks:
                rrf += 1.0 / (K + bm25_ranks[doc_id])
            b = (cand_bm25[doc_id] - min_b) / (max_b - min_b) if max_b > min_b else 0.5
            return alpha * rrf + (1 - alpha) * b

        scored = [(doc_id, blend(doc_id)) for doc_id in candidates]
        scored.sort(key=lambda x: -x[1])
        hits = [
            {"text": self.store.texts[idx], "score": round(s, 4), "idx": idx,
             "method": "hybrid"}
            for idx, s in scored[:top_k]
        ]
        hits = self._fuse_with_tree(hits, qv, top_k=top_k)
        return self._format(self._annotate(hits), fmt)

    def embed(self, text: str) -> list[float]:
        return self.encoder.embed(text).tolist()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return self.encoder.embed_many(texts).tolist()

    def add_document(self, text: str) -> str:
        vec = self.encoder.embed(text)
        self.store.add(vec, text)
        return f"Added, total vectors: {len(self.store)}"

    def add_documents(self, texts: list[str]) -> str:
        vecs = self.encoder.embed_many(texts)
        self.store.add_many(vecs, texts)
        return f"Added {len(texts)} docs, total vectors: {len(self.store)}"

    def save(self, path: str) -> str:
        dim = self.encoder.dim
        StorageIO.save(path, self.store.vectors, self.store.texts, dim)
        return f"Saved {len(self.store)} vectors to {path}"

    def info(self) -> str:
        n = len(self.store)
        sample = self.store.texts[:3] if n > 0 else []
        delta = getattr(self, '_delta_count', 0)
        return json.dumps({"vectors": n, "delta": delta, "sample_texts": sample}, ensure_ascii=False)


projects: dict[str, EmbedderApp] = {}
mcp = FastMCP("embedder")


@mcp.tool()
def search(query: str, project: str, top_k: int = 5, fmt: str = "text") -> str:
    """Search documents by semantic similarity, includes AST context (children/parent/siblings)."""
    global projects
    if not projects:
        return "Error: server not initialized"
    target_app = projects.get(project)
    if target_app is None:
        return f"Error: project '{project}' not found"
    return target_app.search(query, top_k=top_k, fmt=fmt)


@mcp.tool()
def hybrid_search(query: str, project: str, top_k: int = 5, alpha: float = 0.7, fmt: str = "text") -> str:
    """Search by hybrid BM25 + embedding (alpha=1: pure RRF, alpha=0: pure BM25), includes AST context."""
    global projects
    if not projects:
        return "Error: server not initialized"
    target_app = projects.get(project)
    if target_app is None:
        return f"Error: project '{project}' not found"
    return target_app.hybrid_search(query, top_k=top_k, alpha=alpha, fmt=fmt)


@mcp.tool()
def embed(text: str, project: str) -> str:
    """Embed a single text into a vector."""
    global projects
    if not projects:
        return "Error: server not initialized"
    target_app = projects.get(project)
    if target_app is None:
        return f"Error: project '{project}' not found"
    return json.dumps(target_app.embed(text))


@mcp.tool()
def embed_many(texts: list[str], project: str) -> str:
    """Embed multiple texts into vectors."""
    global projects
    if not projects:
        return "Error: server not initialized"
    target_app = projects.get(project)
    if target_app is None:
        return f"Error: project '{project}' not found"
    return json.dumps(target_app.embed_many(texts))


@mcp.tool()
def store_info(project: str) -> str:
    """Return store statistics and sample texts."""
    global projects
    if not projects:
        return "Error: server not initialized"
    target_app = projects.get(project)
    if target_app is None:
        return f"Error: project '{project}' not found"
    return target_app.info()


@mcp.tool()
def init_store(data_path: str, project: str) -> str:
    """Load vectors from a saved .npz file into the store."""
    global projects
    if not projects:
        return "Error: server not initialized"
    target_app = projects.get(project)
    if target_app is None:
        return f"Error: project '{project}' not found"
    return target_app.init(data_path)


@mcp.tool()
def add_document(text: str, project: str) -> str:
    """Add a single document: embed it and store."""
    global projects
    if not projects:
        return "Error: server not initialized"
    target_app = projects.get(project)
    if target_app is None:
        return f"Error: project '{project}' not found"
    return target_app.add_document(text)


@mcp.tool()
def add_documents(texts: list[str], project: str) -> str:
    """Add multiple documents at once."""
    global projects
    if not projects:
        return "Error: server not initialized"
    target_app = projects.get(project)
    if target_app is None:
        return f"Error: project '{project}' not found"
    return target_app.add_documents(texts)


@mcp.tool()
def save_store(path: str, project: str) -> str:
    """Save current store to a .npz file."""
    global projects
    if not projects:
        return "Error: server not initialized"
    target_app = projects.get(project)
    if target_app is None:
        return f"Error: project '{project}' not found"
    return target_app.save(path)


@mcp.tool()
def load_delta(data_path: str, project: str) -> str:
    """Load delta vectors on top of the existing store."""
    global projects
    if not projects:
        return "Error: server not initialized"
    target_app = projects.get(project)
    if target_app is None:
        return f"Error: project '{project}' not found"
    return target_app.load_delta(data_path)


@mcp.tool()
def clear_delta(project: str) -> str:
    """Remove delta layer from the store."""
    global projects
    if not projects:
        return "Error: server not initialized"
    target_app = projects.get(project)
    if target_app is None:
        return f"Error: project '{project}' not found"
    return target_app.clear_delta()


async def main():
    global projects

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", help="Project name (default: basename of workdir)")
    args = parser.parse_args()

    model_name = "all-MiniLM-L6-v2"
    device = None
    data_dir = "data"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(script_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        model_name = cfg.get("model_name", model_name)
        device = cfg.get("device")
        store_root = cfg.get("embedding_store", "")
        if store_root:
            store_root = os.path.expandvars(os.path.expanduser(store_root))

    project_name = args.project or os.path.basename(os.getcwd())

    if store_root:
        data_dir = os.path.join(store_root, project_name)

    if project_name not in projects:
        projects[project_name] = EmbedderApp(project_name, model_name, data_dir=data_dir, device=device)

    app = projects[project_name]

    data_path = os.path.join(data_dir, "enriched_vectors.npz")
    if os.path.exists(data_path):
        app.init(data_path)

    delta_path = os.path.join(data_dir, "delta.npz")
    if os.path.exists(delta_path):
        app.load_delta(delta_path)

    await mcp.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
