import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json, glob
from embedder import EmbeddingModel, VectorStore, StorageIO


def load_config(path="config.json"):
    with open(path) as f:
        return json.load(f)


def scan_files(root=".", exclude={"/venv/", "/__pycache__/", "/.", "/node_modules/"}):
    exts = {".py", ".md", ".txt", ".json", ".sh", ".yaml", ".yml", ".toml", ".cfg", ".ini"}
    files = []
    for ext in exts:
        files.extend(glob.glob(f"{root}/**/*{ext}", recursive=True))
    return [f for f in files if not any(x in f for x in exclude)]


def build_store(cfg):
    path = cfg["embeddings_file"]
    if os.path.exists(path):
        print(f"Loading existing: {path}")
        vecs, texts, dim = StorageIO.load(path)
        store = VectorStore()
        store.vectors = vecs
        store.texts = texts
        model = EmbeddingModel(cfg["model_name"])
        return model, store

    model = EmbeddingModel(cfg["model_name"])
    store = VectorStore()
    files = scan_files()
    total = 0
    for f in files:
        with open(f, errors="ignore") as fh:
            lines = fh.readlines()
        chunks = [l.rstrip() for l in lines]
        texts = [f"{f}:{i+1} {t}" for i, t in enumerate(chunks)]
        vecs = model.embed_many(chunks)
        store.add_many(vecs, texts)
        total += len(chunks)
        print(f"  {f}: {len(chunks)}")
    StorageIO.save(path, store.vectors, store.texts, model.dim)
    print(f"Saved {total} lines to {path}")
    return model, store


if __name__ == "__main__":
    cfg = load_config()
    model, store = build_store(cfg)

    while True:
        q = input("\n> ").strip()
        if not q or q in ("exit", "quit"):
            break
        results = store.search(model.embed(q), top_k=cfg["top_k"])
        for r in results:
            print(f"  [{r['score']:.3f}] {r['text']}")
