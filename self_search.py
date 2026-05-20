import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from embedder import EmbeddingModel, VectorStore, StorageIO


def embed_file(path: str, chunk_size: int = 1):
    model = EmbeddingModel()
    store = VectorStore()

    with open(path) as f:
        lines = f.readlines()

    if chunk_size > 1:
        chunks = ["".join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]
    else:
        chunks = [l.rstrip("\n") for l in lines]

    vecs = model.embed_many(chunks)
    store.add_many(vecs, chunks)
    return model, store


if __name__ == "__main__":
    model, store = embed_file("embedder.py")

    while True:
        q = input("\n> ").strip()
        if not q or q in ("exit", "quit"):
            break
        results = store.search(model.embed(q), top_k=3)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['text']}")
