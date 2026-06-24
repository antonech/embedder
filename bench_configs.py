#!/usr/bin/env python3
"""Benchmark all batch_size × float_type combos."""
import subprocess, time, json, os, sys
from pathlib import Path

CONFIG = Path.home() / "project/embedder/config.json"
PROJECT = str(Path.home() / "project/magic_enum")
STORE = Path.home() / "project/embedder_store/magic_enum"
EMBEDDER = Path.home() / "project/embedder/embedder.py"
VENV = Path.home() / "project/embedder/venv/bin/python3"

combos = [
    ("fp32", 1024),
    ("fp32", 2048),
    ("fp32", 4096),
    ("fp16", 1024),
    ("fp16", 2048),
    ("fp16", 4096),
]

results = []
for ft, bs in combos:
    # Clear existing index
    for f in ["enriched_vectors.npz", "delta.npz", "delta_texts.json"]:
        (STORE / f).unlink(missing_ok=True)

    # Write config
    cfg = {
        "model_name": "intfloat/e5-small-v2",
        "batch_size": bs,
        "enrichment": ["signature", "body", "docstring"],
        "use_clang": True,
        "embedding_store": "~/project/embedder_store",
        "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    }
    if ft == "fp16":
        cfg["float_type"] = "fp16"
    with open(CONFIG, "w") as f:
        json.dump(cfg, f, indent=4)

    print(f"\n{'='*60}")
    print(f"  {ft}  batch_size={bs}")
    print(f"{'='*60}")
    sys.stdout.flush()

    start = time.time()
    ret = subprocess.run(
        [str(VENV), str(EMBEDDER), "--build-all", "--root", PROJECT],
        capture_output=True, text=True, timeout=600
    )
    elapsed = time.time() - start

    # Count chunks from output
    lines = ret.stdout.strip().split("\n")
    chunk_count = 0
    for l in lines:
        if "flat chunks ->" in l:
            chunk_count = int(l.split()[4])
        if "chunks from" in l:
            chunk_count = int(l.split()[3])

    results.append((ft, bs, elapsed, chunk_count))
    print(f"  -> {elapsed:.1f}s for {chunk_count} chunks")
    # Show any errors
    if ret.stderr.strip():
        print(f"  stderr: {ret.stderr.strip()[:200]}")

print(f"\n{'='*60}")
print(f"  RESULTS")
print(f"{'='*60}")
print(f"  {'float':>6} {'batch':>6} {'time':>8} {'chunks':>8} {'chunks/s':>10}")
for ft, bs, elapsed, n in results:
    rate = n / elapsed if elapsed > 0 else 0
    print(f"  {ft:>6} {bs:>6} {elapsed:>7.1f}s {n:>8} {rate:>9.1f}")
