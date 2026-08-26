import json

import numpy as np
from rank_bm25 import BM25Okapi

from bm25 import Postings, BM25Scorer
from common import bm25_tokenize
from embedder import StorageIO
import mcp_server

TEXTS = [
    "Class svc.py Service | Methods: create_user, delete_user | Manages user records.",
    "Method svc.py create_user | (self, name, email) | In class: Service | Create a new user record.",
    "Method svc.py delete_user | (self, user_id) | In class: Service | Delete a user.",
    "Function util.py hash_table_lookup | lookup in a hash table",
    "[file] README.md This project implements user management and hashing utilities.",
]


def _okapi(texts):
    return BM25Okapi([bm25_tokenize(t) for t in texts])


def _assert_scores_match(scorer, bm25, query, n):
    tokens = bm25_tokenize(query)
    mine = scorer.get_scores(tokens)
    ref = bm25.get_scores(tokens)
    assert mine.shape == ref.shape == (n,)
    assert np.allclose(mine, ref, rtol=1e-9, atol=1e-9)


def test_scores_match_bm25okapi():
    base = Postings.build([bm25_tokenize(t) for t in TEXTS])
    scorer = BM25Scorer(base)
    ref = _okapi(TEXTS)
    for q in ("user management", "hash table", "delete user record",
              "service", "nonexistent term", "readme"):
        _assert_scores_match(scorer, ref, q, len(TEXTS))


def test_negative_idf_floor_matches():
    # "the" is in more than half the documents, so its raw idf is negative and
    # BM25Okapi floors it to 0.25 * mean idf; a naive log formulation would not.
    texts = ["the cat", "the dog", "the bird", "goldfish"]
    base = Postings.build([bm25_tokenize(t) for t in texts])
    scorer = BM25Scorer(base)
    ref = _okapi(texts)
    for q in ("the", "the cat", "goldfish"):
        _assert_scores_match(scorer, ref, q, len(texts))


def test_overlay_matches_a_single_combined_index():
    # Delta semantics: scoring base+overlay must equal one index over all texts,
    # because df, avgdl and the idf mean all shift when documents are added.
    split = 3
    base = Postings.build([bm25_tokenize(t) for t in TEXTS[:split]])
    overlay = Postings.build([bm25_tokenize(t) for t in TEXTS[split:]],
                             id_offset=split)
    scorer = BM25Scorer(base, [overlay])
    ref = _okapi(TEXTS)
    for q in ("user", "service methods", "hash", "readme project"):
        _assert_scores_match(scorer, ref, q, len(TEXTS))


def test_postings_npz_roundtrip(tmp_path):
    path = str(tmp_path / "idx.npz")
    vecs = np.ones((len(TEXTS), 4), dtype=np.float32)
    postings = Postings.build([bm25_tokenize(t) for t in TEXTS])
    StorageIO.save(path, vecs, TEXTS, 4, bm25_arrays=postings.to_arrays())

    loaded = StorageIO.load_bm25(path)
    assert loaded is not None
    assert loaded.terms == postings.terms
    assert np.array_equal(loaded.post_docs, postings.post_docs)
    assert np.array_equal(loaded.doc_len, postings.doc_len)

    scorer = BM25Scorer(loaded)
    _assert_scores_match(scorer, _okapi(TEXTS), "user records", len(TEXTS))


def test_load_bm25_returns_none_for_legacy_index(tmp_path):
    path = str(tmp_path / "legacy.npz")
    StorageIO.save(path, np.ones((1, 4), dtype=np.float32), ["x"], 4)
    assert StorageIO.load_bm25(path) is None


def _save_postings_index(path, texts, dim=4):
    enc_vecs = np.ones((len(texts), dim), dtype=np.float32)
    postings = Postings.build([bm25_tokenize(t) for t in texts])
    StorageIO.save(str(path), enc_vecs, texts, dim,
                   bm25_arrays=postings.to_arrays())


def test_app_uses_postings_without_rank_bm25(app, tmp_path, monkeypatch):
    idx = tmp_path / "enriched_vectors.npz"
    _save_postings_index(idx, TEXTS)

    # _build_bm25 does `from rank_bm25 import BM25Okapi`; if the fast path is used
    # the legacy library is never touched, and a None sentinel would explode it.
    monkeypatch.setattr("rank_bm25.BM25Okapi", None)
    app.init(str(idx))
    assert app._bm25_base is not None and app._bm25 is None

    hits = json.loads(app.search("user", top_k=2, mode="bm25", fmt="json"))
    assert hits and all(h["method"] == "bm25" for h in hits)
    assert "user" in hits[0]["text"]


def test_app_postings_delta_overlay_and_clear(app, tmp_path):
    idx = tmp_path / "enriched_vectors.npz"
    _save_postings_index(idx, TEXTS)
    app.init(str(idx))

    delta = tmp_path / "delta.npz"
    StorageIO.save(str(delta), np.ones((1, 4), dtype=np.float32),
                   ["Function new.py zebras | zebras are striped"], 4)
    assert app.load_delta(str(delta)) == "Loaded 1 delta vectors"

    hits = json.loads(app.search("zebras", top_k=1, mode="bm25", fmt="json"))
    assert "zebras" in hits[0]["text"]

    assert app.clear_delta() == "Delta cleared"
    hits = json.loads(app.search("zebras", top_k=1, mode="bm25", fmt="json"))
    assert "zebras" not in hits[0]["text"]


def test_app_postings_add_document_updates_scorer(app, tmp_path):
    idx = tmp_path / "enriched_vectors.npz"
    _save_postings_index(idx, TEXTS)
    app.init(str(idx))
    app.add_document("Function added.py quokka_handler | handles quokkas")

    # Assert at the scoring layer: search() would first expand the query via PRF,
    # which legitimately mixes in terms from other documents.
    scores = app._bm25_scores(["quokka"])
    added = len(TEXTS)  # added document is the row right after the base index
    assert scores[added] > 0
    assert np.all(scores[:added] == 0)
    assert not app._bm25_dirty


def test_app_postings_survive_save_and_reinit(app, tmp_path):
    idx = tmp_path / "enriched_vectors.npz"
    _save_postings_index(idx, TEXTS)
    app.init(str(idx))
    app.add_document("Function added.py wombat | handles wombats")

    out = tmp_path / "saved.npz"
    app.save(str(out))

    app2 = mcp_server.EmbedderApp("proj", "fake-model", data_dir=str(tmp_path))
    app2.init(str(out))
    assert app2._bm25_base is not None
    assert app2._bm25_base.n_docs == len(TEXTS) + 1  # overlay merged into the base
    scores = app2._bm25_scores(["wombat"])
    assert scores[-1] > 0 and np.all(scores[:-1] == 0)


def test_app_falls_back_to_legacy_when_postings_mismatch_store(app, tmp_path, caplog):
    idx = tmp_path / "enriched_vectors.npz"
    _save_postings_index(idx, TEXTS)
    # Corrupt: drop one row so the postings no longer describe the store rows.
    dropped = len(TEXTS) - 1
    StorageIO.save(str(idx), np.ones((dropped, 4), dtype=np.float32),
                   TEXTS[:-1], 4,
                   bm25_arrays=Postings.build([bm25_tokenize(t) for t in TEXTS]).to_arrays())
    with caplog.at_level("WARNING"):
        app.init(str(idx))
    assert "rebuilding BM25 from texts" in caplog.text
    assert app._bm25_base is None and app._bm25 is not None
