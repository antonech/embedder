"""Persisted inverted-index BM25, score-equivalent to rank_bm25.BM25Okapi.

rank_bm25 builds all of its statistics in one constructor and offers no
incremental update and no persistence, so mcp_server used to re-tokenize the
entire corpus and rebuild on every startup, delta load, delta clear and
document insert (~11.7s on a 375k-chunk index, of which 9.4s is regex
tokenization). Here the inverted index is computed once by the index builders
(when the chunk texts are already in memory) and persisted next to the vectors,
and runtime mutations only extend a small in-memory overlay.

Scoring replicates BM25Okapi exactly, including its epsilon floor: idf is
log(N - df + 0.5) - log(df + 0.5), and terms whose idf is negative are floored
to 0.25 * mean(idf over the unfloored vocabulary). That floor depends on global
stats, so the overlay cannot just append postings -- N, avgdl and the mean idf
all shift when documents are added, and are recomputed here.
"""

import math
from array import array
from collections import Counter

import numpy as np

# BM25Okapi defaults; persisted indices assume these, so changing them means
# rebuilding (same as changing the tokenizer).
K1 = 1.5
B = 0.75
EPSILON = 0.25


class Postings:
    """Inverted index for one tier of the corpus.

    doc ids in post_docs are global: the persisted base covers 0..n-1 and
    overlay tiers are built with id_offset so their ids continue the sequence,
    matching the row order of the VectorStore they describe.
    """

    __slots__ = ("terms", "term_to_id", "post_docs", "post_tfs", "post_offsets",
                 "doc_len")

    def __init__(self, terms, post_docs, post_tfs, post_offsets, doc_len):
        self.terms = terms
        self.term_to_id = {t: i for i, t in enumerate(terms)}
        self.post_docs = post_docs
        self.post_tfs = post_tfs
        self.post_offsets = post_offsets
        self.doc_len = doc_len

    @property
    def n_docs(self) -> int:
        return len(self.doc_len)

    @property
    def len_sum(self) -> int:
        return int(self.doc_len.sum())

    def df(self, term_id: int) -> int:
        return int(self.post_offsets[term_id + 1] - self.post_offsets[term_id])

    def span(self, term: str):
        i = self.term_to_id.get(term)
        if i is None:
            return None
        lo, hi = int(self.post_offsets[i]), int(self.post_offsets[i + 1])
        return self.post_docs[lo:hi], self.post_tfs[lo:hi]

    @classmethod
    def build(cls, docs_tokens: list[list[str]], id_offset: int = 0) -> "Postings":
        """Build postings from pre-tokenized documents.

        array.array keeps construction memory at 4 bytes per posting instead of
        a Python object per (doc, tf) pair -- on the order of 10^8 postings for a
        large index, that difference is what makes building feasible at all.
        """
        per_term: dict[str, tuple[array, array]] = {}
        doc_len = np.empty(len(docs_tokens), dtype=np.int32)
        for d, toks in enumerate(docs_tokens):
            doc_len[d] = len(toks)
            for term, tf in Counter(toks).items():
                pair = per_term.get(term)
                if pair is None:
                    per_term[term] = (array("i", [id_offset + d]), array("i", [tf]))
                else:
                    pair[0].append(id_offset + d)
                    pair[1].append(tf)

        terms = sorted(per_term)
        post_offsets = np.empty(len(terms) + 1, dtype=np.int64)
        total = 0
        for i, t in enumerate(terms):
            post_offsets[i] = total
            total += len(per_term[t][0])
        post_offsets[len(terms)] = total

        post_docs = np.empty(total, dtype=np.int32)
        post_tfs = np.empty(total, dtype=np.int32)
        for i, t in enumerate(terms):
            lo = int(post_offsets[i])
            docs, tfs = per_term[t]
            post_docs[lo:lo + len(docs)] = np.frombuffer(docs, dtype=np.int32)
            post_tfs[lo:lo + len(tfs)] = np.frombuffer(tfs, dtype=np.int32)
        return cls(terms, post_docs, post_tfs, post_offsets, doc_len)

    def shifted(self, id_offset: int) -> "Postings":
        """Copy with global doc ids moved by id_offset (for loading delta tiers)."""
        return Postings(self.terms, self.post_docs + id_offset, self.post_tfs,
                        self.post_offsets, self.doc_len)

    def to_arrays(self) -> dict:
        blob = bytearray()
        offsets = np.empty(len(self.terms) + 1, dtype=np.int64)
        offsets[0] = 0
        for i, t in enumerate(self.terms):
            blob.extend(t.encode("utf-8"))
            offsets[i + 1] = len(blob)
        return {
            "bm25_terms_blob": np.frombuffer(bytes(blob), dtype=np.uint8),
            "bm25_terms_offsets": offsets,
            "bm25_post_docs": self.post_docs,
            "bm25_post_tfs": self.post_tfs,
            "bm25_post_offsets": self.post_offsets,
            "bm25_doc_len": self.doc_len,
        }

    @classmethod
    def from_arrays(cls, data) -> "Postings":
        blob = data["bm25_terms_blob"].tobytes()
        offsets = data["bm25_terms_offsets"]
        terms = [blob[offsets[i]:offsets[i + 1]].decode("utf-8")
                 for i in range(len(offsets) - 1)]
        return cls(terms, data["bm25_post_docs"], data["bm25_post_tfs"],
                   data["bm25_post_offsets"], data["bm25_doc_len"])

    @classmethod
    def merge(cls, tiers: list["Postings"]) -> "Postings":
        """Merge tiers into one index without retokenizing (used by save_store)."""
        per_term: dict[str, list] = {}
        for tier in tiers:
            for i, t in enumerate(tier.terms):
                lo, hi = int(tier.post_offsets[i]), int(tier.post_offsets[i + 1])
                per_term.setdefault(t, []).append(
                    (tier.post_docs[lo:hi], tier.post_tfs[lo:hi]))
        terms = sorted(per_term)
        post_offsets = np.empty(len(terms) + 1, dtype=np.int64)
        total = 0
        for i, t in enumerate(terms):
            post_offsets[i] = total
            total += sum(len(docs) for docs, _ in per_term[t])
        post_offsets[len(terms)] = total
        post_docs = np.empty(total, dtype=np.int32)
        post_tfs = np.empty(total, dtype=np.int32)
        for i, t in enumerate(terms):
            lo = int(post_offsets[i])
            for docs, tfs in per_term[t]:
                post_docs[lo:lo + len(docs)] = docs
                post_tfs[lo:lo + len(tfs)] = tfs
                lo += len(docs)
        doc_len = np.concatenate([t.doc_len for t in tiers]).astype(np.int32)
        return cls(terms, post_docs, post_tfs, post_offsets, doc_len)


class BM25Scorer:
    """BM25Okapi-equivalent scoring over a base index plus overlay tiers.

    The overlay changes N, avgdl and every term's df, which changes idf and
    (through the epsilon floor) the mean idf, so combined stats are recomputed
    whenever a tier is added or removed -- a vectorized pass over the vocabulary,
    milliseconds, against 11.7s of retokenization.
    """

    def __init__(self, base: Postings, overlays: list[Postings] = ()):
        self.n_docs = base.n_docs + sum(o.n_docs for o in overlays)
        self.avgdl = ((base.len_sum + sum(o.len_sum for o in overlays))
                      / self.n_docs)
        self._tiers = [base, *overlays]
        self._doc_len = np.concatenate([t.doc_len for t in self._tiers])
        # TF normalization is per document and query-independent: precompute it.
        self._norm = K1 * (1 - B + B * self._doc_len / self.avgdl)

        # Combined df: base vocabulary first, overlay-only terms after.
        df = np.diff(base.post_offsets).astype(np.int64)
        self._extra_df: dict[str, int] = {}
        for tier in overlays:
            for i, t in enumerate(tier.terms):
                tdf = tier.df(i)
                bi = base.term_to_id.get(t)
                if bi is None:
                    self._extra_df[t] = self._extra_df.get(t, 0) + tdf
                else:
                    df[bi] += tdf
        raw = np.log(self.n_docs - df + 0.5) - np.log(df + 0.5)
        extra_raw = np.array(
            [math.log(self.n_docs - d + 0.5) - math.log(d + 0.5)
             for d in self._extra_df.values()])
        total = len(raw) + len(extra_raw)
        mean = (raw.sum() + extra_raw.sum()) / total
        self._eps = EPSILON * float(mean)
        self._idf = np.where(raw < 0, self._eps, raw)
        self._idf_extra = {t: (self._eps if r < 0 else float(r))
                           for t, r in zip(self._extra_df, extra_raw)}

    def idf(self, term: str) -> float:
        i = self._tiers[0].term_to_id.get(term)
        if i is not None:
            return float(self._idf[i])
        return self._idf_extra.get(term, 0.0)

    def get_scores(self, tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs)
        for t in tokens:
            idf = self.idf(t)
            if not idf:  # unknown term (idf 0.0) contributes nothing
                continue
            for tier in self._tiers:
                span = tier.span(t)
                if span is None:
                    continue
                docs, tfs = span
                np.add.at(scores, docs,
                          idf * tfs * (K1 + 1) / (tfs + self._norm[docs]))
        return scores
