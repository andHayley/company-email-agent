"""BM25 ranking over an inverted index of document term frequencies."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import chain
from math import log1p

import numpy as np

from company_email_agent.tokenizer import tokenize


_K1 = 1.5
_B = 0.75


@dataclass(frozen=True)
class BM25Index:
    # Each term has a slice into the parallel arrays of ids and BM25 contributions.
    postings: dict[str, slice]
    doc_ids: np.ndarray
    contributions: np.ndarray

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if k <= 0:
            return []

        spans = [
            self.postings[term]
            for term in dict.fromkeys(tokenize(query))
            if term in self.postings
        ]
        if not spans:
            return []

        if len(spans) == 1:
            span = spans[0]
            doc_ids = self.doc_ids[span]
            scores = self.contributions[span]
        else:
            ids = np.concatenate([self.doc_ids[span] for span in spans])
            weights = np.concatenate([self.contributions[span] for span in spans])
            # Remap only matching ids to a compact range before summing scores.
            doc_ids, compact_ids = np.unique(ids, return_inverse=True)
            scores = np.bincount(compact_ids, weights=weights)

        top = np.lexsort((doc_ids, -scores))[:k]
        return [(int(doc_ids[i]), float(scores[i])) for i in top]


def build_index(docs: list[str]) -> BM25Index:
    frequencies: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    lengths = []
    for doc_id, doc in enumerate(docs):
        tokens = tokenize(doc)
        lengths.append(len(tokens))
        for term, frequency in Counter(tokens).items():
            frequencies[term].append((doc_id, frequency))

    n_docs = len(lengths)
    # With no tokens there are no postings to score; the divisor just stays nonzero.
    average_length = (sum(lengths) / n_docs if n_docs else 0) or 1
    length_norms = _K1 * (1 - _B + _B * np.asarray(lengths) / average_length)

    postings = {}
    document_frequencies = []
    term_weights = []
    offset = 0
    for term, matches in frequencies.items():
        df = len(matches)
        postings[term] = slice(offset, offset + df)
        document_frequencies.append(df)
        term_weights.append((_K1 + 1) * log1p((n_docs - df + 0.5) / (df + 0.5)))
        offset += df

    ids, counts = np.fromiter(
        chain.from_iterable(frequencies.values()), dtype=np.dtype((np.intp, 2)), count=offset
    ).T
    weights = np.repeat(term_weights, np.asarray(document_frequencies, dtype=np.intp))
    contributions = weights * counts / (counts + length_norms[ids])
    return BM25Index(postings, ids.copy(), contributions)
