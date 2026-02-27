"""
Layer 2 retrieval: scoped BM25 search.

Performs BM25 keyword retrieval over a (pre-filtered) function index.
By operating on the subset identified in Layer 1, precision is significantly
higher than searching the full project index.

No external dependencies required — implements BM25 from scratch.
"""

import math
import re
from typing import Optional


def _tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25.
    - English: lowercase words
    - Chinese: individual characters + 2-grams
    """
    tokens = []
    # English words
    tokens.extend(re.findall(r'[a-zA-Z][a-zA-Z0-9_]*', text.lower()))
    # Chinese characters
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    tokens.extend(chinese_chars)
    # Chinese bigrams
    for i in range(len(chinese_chars) - 1):
        tokens.append(chinese_chars[i] + chinese_chars[i + 1])
    return tokens


def _build_document(func_name: str, info: dict) -> str:
    """
    Combine all searchable fields of a function into a single document string.
    Fields are weighted by repetition: more important fields appear more times.
    """
    parts = [
        func_name,                              # weight 1×
        func_name,                              # boost function name
        info.get("signature", ""),
        info.get("summary", ""),
        info.get("summary", ""),                # boost summary (most important)
        info.get("scenario", ""),
        info.get("scenario", ""),               # boost scenario
        info.get("subsystem", ""),
        " ".join(info.get("data_structures", [])),
        " ".join(info.get("data_structures", [])),  # boost data structures
        info.get("file", ""),
        " ".join(info.get("known_callers", [])),
    ]
    return " ".join(filter(None, parts))


class BM25Index:
    """
    BM25 index built over the function index dictionary.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._func_names: list[str] = []
        self._tokenized_docs: list[list[str]] = []
        self._idf: dict[str, float] = {}
        self._avg_dl: float = 0.0
        self._doc_lens: list[int] = []

    def build(self, function_index: dict):
        """Build the BM25 index from a (possibly filtered) function index."""
        self._func_names = []
        self._tokenized_docs = []

        for func_name, info in function_index.items():
            doc_text = _build_document(func_name, info)
            tokens = _tokenize(doc_text)
            self._func_names.append(func_name)
            self._tokenized_docs.append(tokens)

        n = len(self._tokenized_docs)
        if n == 0:
            return

        self._doc_lens = [len(doc) for doc in self._tokenized_docs]
        self._avg_dl = sum(self._doc_lens) / n

        # Compute IDF for each term
        df: dict[str, int] = {}
        for doc in self._tokenized_docs:
            seen = set(doc)
            for term in seen:
                df[term] = df.get(term, 0) + 1

        self._idf = {}
        for term, freq in df.items():
            # BM25 IDF formula
            self._idf[term] = math.log((n - freq + 0.5) / (freq + 0.5) + 1)

    def search(self, query: str, top_k: int = 25) -> list[tuple[str, float]]:
        """
        Search the index.

        Returns:
            list of (func_name, score) sorted by descending score, length <= top_k
        """
        if not self._func_names:
            return []

        query_tokens = _tokenize(query)
        scores: list[float] = [0.0] * len(self._func_names)

        for term in query_tokens:
            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue
            for i, doc in enumerate(self._tokenized_docs):
                tf = doc.count(term)
                if tf == 0:
                    continue
                dl = self._doc_lens[i]
                norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
                )
                scores[i] += idf * norm

        # Pair up and sort
        ranked = sorted(
            ((self._func_names[i], scores[i]) for i in range(len(self._func_names)) if scores[i] > 0),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]


def bm25_search_scoped(
    query: str,
    function_index: dict,
    top_k: int = 25,
) -> list[str]:
    """
    Build a BM25 index over the given function_index subset and search it.

    Args:
        query: feature description (the user's input)
        function_index: already filtered to candidate subsystem (Layer 1 output)
        top_k: number of results to return

    Returns:
        list of function names, ranked by relevance
    """
    idx = BM25Index()
    idx.build(function_index)
    results = idx.search(query, top_k=top_k)
    return [name for name, _score in results]
