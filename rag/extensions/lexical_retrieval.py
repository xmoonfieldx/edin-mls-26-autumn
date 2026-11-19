import re
from collections import Counter, defaultdict
from math import log1p

TOKEN_PATTERN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


class BM25Retriever:
    """Use corpus-wide BM25 statistics for search or shortlist reranking."""

    def __init__(self, records: dict[str, str], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.postings = defaultdict(dict)
        self.document_lengths = {}
        self.record_ids = sorted(records)

        for record_id in self.record_ids:
            tokens = tokenize(records[record_id])
            self.document_lengths[record_id] = len(tokens)
            for term, frequency in Counter(tokens).items():
                self.postings[term][record_id] = frequency

        document_count = len(self.record_ids)
        total_length = sum(self.document_lengths.values())
        if total_length == 0:
            raise ValueError("BM25 needs a corpus with at least one token")

        average_length = total_length / document_count

        self.length_penalty = {}
        for record_id, length in self.document_lengths.items():
            self.length_penalty[record_id] = k1 * (
                1 - b + b * length / average_length
            )

        self.idf = {}
        for term, posting in self.postings.items():
            document_frequency = len(posting)
            self.idf[term] = log1p(
                (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )

    def term_score(self, term: str, record_id: str, frequency: int) -> float:
        return (
            self.idf[term]
            * frequency
            * (self.k1 + 1)
            / (frequency + self.length_penalty[record_id])
        )

    def score_candidates(
        self, transcript: str, candidate_ids: list[str]
    ) -> dict[str, float]:
        # Unmatched terms contribute zero under BM25.
        scores = dict.fromkeys(candidate_ids, 0.0)
        for term in dict.fromkeys(tokenize(transcript)):
            posting = self.postings.get(term, {})
            for record_id in candidate_ids:
                frequency = posting.get(record_id, 0)
                if frequency:
                    scores[record_id] += self.term_score(term, record_id, frequency)
        return scores

    def rerank(self, transcript: str, candidate_ids: list[str]) -> list[str]:
        scores = self.score_candidates(transcript, candidate_ids)
        return sorted(candidate_ids, key=scores.__getitem__, reverse=True)

    def retrieve(self, transcript: str, k: int = 100) -> list[str]:
        # Search all matching corpus postings, not just a dense shortlist.
        scores = defaultdict(float)
        for term in dict.fromkeys(tokenize(transcript)):
            for record_id, frequency in self.postings.get(term, {}).items():
                scores[record_id] += self.term_score(term, record_id, frequency)

        ranked = sorted(scores, key=lambda record_id: (-scores[record_id], record_id))
        return ranked[:k]


def reciprocal_rank_fusion(
    dense_ids: list[str],
    lexical_ids: list[str],
    k: int,
    constant: int = 60,
) -> list[str]:
    scores = {}
    for ranking in (dense_ids, lexical_ids):
        for rank, record_id in enumerate(ranking, start=1):
            scores[record_id] = scores.get(record_id, 0.0) + 1 / (constant + rank)

    # Dense IDs enter first, so exact fusion ties favour dense order.
    return sorted(scores, key=scores.__getitem__, reverse=True)[:k]
