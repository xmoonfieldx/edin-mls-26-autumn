"""One-hop passage links through corpus-derived, low-frequency word phrases."""
from collections import defaultdict
from math import log

from extensions.lexical_retrieval import tokenize


# Keep connecting words inside phrases (e.g. 'return on investment'), but
# require at least two content words to suppress generic grammatical links.
STOP_WORDS = frozenset("""
a an and are as at be been being but by can could did do does for from had has
have he her his how i if in into is it its may me my not of on or our she should
so than that the their them there these they this those to was we were what when
where which who will with would you your
""".split())


def phrases(text):
    tokens = tokenize(text)
    return {
        " ".join(tokens[i:i + width])
        for width in (2, 3)
        for i in range(len(tokens) - width + 1)
        if sum(token.isalpha() and token not in STOP_WORDS
               for token in tokens[i:i + width]) >= 2
    }


class PhraseGraphIndex:
    """A passage–phrase inverted index; no qrels or query text in construction.

    Only phrases in 2..max_document_frequency distinct passages survive.
    A seed contributes log((N+1)/(df+1)) / seed_rank through each shared phrase.
    Contributions add across phrases and seeds. Expansion never recurses.
    """

    def __init__(self, records, max_document_frequency=100):
        if max_document_frequency < 2:
            raise ValueError("maximum phrase document frequency must be at least 2")
        postings = {}
        blocked = set()
        for rid in sorted(records):
            for phrase in sorted(phrases(records[rid])):
                if phrase in blocked:
                    continue
                posting = postings.setdefault(phrase, [])
                posting.append(rid)
                if len(posting) > max_document_frequency:
                    del postings[phrase]
                    blocked.add(phrase)
        self.postings = {p: ids for p, ids in postings.items() if len(ids) >= 2}
        self.record_phrases = defaultdict(list)
        for phrase in sorted(self.postings):
            for rid in self.postings[phrase]:
                self.record_phrases[rid].append(phrase)
        self.weights = {p: log((len(records) + 1) / (len(ids) + 1))
                        for p, ids in self.postings.items()}
        self.max_document_frequency = max_document_frequency

    def expand(self, seed_ids, exclude, k=20):
        scores = defaultdict(float)
        links = defaultdict(list)
        for rank, seed_id in enumerate(seed_ids, start=1):
            for phrase in self.record_phrases.get(seed_id, ()):
                contribution = self.weights[phrase] / rank
                if contribution <= 0:
                    continue
                for rid in self.postings[phrase]:
                    if rid in exclude:
                        continue
                    scores[rid] += contribution
                    links[rid].append({
                        "seed_id": seed_id, "seed_rank": rank, "phrase": phrase,
                        "document_frequency": len(self.postings[phrase]),
                        "contribution": contribution,
                    })
        ranked = sorted(scores, key=lambda rid: (-scores[rid], rid))[:k]
        return [{"record_id": rid, "score": scores[rid], "links": links[rid]}
                for rid in ranked]
