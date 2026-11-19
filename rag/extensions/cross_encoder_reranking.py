class CrossEncoderReranker:
    def __init__(self, data, device: str = "cpu", batch_size: int = 16):
        from sentence_transformers import CrossEncoder

        self.data = data
        self.batch_size = batch_size
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L6-v2",
            device=device,
        )

    def score_candidates(
        self, transcript: str, candidate_ids: list[str]
    ) -> dict[str, float]:
        if not candidate_ids:
            return {}

        pairs = []
        for record_id in candidate_ids:
            passage = self.data.record_text(record_id)
            pairs.append((transcript, passage))

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        scores_by_id = {}
        for record_id, score in zip(candidate_ids, scores, strict=True):
            scores_by_id[record_id] = float(score)
        return scores_by_id

    def rerank(self, transcript: str, candidate_ids: list[str]) -> list[str]:
        scores = self.score_candidates(transcript, candidate_ids)
        # Stable sorting preserves dense order when scores tie.
        return sorted(candidate_ids, key=scores.__getitem__, reverse=True)