class CrossEncoderReranker:
    """Score query–passage pairs with the same local MiniLM model."""

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

        pairs = [(transcript, self.data.record_text(record_id)) for record_id in candidate_ids]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        return {record_id: float(score)
                for record_id, score in zip(candidate_ids, scores, strict=True)}

    def rerank(self, transcript: str, candidate_ids: list[str]) -> list[str]:
        scores = self.score_candidates(transcript, candidate_ids)
        # Equal scores keep the incoming candidate order.
        return sorted(candidate_ids, key=scores.__getitem__, reverse=True)
