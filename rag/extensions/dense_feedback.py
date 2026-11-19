import numpy as np


class DenseFeedbackRetriever:
    def __init__(self, retriever, feedback_k: int = 3, weight: float = 0.2):
        if feedback_k < 1:
            raise ValueError("feedback_k must be positive")
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be between 0 and 1")

        self.retriever = retriever
        self.feedback_k = feedback_k
        self.weight = weight

        # Each passage ID points to a row in the embedding matrix.
        self.row_by_id = {}
        for row_index, record_id in enumerate(retriever.record_ids):
            self.row_by_id[record_id] = row_index

    def retrieve(self, query_vector: np.ndarray, k: int = 5):
        # Use the first search's top hits as feedback.
        first_result = self.retriever.retrieve(query_vector, k=self.feedback_k)
        feedback_ids = first_result["record_ids"]

        feedback_rows = []
        for record_id in feedback_ids:
            row_index = self.row_by_id[record_id]
            feedback_rows.append(row_index)

        feedback_vectors = self.retriever.embeddings[feedback_rows]
        average_feedback = feedback_vectors.mean(axis=0)

        # Move the query slightly toward the average of the feedback passages.
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        query = query / np.linalg.norm(query)
        refined_query = (1.0 - self.weight) * query + self.weight * average_feedback

        # Search the full corpus again with the refined query.
        result = self.retriever.retrieve(refined_query, k=k)
        result["feedback_ids"] = feedback_ids
        return result