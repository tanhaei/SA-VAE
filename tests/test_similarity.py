from __future__ import annotations

import unittest

import numpy as np

from savae.similarity import cosine_similarity_matrix, normalized_similarity_weights


class SimilarityTests(unittest.TestCase):
    def test_cosine_similarity_known_vectors(self) -> None:
        queries = np.array([[1.0, 0.0], [0.0, 1.0]])
        donors = np.array([[1.0, 0.0], [1.0, 1.0]])
        similarity = cosine_similarity_matrix(queries, donors)
        self.assertAlmostEqual(similarity[0, 0], 1.0, places=7)
        self.assertAlmostEqual(similarity[1, 0], 0.0, places=7)
        self.assertAlmostEqual(similarity[0, 1], 1 / np.sqrt(2), places=7)

    def test_weights_are_positive_and_normalized(self) -> None:
        similarities = np.array([[0.9, 0.4, -0.1], [0.2, 0.2, 0.2]])
        weights = normalized_similarity_weights(similarities, temperature=0.3)
        np.testing.assert_allclose(weights.sum(axis=1), np.ones(2), atol=1e-12)
        self.assertTrue(np.all(weights > 0))
        self.assertGreater(weights[0, 0], weights[0, 1])
        np.testing.assert_allclose(weights[1], np.full(3, 1 / 3), atol=1e-12)


if __name__ == "__main__":
    unittest.main()

