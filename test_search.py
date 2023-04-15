import unittest
from unittest.mock import MagicMock

from search import Matches, Search


class TestSearch(unittest.TestCase):
    def setUp(self):
        self.query = "test query"
        self.moderation = MagicMock()
        self.embedded = MagicMock()
        self.pinecone = MagicMock()

    def test_search_safe(self):
        is_safe = True
        moderation_classification = "safe"
        self.moderation.is_safe.return_value = (is_safe, moderation_classification)

        embeds = [1, 2, 3]
        self.embedded.create.return_value = embeds

        matches_data = [
            {"metadata": {"text": "match 1", "prediction": "positive"}},
            {"metadata": {"text": "match 2", "prediction": "neutral"}},
            {"metadata": {"text": "match 3", "prediction": "negative"}},
        ]
        self.pinecone.query.return_value = matches_data

        expected_matches = [
            Matches("match 1", "positive"),
            Matches("match 2", "neutral"),
            Matches("match 3", "negative"),
        ]

        s = Search()
        actual_is_safe, actual_classification, actual_matches = s.search(
            self.query, self.moderation, self.embedded, self.pinecone
        )

        self.assertEqual(actual_is_safe, is_safe)
        self.assertEqual(actual_classification, moderation_classification)
        self.assertEqual(actual_matches, expected_matches)

    def test_search_not_safe(self):
        is_safe = False
        moderation_classification = "not_safe"
        self.moderation.is_safe.return_value = (is_safe, moderation_classification)

        embeds = [1, 2, 3]
        self.embedded.create.return_value = embeds

        matches_data = [
            {"metadata": {"text": "match 1", "prediction": "positive"}},
            {"metadata": {"text": "match 2", "prediction": "neutral"}},
            {"metadata": {"text": "match 3", "prediction": "negative"}},
        ]
        self.pinecone.query.return_value = matches_data

        expected_matches = []

        s = Search()
        actual_is_safe, actual_classification, actual_matches = s.search(
            self.query, self.moderation, self.embedded, self.pinecone
        )

        self.assertEqual(actual_is_safe, is_safe)
        self.assertEqual(actual_classification, moderation_classification)
        self.assertEqual(actual_matches, expected_matches)


if __name__ == '__main__':
    unittest.main()
