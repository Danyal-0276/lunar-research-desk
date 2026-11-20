"""Offline regression checks for citation validation. Run python -m unittest."""

import unittest

from generation import ABSTAIN, Claim, Response, render


class CitationTests(unittest.TestCase):
    rows = [
        {"source": "moon", "page": 2, "body": "Twelve astronauts walked on the Moon."}
    ]

    def result(self, passage, quote):
        claim = Claim(answer="Twelve astronauts.", passage=passage, quote=quote)
        return render(Response(claims=[claim]), self.rows)

    def test_page_link_comes_from_metadata(self):
        self.assertIn("pdf#page=2", self.result(1, self.rows[0]["body"]))

    def test_invented_source(self):
        self.assertEqual(self.result(2, self.rows[0]["body"]), ABSTAIN)

    def test_invented_quote(self):
        self.assertEqual(
            self.result(1, "Thirty astronauts walked on the Moon."), ABSTAIN
        )

    def test_empty_answer(self):
        self.assertEqual(render(Response(claims=[]), self.rows), ABSTAIN)
