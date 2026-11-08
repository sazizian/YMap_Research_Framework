import unittest

import numpy as np

from ymap_research.paper_search import search_candidates


def candidate(sid, ddl=None):
    return {"schema_id": sid, "paper_ddl": ddl or sid}


class PaperSearchTests(unittest.TestCase):
    def test_unique_budget_across_streams_closes_without_reading_more(self):
        closed, visited = [], []

        def stream(name, rows):
            try:
                yield from rows
                self.fail("read beyond the unique budget")
            finally:
                closed.append(name)

        def predict(rows):
            visited.extend(row["schema_id"] for row in rows)
            return np.ones((2, len(rows), 3))

        result = search_candidates([
            stream("one", [candidate("a"), candidate("c")]),
            stream("two", [candidate("b", "a"), candidate("d")]),
        ], predict, [[1, 1, 1], [2, 2, 2]], [1, 1, 1], batch_size=1, budget=2, kappa=0)
        self.assertEqual(visited, ["a", "c"])
        self.assertCountEqual(closed, ["one", "two"])
        self.assertEqual(result["raw_candidates_seen"], 3)
        self.assertEqual(result["stop_reason"], "unique_budget")
        self.assertEqual([r["stable_batches"] for r in result["trace"]], [0, 0, 0])

    def test_risk_front_reconsiders_mean_dominated_candidates(self):
        def predict(rows):
            values = {"uncertain": [1, 9], "certain": [6, 6]}
            return np.array([[[values[row["schema_id"]][member]] * 3 for row in rows] for member in range(2)])

        result = search_candidates([[candidate("uncertain"), candidate("certain")]],
                                   predict, [[1, 1, 1], [10, 10, 10]], [1, 1, 1],
                                   batch_size=1, patience=1, kappa=1)
        self.assertEqual(result["stop_reason"], "neutral_front_stable")
        self.assertEqual(result["neutral_front_indices"], [0])
        self.assertEqual(result["selected_indices"], [1])


if __name__ == "__main__":
    unittest.main()
