"""Contract checks for the explain-to-Mermaid adapter and query catalog."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('query_diagrams', ROOT / 'scripts/query_diagrams.py')
diagrams = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagrams)


class QueryDiagrams(unittest.TestCase):
    def test_tree_edges_preserve_join_sides_and_flow_toward_result(self):
        plan = ('JOIN inner on a [stream probe; materialize build]\n'
                '  FILTER [stream]\n    SCAN frame [project a]\n'
                '  SCAN frame [project b]')
        graph = diagrams.mermaid(plan)
        self.assertIn('n1 -->|left| n0', graph)
        self.assertIn('n2 --> n1', graph)
        self.assertIn('n3 -->|right| n0', graph)
        self.assertIn('n0 --> result', graph)
        self.assertIn('build]"]:::boundary', graph)

    def test_invalid_indentation_is_rejected(self):
        with self.assertRaises(ValueError):
            diagrams.mermaid('ROOT\n    SCAN frame')

    def test_labels_cannot_break_mermaid_quotes(self):
        escaped = diagrams.quote('x "<script>&#')
        self.assertNotIn('"', escaped)
        self.assertNotIn('<', escaped)
        self.assertIn('#34;', escaped)

    def test_capture_preserves_errors_and_both_plans(self):
        captured = ('QUERY\tq1\nORIGINAL\nSORT a\n  SCAN frame\n\n'
                    'OPTIMIZED\nTOP_K 10 by a\n  SCAN frame\nEND_QUERY\n'
                    'QUERY\tq2\nERROR\tunsupported: not translated\nEND_QUERY\n')
        records = diagrams.parse_output(captured)
        self.assertEqual(set(records), {'q1', 'q2'})
        self.assertTrue(records['q1']['optimized'].startswith('TOP_K'))
        self.assertEqual(records['q2']['error'], 'unsupported: not translated')

    def test_query_sources_cover_all_translated_branches(self):
        self.assertEqual(sum(map(len, diagrams.SUITES.values())), 179)
        for suite, numbers in diagrams.SUITES.items():
            found = 0
            for number in numbers:
                source = diagrams.source_for(suite, f'q{number}')
                found += not source.startswith('No Mojo')
            expected = 36 if suite == 'tpcds' else len(numbers)
            self.assertEqual(found, expected, suite)
        self.assertIn('q48', diagrams.source_for('tpcds', 'q13'))
        self.assertIn('how="left"', diagrams.source_for('h2o_join', 'q3'))

    def test_eager_diagrams_cover_every_h2o_query(self):
        for suite in ('h2o_groupby', 'h2o_join'):
            for number in diagrams.SUITES[suite]:
                self.assertIn('n0 --> result', diagrams.mermaid(diagrams.eager_plan(suite, f'q{number}')))
        self.assertIn('len(v1)', diagrams.eager_plan('h2o_groupby', 'q10'))


if __name__ == '__main__':
    unittest.main()
