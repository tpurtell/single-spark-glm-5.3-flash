#!/usr/bin/env python3
"""CPU-only guards for per-pass prefix reuse and isolation qualification."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location(
    'prefix_replay', Path(__file__).with_name('test-prefix-replay-vllm.py'))
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class ReplayScoringTests(unittest.TestCase):
    def passes(self, hits=(0, 100, 90, 100)):
        return [dict(passed=True, prefix_cache_hit_delta=count) for count in hits]

    def test_all_three_replays_must_hit(self):
        self.assertTrue(replay.passes_qualify(self.passes()))
        for hits in ((0, 0, 90, 100), (0, 100, 0, 100), (0, 100, 90, 0)):
            self.assertFalse(replay.passes_qualify(self.passes(hits)))

    def test_partial_run_does_not_qualify(self):
        self.assertFalse(replay.passes_qualify(self.passes()[:2]))
        self.assertFalse(replay.passes_qualify([]))

    def test_wrong_answer_does_not_qualify(self):
        passes = self.passes()
        passes[2]['passed'] = False
        self.assertFalse(replay.passes_qualify(passes))

    def test_contaminated_cold_or_reset_counter_does_not_qualify(self):
        self.assertFalse(replay.passes_qualify(self.passes((1, 100, 90, 100))))
        self.assertFalse(replay.passes_qualify(self.passes((0, 100, -1, 100))))

    def test_main_records_individual_deltas_and_retains_failed_complete_run(self):
        for changed_hit in (90, 0):
            with self.subTest(changed_hit=changed_hit), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'result.json'
                client = SimpleNamespace(
                    FACTS=(('position', 'key', 'red'),),
                    render_exact_prompt=Mock(side_effect=[([1], 0, 0, 0), ([2], 0, 0, 0)]),
                    stream_completion=Mock(side_effect=[
                        (f'key={value}', {'prompt_tokens': 32}, 1.0, 2.0, 'stop')
                        for value in ('red', 'red', 'amethyst-3926', 'red')
                    ]),
                )
                total = 200 + changed_hit
                counters = [0, 0, 0, 0, 100, 100, 100 + changed_hit,
                            100 + changed_hit, total, total]
                argv = ['test-prefix-replay-vllm.py', '--base-url', 'http://unused',
                        '--tokens', '32', '--output', str(output)]
                with patch.object(replay.sys, 'argv', argv), \
                     patch.object(replay, 'load_long_context', return_value=client), \
                     patch.object(replay, 'prefix_hits', side_effect=counters), \
                     patch('builtins.print'):
                    if changed_hit:
                        replay.main()
                    else:
                        with self.assertRaises(SystemExit) as error:
                            replay.main()
                        self.assertEqual(error.exception.code, 1)
                report = json.loads(output.read_text())
                self.assertEqual(report['schema'], 'glm53-prefix-replay.v4')
                self.assertTrue(report['complete'])
                self.assertEqual(report['passed'], bool(changed_hit))
                self.assertEqual(report['prefix_cache_hit_delta'], total)
                self.assertEqual([item['prefix_cache_hit_delta'] for item in report['passes']],
                                 [0, 100, changed_hit, 100])
                self.assertTrue(all(item['passed'] for item in report['passes']))
                self.assertEqual(client.FACTS, (('position', 'key', 'red'),))


if __name__ == '__main__':
    unittest.main()
