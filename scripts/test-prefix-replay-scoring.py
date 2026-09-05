#!/usr/bin/env python3
"""CPU-only guards for per-pass prefix reuse and isolation qualification."""
import importlib.util
from pathlib import Path
import unittest


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


if __name__ == '__main__':
    unittest.main()
