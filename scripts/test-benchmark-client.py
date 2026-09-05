#!/usr/bin/env python3
"""CPU-only serving-client guards; no network, Docker, or GPU calls."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('benchmark', Path(__file__).with_name('benchmark-dflash2-vllm.py'))
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class ClientTests(unittest.TestCase):
    def stream(self, concurrency, temperature):
        owner = self

        class Response:
            status = 200

            def __init__(self):
                events = [
                    {'choices': [{'index': i, 'text': 'OK', 'token_ids': [7, 8],
                                  'finish_reason': 'stop'} for i in range(concurrency)]},
                    {'usage': {'completion_tokens': 2 * concurrency}},
                ]
                self.lines = iter([('data: ' + json.dumps(event) + '\n').encode()
                                   for event in events] + [b'data: [DONE]\n'])

            def readline(self):
                return next(self.lines, b'')

        class Connection:
            def __init__(self, *args, **kwargs):
                pass

            def request(self, method, path, body, headers):
                owner.payload = json.loads(body)

            def getresponse(self):
                return Response()

            def close(self):
                pass

        counters = {'draft_tokens': 0, 'accepted_tokens': 0, 'target_verification_passes': 0}
        with patch.object(benchmark.http.client, 'HTTPConnection', Connection), \
             patch.object(benchmark, 'metrics', return_value=counters):
            return benchmark.stream_completion('http://unused', 'model', [1, 2, 3],
                concurrency, 32, 10, force_length=False, seed=42, temperature=temperature)

    def test_greedy_c6_uses_six_prompts_with_n_one(self):
        result = self.stream(6, 0)
        self.assertEqual(self.payload['n'], 1)
        self.assertEqual(self.payload['prompt'], [[1, 2, 3]] * 6)
        self.assertEqual(result['completion_tokens'], 12)

    def test_sampled_c6_preserves_n_contract(self):
        self.stream(6, 0.2)
        self.assertEqual(self.payload['n'], 6)
        self.assertEqual(self.payload['prompt'], [1, 2, 3])

    def test_one_output_chunk_has_no_resolved_decode_rate(self):
        result = self.stream(1, 0)
        self.assertFalse(result['decode_timing_resolved'])
        self.assertIsNone(result['batch_window_decode_tokens_per_second'])
        self.assertIsNone(benchmark.summarize_runs([result])['median_decode_tokens_per_second'])


if __name__ == '__main__':
    unittest.main()
