#!/usr/bin/env python3
"""CPU-only fixture/scoring regressions; no inference or local GPU access."""
import importlib.util
import re
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('multi_needle', Path(__file__).with_name('test-multi-needle-vllm.py'))
multi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multi)


class ContextFixtureTests(unittest.TestCase):
    def setUp(self):
        self.vocabulary = {}
        self.reverse = {}

    def tokenize(self, _base, _model, text):
        ids = []
        for piece in re.findall(r'\s*\S+', text):
            if piece not in self.vocabulary:
                token = len(self.vocabulary) + 1
                self.vocabulary[piece] = token
                self.reverse[token] = piece
            ids.append(self.vocabulary[piece])
        return ids

    def detokenize(self, _base, _model, ids):
        return ''.join(self.reverse[token] for token in ids)

    def test_exact_body_sizes_and_all_facts_once(self):
        with patch.object(multi, 'tokenize', self.tokenize), patch.object(multi, 'detokenize', self.detokenize):
            for size in (8192, 10000, 32768):
                with self.subTest(size=size):
                    text, positions = multi.build_exact_prompt('unused', 'unused', size, 'test')
                    self.assertEqual(len(self.tokenize(None, None, text)), size)
                    self.assertEqual(len(positions), 6)
                    for _, key, value in multi.FACTS:
                        self.assertEqual(text.count(f'{key}={value}'), 1)

    def test_rejects_depth_without_room_after_99_percent_record(self):
        with patch.object(multi, 'tokenize', self.tokenize), patch.object(multi, 'detokenize', self.detokenize):
            with self.assertRaisesRegex(RuntimeError, 'no room'):
                multi.build_exact_prompt('unused', 'unused', 2048, 'test')

    def test_chat_overhead_is_counted(self):
        body_sizes = []

        def body(_base, _model, size, _nonce):
            body_sizes.append(size)
            return str(size), []

        def post(_base, path, payload, _timeout):
            if path == '/tokenize':
                return {'tokens': [999]}
            size = int(payload['messages'][0]['content'])
            return {'token_ids': [1] * (size + 12)}

        with patch.object(multi, 'build_exact_prompt', body), patch.object(multi, 'post', post):
            ids, _, _, size = multi.render_exact_prompt('unused', 'unused', 131072, 'test')
            self.assertEqual(len(ids), 131072)
            self.assertEqual(size, 131059)
            self.assertEqual(body_sizes, [131008, 131059])


if __name__ == '__main__':
    unittest.main()
