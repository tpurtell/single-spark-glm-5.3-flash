#!/usr/bin/env python3
"""CPU-only guards for honest RULER-lite sizes and output scoring."""
import importlib.util
from pathlib import Path
import random
import re
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('ruler', Path(__file__).with_name('ruler-lite.py'))
ruler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ruler)


class RulerFixtures(unittest.TestCase):
    def test_unique_vocabulary(self):
        self.assertEqual(len(ruler.CWE_WORDS), len(set(ruler.CWE_WORDS)))

    def test_exact_rendered_size_and_answer_prefix_last(self):
        render = lambda text: [0] * (13 + len(re.findall(r'\S+', text)))
        for task in (ruler.task_sniah, ruler.task_mkniah, ruler.task_vartrack, ruler.task_cwe):
            prompt, gold, suffix, _ = task(random.Random(42))
            for size in (2048, 8192, 32768):
                text, tokens = ruler.exact_visible_case('', '', prompt, suffix, size, render)
                self.assertEqual(len(tokens), size)
                self.assertTrue(text.endswith(suffix))
                for value in gold:
                    self.assertIn(value, text)

    def test_length_finish_cannot_pass_gold_inclusion(self):
        response = {'prediction': '12345', 'seconds': 1, 'response': {},
                    'finish_reason': 'length', 'usage': {'prompt_tokens': 2048}}
        task = lambda rng: ('facts', ['12345'], '', 'sniah')
        with patch.object(ruler, 'exact_visible_case', return_value=('prompt', [0] * 2048)), \
             patch.object(ruler, 'chat', return_value=response):
            case = ruler.run_case('', '', 2048, task, random.Random(42), 'glm-closed-think')
        self.assertTrue(case['semantic_match'])
        self.assertFalse(case['ok'])


if __name__ == '__main__':
    unittest.main()
