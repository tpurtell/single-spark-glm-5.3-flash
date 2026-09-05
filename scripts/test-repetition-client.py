#!/usr/bin/env python3
"""CPU-only checks for repetition scoring and warmup/timing exclusion."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location(
    'repetition', Path(__file__).with_name('benchmark-repetition.py'))
repetition = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repetition)


class RepetitionTests(unittest.TestCase):
    def test_warmup_contract_and_unresolved_timing(self):
        def result(words, finish, rate, resolved=True):
            return dict(content=[' '.join(words)], finish_reasons=[finish],
                        batch_window_decode_tokens_per_second=rate,
                        decode_timing_resolved=resolved)

        client = SimpleNamespace(
            post_json=Mock(return_value={'tokens': [1]}),
            render_prompt=Mock(return_value=[1, 2]),
            stream_completion=Mock(side_effect=[
                result(['orchid'], 'stop', 999),  # excluded warmup
                result(['orchid'] * 100, 'stop', 10),
                result(['orchid'] * 100, 'length', 20),
                result(['Orchid'] * 100, 'stop', None, False),
            ]),
        )
        fake_spec = SimpleNamespace(name='repetition_test_client',
                                    loader=SimpleNamespace(exec_module=Mock()))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result.json'
            argv = ['benchmark-repetition.py', '--base-url', 'http://unused',
                    '--kv-cache', 'nvfp4_ds_mla', '--repeats', '3',
                    '--output', str(output)]
            with patch.object(repetition.sys, 'argv', argv), \
                 patch.object(repetition.importlib.util, 'spec_from_file_location',
                              return_value=fake_spec), \
                 patch.object(repetition.importlib.util, 'module_from_spec',
                              return_value=client), \
                 patch.dict(repetition.sys.modules), patch('builtins.print'):
                repetition.main()
            report = json.loads(output.read_text())
        self.assertTrue(report['complete'])
        self.assertEqual(report['summary']['median_decode_tokens_per_second'], 15)
        self.assertEqual(report['summary']['resolved_runs'], 2)
        self.assertEqual(report['summary']['total_runs'], 3)
        self.assertEqual(report['summary']['exact_contract_passes'], 1)
        self.assertEqual(report['summary']['observed_counts'], [100, 100, 100])
        self.assertEqual(len({run['nonce'] for run in report['runs']}), 4)
        self.assertEqual(client.stream_completion.call_count, 4)


if __name__ == '__main__':
    unittest.main()
