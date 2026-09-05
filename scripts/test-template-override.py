#!/usr/bin/env python3
"""CPU-only startup template regression checks; run inside the image."""
import importlib.util
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('refresh_template', ROOT / 'scripts/refresh-chat-template.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TemplateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.destination = Path(self.temp.name) / 'template'
        self.bundled = ROOT / 'data/chat_template.jinja'

    def test_offline_first_boot_uses_verified_bundle(self):
        with patch.object(module, 'fetch', side_effect=AssertionError('offline fetched')):
            receipt = module.refresh(self.destination, self.bundled, offline=True)
        self.assertEqual(receipt['source'], 'bundled')
        self.assertEqual(receipt['sha256'], module.BUNDLED_SHA256)
        self.assertEqual((self.destination / 'chat_template.jinja').read_bytes(), self.bundled.read_bytes())

    def test_resolves_current_revision_before_download_and_reuses_cache(self):
        revision = 'a' * 40
        updated = self.bundled.read_bytes() + b'\n{# simulated upstream revision #}\n'
        with patch.object(module, 'fetch', side_effect=[json.dumps({'sha': revision}).encode(), updated]) as fetch:
            receipt = module.refresh(self.destination, self.bundled)
        self.assertIn(f'/resolve/{revision}/chat_template.jinja', fetch.call_args_list[1].args[0])
        self.assertEqual(receipt['source'], 'official-current')
        cached = module.refresh(self.destination, self.bundled, offline=True)
        self.assertEqual(cached['revision'], revision)
        self.assertEqual(cached['source'], 'cached')
        self.assertEqual((self.destination / 'chat_template.jinja').read_bytes(), updated)

    def test_network_failure_falls_back_without_corrupting_template(self):
        with patch.object(module, 'fetch', side_effect=urllib.error.URLError('offline')):
            receipt = module.refresh(self.destination, self.bundled)
        self.assertEqual(receipt['source'], 'bundled')
        self.assertIn('refresh_error', receipt)

    def test_rejects_invalid_remote_template_before_replacing_cache(self):
        module.refresh(self.destination, self.bundled, offline=True)
        with patch.object(module, 'fetch', side_effect=[json.dumps({'sha': 'a' * 40}).encode(), b'wrong model']):
            with self.assertRaisesRegex(ValueError, 'expected GLM'):
                module.refresh(self.destination, self.bundled)
        self.assertEqual((self.destination / 'chat_template.jinja').read_bytes(), self.bundled.read_bytes())


if __name__ == '__main__':
    unittest.main()
