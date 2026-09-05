#!/usr/bin/env python3
"""CPU-only regression for GLM compact-state convolution admission."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('conv_patch', Path(__file__).resolve().parents[1] / 'patches/port-glm-replayssm-conv-window.py')
port = importlib.util.module_from_spec(spec)
spec.loader.exec_module(port)


class ConvPatchTests(unittest.TestCase):
    def test_compact_slot_is_not_token_capacity(self):
        source = 'class Layer:\n    def choose(self, spec_state_indices_tensor):\n        if True:\n' + port.OLD + '            return conv_mql\n'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'models/glm5next/nvidia/kda.py'
            target.parent.mkdir(parents=True)
            target.write_text(source)
            port.main(root)
            patched = target.read_text()
            port.main(root)
            self.assertEqual(target.read_text(), patched)
            namespace = {}
            exec(patched, namespace)
            layer = namespace['Layer']()
            for drafts in (5, 7):
                layer.num_spec = drafts
                layer.use_replayssm_spec = True
                self.assertEqual(layer.choose(SimpleNamespace(size=lambda _axis: 1)), drafts + 1)
                layer.use_replayssm_spec = False
                self.assertEqual(layer.choose(SimpleNamespace(size=lambda _axis: 6)), 6)


if __name__ == '__main__':
    unittest.main()
