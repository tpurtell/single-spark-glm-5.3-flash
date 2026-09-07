#!/usr/bin/env python3
"""CPU-only launcher contract tests; Docker and GPU access are fully mocked."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='glm53-launcher-test-')
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        target = base / 'model'
        target.mkdir()
        for name in ('config.json', 'processor_config.json', 'quantization_config.json'):
            (target / name).write_text('{}')
        (target / 'model.safetensors').touch()
        (target / 'model.safetensors.index.json').write_text(json.dumps({
            'weight_map': {'test.weight': 'model.safetensors'},
        }))
        draft = base / 'draft' / 'snapshots' / 'bf582e4eacc1810f76656d1811693ff6c6737d2a'
        draft.mkdir(parents=True)
        (draft / 'config.json').write_text(json.dumps({
            'architectures': ['DFlash2DraftModel'],
            'dflash_config': {'block_size': 8, 'target_layer_ids': [6, 15, 25, 34, 43],
                              'mask_token_id': 123},
        }))
        (draft / 'model.safetensors').touch()
        self.invocation = base / 'docker-arguments'
        hooks = base / 'bash-hooks'
        hooks.write_text('''uname() { printf 'aarch64\\n'; }
docker() {
  if [[ "$1" == image && "$2" == inspect ]]; then return 0; fi
  if [[ "$1" == inspect ]]; then return 1; fi
  if [[ "$1" == run ]]; then printf '%s\\0' "$@" > "$GLM53_TEST_ARGUMENTS"; return 0; fi
  return 77
}
''')
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(('GLM53_', 'VLLM_', 'MODEL_', 'DFLASH_', 'KV_CACHE_'))
                    and k not in ('GPU_MEMORY_UTILIZATION', 'LANGUAGE_MODEL_ONLY',
                                  'LIMIT_MM_PER_PROMPT')}
        self.env.update(BASH_ENV=str(hooks), GLM53_TEST_ARGUMENTS=str(self.invocation),
                        MODEL_DIR_OVERRIDE=str(target), DFLASH_REPO_DIR=str(base / 'draft'),
                        CACHE_DIR=str(base / 'cache'))

    def launch(self, **overrides):
        return subprocess.run(['bash', str(ROOT / 'start.sh')],
                              env={**self.env, **overrides}, text=True,
                              capture_output=True)

    def arguments(self):
        return self.invocation.read_bytes().decode().rstrip('\0').split('\0')

    def test_defaults(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.arguments()
        for flag, value in {'--load-format': 'instanttensor', '--block-size': '256',
                            '--max-num-seqs': '6', '--max-model-len': '262144',
                            '--max-num-batched-tokens': '2048',
                            '--gpu-memory-utilization': '0.87',
                            '--kv-cache-dtype': 'fp8_ds_mla'}.items():
            self.assertEqual(args[args.index(flag) + 1], value)
        capture_start = args.index('--cudagraph-capture-sizes') + 1
        capture_end = args.index('--gpu-memory-utilization')
        captures = list(map(int, args[capture_start:capture_end]))
        self.assertEqual(max(captures), 36)
        self.assertTrue(set(range(6, 37, 6)).issubset(captures))
        self.assertIn('VLLM_EXL3_TRELLIS_MAX_M=36', args)
        self.assertIn('VLLM_EXL3_PREFILL_CAPACITY=1024', args)
        self.assertIn('--enable-prefix-caching', args)
        self.assertIn('--use-replayssm', args)
        self.assertNotIn('--language-model-only', args)
        self.assertEqual(json.loads(args[args.index('--limit-mm-per-prompt') + 1]),
                         {'image': 8, 'video': 0})

    def test_nvfp4_profile_override(self):
        result = self.launch(KV_CACHE_PROFILE='nvfp4')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.arguments()
        self.assertEqual(args[args.index('--kv-cache-dtype') + 1], 'nvfp4_ds_mla')

    def test_nvfp4_dtype_shorthand(self):
        result = self.launch(KV_CACHE_DTYPE='nvfp4_ds_mla')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.arguments()
        self.assertEqual(args[args.index('--kv-cache-dtype') + 1], 'nvfp4_ds_mla')

    def test_invalid_cache_dtype_shorthand(self):
        result = self.launch(KV_CACHE_DTYPE='float16')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('KV_CACHE_DTYPE must be fp8_ds_mla or nvfp4_ds_mla',
                      result.stderr)

    def test_text_only_override_keeps_256k_profile(self):
        result = self.launch(LANGUAGE_MODEL_ONLY='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.arguments()
        self.assertIn('--language-model-only', args)
        self.assertEqual(args[args.index('--max-model-len') + 1], '262144')

    def test_historical_1m_profile_is_explicit(self):
        result = self.launch(LANGUAGE_MODEL_ONLY='1', MAX_MODEL_LEN='1048576',
                             MAX_NUM_BATCHED_TOKENS='512',
                             VLLM_EXL3_PREFILL_CAPACITY='512',
                             GPU_MEMORY_UTILIZATION='0.87')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.arguments()
        for flag, value in {'--max-model-len': '1048576',
                            '--max-num-batched-tokens': '512',
                            '--gpu-memory-utilization': '0.87'}.items():
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertIn('VLLM_EXL3_PREFILL_CAPACITY=512', args)
        self.assertIn('--language-model-only', args)

    def test_native_mtp_is_fixed_when_adaptation_disabled(self):
        for depth in (3, 5, 7):
            result = self.launch(SPECULATIVE_METHOD='mtp', MTP_TOKENS=str(depth),
                                 ADAPTIVE_MTP='0', USE_REPLAYSSM='1')
            self.assertEqual(result.returncode, 0, result.stderr)
            args = self.arguments()
            config = json.loads(args[args.index('--speculative-config') + 1])
            self.assertEqual(config, {'method': 'mtp', 'num_speculative_tokens': depth})
            self.assertIn('VLLM_ADAPTIVE_MTP=0', args)
            self.assertFalse(any('/draft-repo' in arg for arg in args))
            self.assertEqual(args[args.index('--max-cudagraph-capture-size') + 1],
                             str(6 * (depth + 1)))

    def test_cap(self):
        self.assertEqual(self.launch(GPU_MEMORY_UTILIZATION='0.87').returncode, 0)
        for value in ('0.8701', '0.88', 'nan', 'inf', '-inf', '0', '-0.1'):
            with self.subTest(value=value):
                self.assertNotEqual(self.launch(GPU_MEMORY_UTILIZATION=value).returncode, 0)

    def test_lower_utilization_override(self):
        result = self.launch(GPU_MEMORY_UTILIZATION='0.84')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.arguments()
        self.assertEqual(args[args.index('--gpu-memory-utilization') + 1], '0.84')

    def test_full_rollback_uses_comparison_defaults(self):
        self.assertEqual(self.launch().returncode, 0)
        self.assertIn('--use-replayssm', self.arguments())
        self.assertIn('GLM53_REPLAYSSM_ACTIVE=1', self.arguments())
        self.assertEqual(self.launch(USE_REPLAYSSM='0').returncode, 0)
        args = self.arguments()
        self.assertNotIn('--use-replayssm', args)
        self.assertEqual(args[args.index('--max-model-len') + 1], '262144')
        self.assertIn('VLLM_EXL3_PREFILL_CAPACITY=1024', args)

    def test_alternate_profiles_use_comparison_defaults(self):
        for overrides in ({'KV_CACHE_PROFILE': 'nvfp4'}, {'DFLASH_TOKENS': '7'},
                          {'SPECULATIVE_METHOD': 'none'}):
            result = self.launch(**overrides)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = self.arguments()
            self.assertEqual(args[args.index('--max-model-len') + 1], '262144')
            self.assertEqual(args[args.index('--max-num-batched-tokens') + 1], '2048')
            self.assertEqual(args[args.index('--gpu-memory-utilization') + 1], '0.87')
            self.assertIn('VLLM_EXL3_PREFILL_CAPACITY=1024', args)
            self.assertIn('--enable-prefix-caching', args)

    def test_custom_multimodal_json(self):
        limits = '{"image":1,"video":0}'
        result = self.launch(LANGUAGE_MODEL_ONLY='0', LIMIT_MM_PER_PROMPT=limits,
                             KV_CACHE_PROFILE='fp8')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.arguments()
        self.assertEqual(json.loads(args[args.index('--limit-mm-per-prompt') + 1]),
                         json.loads(limits))
        self.assertEqual(args[args.index('--kv-cache-dtype') + 1], 'fp8_ds_mla')

    def test_invalid_multimodal_json(self):
        for limits in ('{"image":17}', '{"image":true}', '{"video":2}', '{"audio":1}'):
            with self.subTest(limits=limits):
                self.assertNotEqual(self.launch(LANGUAGE_MODEL_ONLY='0',
                                                LIMIT_MM_PER_PROMPT=limits).returncode, 0)


if __name__ == '__main__':
    unittest.main()
