#!/usr/bin/env python3
"""Capture one warmed remote decode; never starts Docker or a local GPU job."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--base-url', required=True)
parser.add_argument('--model', default='vcruz305/GLM-5.3-Flash-EXL3-K2')
parser.add_argument('--concurrency', type=int, default=1)
parser.add_argument('--tokens', type=int, default=128)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
spec = importlib.util.spec_from_file_location('glm53_profile_benchmark',
                                            Path(__file__).with_name('benchmark-dflash2-vllm.py'))
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)
close = benchmark.post_json(args.base_url, '/tokenize',
                            {'model': args.model, 'prompt': '</think>', 'add_special_tokens': False}, 30)['tokens']
prompt = benchmark.render_prompt(args.base_url, args.model, benchmark.CODE_AGENT_PROMPT, close, 30)


def control(path):
    with urllib.request.urlopen(urllib.request.Request(args.base_url.rstrip('/') + path,
                                                       data=b'', method='POST'), timeout=300) as response:
        response.read()


control('/start_profile')
try:
    result = benchmark.stream_completion(args.base_url, args.model, prompt, args.concurrency,
                                         args.tokens, 900, force_length=True,
                                         seed=20260829, temperature=0.2)
finally:
    control('/stop_profile')
result['scope'] = 'Diagnostic run with Torch profiling overhead; not throughput qualification'
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + '\n')
print('Remote profile captured; kernel traces are on the serving Spark')
