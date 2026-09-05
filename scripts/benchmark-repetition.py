#!/usr/bin/env python3
"""Cache-busted low-entropy diagnostic; never treat its speed as normal prose."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', default='vcruz305/GLM-5.3-Flash-EXL3-K2')
    parser.add_argument('--kv-cache', choices=['nvfp4_ds_mla', 'fp8_ds_mla'], required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    source = Path(__file__).with_name('benchmark-dflash2-vllm.py')
    spec = importlib.util.spec_from_file_location('glm53_repetition_client', source)
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    spec.loader.exec_module(bench)
    close_think = bench.post_json(args.base_url, '/tokenize', {
        'model': args.model, 'prompt': '</think>', 'add_special_tokens': False,
    }, 300)['tokens']
    report = {
        'schema': 'glm53-repeated-word-diagnostic.v1', 'complete': False,
        'model': args.model, 'kv_cache': args.kv_cache, 'base_url': args.base_url,
        'scope': 'Low-entropy C1 decode diagnostic, not normal-content performance',
        'speculative_method': 'dflash2', 'draft_tokens': 5,
        'requested_word': 'orchid', 'requested_count': 100, 'max_tokens': 1500,
        'temperature': 0, 'seed': 20260905, 'warmups': 1,
        'repeats': args.repeats, 'runs': [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index in range(args.repeats + 1):
        nonce = uuid.uuid4().hex
        prompt = (f'orchid {nonce}\nRepeat only the single word "orchid" exactly '
                  '100 times, separated by spaces. Do not repeat the nonce or add any other text.')
        tokens = bench.render_prompt(args.base_url, args.model, prompt, close_think, 300)
        run = bench.stream_completion(args.base_url, args.model, tokens, 1, 1500,
                                      600, force_length=False, seed=20260905,
                                      temperature=0)
        content = run['content'][0]
        words = content.split()
        run.update(timed=index > 0, nonce=nonce, prompt=prompt,
                   observed_orchid_count=len(re.findall(r'\borchid\b', content, re.I)),
                   exact_contract_passed=words == ['orchid'] * 100
                                         and run['finish_reasons'] == ['stop'])
        report['runs'].append(run)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps({key: run[key] for key in (
            'timed', 'observed_orchid_count', 'exact_contract_passed',
            'batch_window_decode_tokens_per_second', 'finish_reasons')}), flush=True)
    timed = [run for run in report['runs'] if run['timed']]
    rates = [run['batch_window_decode_tokens_per_second'] for run in timed
             if run['decode_timing_resolved']]
    report.update(complete=True, summary={
        'median_decode_tokens_per_second': statistics.median(rates) if rates else None,
        'resolved_runs': len(rates), 'total_runs': len(timed),
        'exact_contract_passes': sum(run['exact_contract_passed'] for run in timed),
        'observed_counts': [run['observed_orchid_count'] for run in timed],
        'finish_reasons': [run['finish_reasons'][0] for run in timed],
    })
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
