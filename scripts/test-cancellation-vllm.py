#!/usr/bin/env python3
"""Disconnect C6 prefill/decode streams, require drained queues and exact recovery.

This is an HTTP client only. Run it from a CPU host against an idle Spark.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
from pathlib import Path
import re
import socket
import time
import urllib.request
from urllib.parse import urlparse
import uuid

from glm53_api import post, visible_prompt


def queues(base):
    with urllib.request.urlopen(base + '/metrics', timeout=30) as response:
        raw = response.read().decode()
    result = {}
    for name in ('num_requests_running', 'num_requests_waiting'):
        values = re.findall(r'^vllm:' + name + r'(?:\{[^\n]*\})? ([\d.eE+-]+)$', raw, re.M)
        if not values:
            raise RuntimeError('missing scheduler metric: ' + name)
        result[name] = sum(map(float, values))
    return result


def drain(base, timeout=120):
    deadline = time.monotonic() + timeout
    while True:
        result = queues(base)
        if not any(result.values()):
            return result
        if time.monotonic() > deadline:
            raise RuntimeError(f'cancelled requests did not drain: {result}')
        time.sleep(1)


def disconnect(base, model, tokens, mode, index, nonce):
    parsed = urlparse(base)
    client = http.client.HTTPSConnection if parsed.scheme == 'https' else http.client.HTTPConnection
    connection = client(parsed.hostname, parsed.port, timeout=600)
    response = None
    started = time.monotonic()
    seen = 0
    saw_finish = False
    try:
        connection.request('POST', '/v1/completions', json.dumps({
            'model': model, 'prompt': tokens, 'max_tokens': 2048, 'min_tokens': 2048,
            'ignore_eos': True, 'temperature': 0, 'stream': True,
            'return_token_ids': True, 'add_special_tokens': False,
            'cache_salt': f'cancellation-{nonce}-{mode}-{index}',
        }), {'Content-Type': 'application/json'})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f'HTTP {response.status}: {response.read().decode()[:400]}')
        if mode == 'prefill':
            # Headers arrive before the long prompt completes. Do not wait
            # for its first generated token before closing the transport.
            time.sleep(0.5)
        else:
            while seen < 8:
                line = response.readline().decode().strip()
                if not line:
                    if response.isclosed():
                        break
                    continue
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    saw_finish = True
                    break
                for choice in json.loads(data).get('choices', []):
                    seen += len(choice.get('token_ids') or [])
                    saw_finish |= choice.get('finish_reason') is not None
            if seen < 8 or saw_finish:
                raise RuntimeError('decode stream finished before cancellation point')
        return {'mode': mode, 'index': index, 'ok': True, 'tokens_observed': seen,
                'seconds_before_disconnect': round(time.monotonic() - started, 3)}
    finally:
        if connection.sock is not None:
            try:
                connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if response is not None:
            response.close()
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', default='vcruz305/GLM-5.3-Flash-EXL3-K2')
    parser.add_argument('--concurrency', type=int, default=6)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.base_url.rstrip('/').removesuffix('/v1')
    nonce = uuid.uuid4().hex
    report = {'schema': 'glm53-cancellation.v1', 'base_url': base, 'model': args.model,
              'concurrency': args.concurrency, 'nonce': nonce, 'phases': [], 'passed': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if any(queues(base).values()):
            raise RuntimeError('endpoint is not idle; do not mix cancellation with other tests')
        token = post(base, '/tokenize', {'model': args.model, 'prompt': ' cancellation',
                                       'add_special_tokens': False})['tokens']
        if not token:
            raise RuntimeError('empty cancellation fixture tokenization')
        for mode, size in (('prefill', 32768), ('decode', 256)):
            tokens = (token * (size // len(token) + 1))[:size]
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                results = list(pool.map(lambda i: disconnect(base, args.model, tokens,
                                                             mode, i, nonce),
                                        range(args.concurrency)))
            report['phases'].append({'mode': mode, 'prompt_tokens': len(tokens),
                                     'cancelled': results, 'queues_after': drain(base)})

            def recovery(i):
                marker = f'RECOVER-{mode.upper()}-{i}'
                prompt = visible_prompt(base, args.model, [{'role': 'user',
                    'content': f'Return exactly {marker} and nothing else.'}])
                response = post(base, '/v1/completions', {'model': args.model, 'prompt': prompt,
                    'add_special_tokens': False, 'temperature': 0, 'max_tokens': 128})
                choice = response['choices'][0]
                return {'expected': marker, 'response': response, 'ok':
                    choice.get('text', '').strip() == marker and choice.get('finish_reason') == 'stop'}

            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                report['phases'][-1]['recovery'] = list(pool.map(recovery, range(args.concurrency)))
            report['phases'][-1]['queues_after_recovery'] = drain(base)
        with urllib.request.urlopen(base + '/health', timeout=30) as response:
            report['health_after'] = response.status
        report['passed'] = report['health_after'] == 200 and all(
            item['ok'] for phase in report['phases'] for item in phase['recovery'])
    except Exception as exc:
        report['error'] = repr(exc)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
