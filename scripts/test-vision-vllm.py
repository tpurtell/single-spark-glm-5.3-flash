#!/usr/bin/env python3
"""CPU HTTP smoke tests: identify colored panels in one or eight inline PNGs."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import struct
import time
import urllib.error
import urllib.request
import zlib


def panels_png(left, right):
    """Generate a deterministic RGB fixture without image-library dependencies."""
    width, height = 256, 128

    def chunk(kind, data):
        return (struct.pack('>I', len(data)) + kind + data
                + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff))

    row = b'\x00' + bytes(left) * (width // 2) + bytes(right) * (width // 2)
    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(row * height)) + chunk(b'IEND', b''))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', default='vcruz305/GLM-5.3-Flash-EXL3-K2')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'schema': 'glm53-vision-smoke.v2', 'base_url': args.base_url,
              'model': args.model, 'complete': False, 'passed': False, 'cases': []}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    save()
    fixtures = [
        [('red', 'blue')], [('blue', 'red')],
        [('red', 'blue'), ('blue', 'green'), ('green', 'red'), ('yellow', 'blue'),
         ('red', 'green'), ('green', 'yellow'), ('blue', 'red'), ('yellow', 'red')],
    ]
    for pairs in fixtures:
        colors = {'red': (255, 0, 0), 'blue': (0, 0, 255),
                  'green': (0, 255, 0), 'yellow': (255, 255, 0)}
        images = [panels_png(colors[left], colors[right]) for left, right in pairs]
        expected = [{'left': left, 'right': right} for left, right in pairs]
        if len(pairs) == 1:
            expected = expected[0]
            instruction = (
                'Identify the color of the left half and the right half of this image. '
                'Reply only with a JSON object with keys "left" and "right", '
                'using lowercase English color names.')
        else:
            instruction = (
                'For each of these eight images, identify the color of its left half '
                'and right half. Reply only with a JSON array of eight objects in '
                'image order, each with keys "left" and "right" and lowercase '
                'English color names.')
        content = [{'type': 'text', 'text': instruction}]
        for png in images:
            content.append({'type': 'image_url', 'image_url': {
                'url': 'data:image/png;base64,' + base64.b64encode(png).decode()}})
        payload = {
            'model': args.model, 'temperature': 0, 'max_tokens': 1024,
            'reasoning_effort': 'low',
            'messages': [{'role': 'user', 'content': content}],
        }
        case = {'expected': expected, 'image_count': len(images),
                'image_sha256': [hashlib.sha256(png).hexdigest() for png in images],
                'passed': False}
        started = time.monotonic()
        try:
            request = urllib.request.Request(
                args.base_url.rstrip('/') + '/v1/chat/completions',
                data=json.dumps(payload).encode(),
                headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.load(response)
            case['response'] = result
            choice = result['choices'][0]
            content = (choice['message'].get('content') or '').strip()
            if content.startswith('```') and content.endswith('```'):
                content = '\n'.join(content.splitlines()[1:-1])
            case['parsed_answer'] = json.loads(content)
            case['passed'] = (case['parsed_answer'] == case['expected']
                              and choice['finish_reason'] == 'stop')
        except urllib.error.HTTPError as error:
            case['error'] = f'HTTP {error.code}: {error.read().decode()}'
        except Exception as error:
            case['error'] = f'{type(error).__name__}: {error}'
        case['seconds'] = time.monotonic() - started
        report['cases'].append(case)
        save()
        print(json.dumps(case), flush=True)
    report['complete'] = True
    report['passed'] = all(case['passed'] for case in report['cases'])
    save()
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
