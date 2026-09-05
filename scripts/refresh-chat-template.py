#!/usr/bin/env python3
"""Refresh the vcruz K2 profile's template from Z.ai, with a bundled fallback."""
import argparse
import hashlib
import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = 'zai-org/GLM-5.3-Flash'
BUNDLED_REVISION = '690b705278a3a58e538fcb37c2ca8b5f9511213c'
BUNDLED_SHA256 = '0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5'


def fetch(url):
    with urllib.request.urlopen(url, timeout=20) as response:
        payload = response.read(1_000_001)
    if len(payload) > 1_000_000:
        raise ValueError('Unexpectedly large chat-template response')
    return payload


def refresh(destination, bundled, offline=False):
    template = bundled.read_bytes()
    if hashlib.sha256(template).hexdigest() != BUNDLED_SHA256:
        raise ValueError('Bundled official chat-template hash mismatch')
    receipt = {'repository': REPO, 'revision': BUNDLED_REVISION, 'source': 'bundled'}
    current = destination / 'chat_template.jinja'
    manifest = destination / 'chat_template.json'
    if current.is_file() and manifest.is_file():
        cached = json.loads(manifest.read_text())
        data = current.read_bytes()
        if cached.get('repository') == REPO and hashlib.sha256(data).hexdigest() == cached.get('sha256'):
            template, receipt = data, {**cached, 'source': 'cached'}
    if not offline:
        try:
            head = json.loads(fetch(f'https://huggingface.co/api/models/{REPO}'))['sha']
            if not re.fullmatch('[0-9a-f]{40}', head):
                raise ValueError('Official model API did not return an immutable revision')
            template = fetch(f'https://huggingface.co/{REPO}/resolve/{head}/chat_template.jinja')
            receipt = {'repository': REPO, 'revision': head, 'source': 'official-current'}
        except (OSError, urllib.error.URLError) as exc:
            receipt['refresh_error'] = str(exc)
    text = template.decode('utf-8')
    if not text.startswith('[gMASK]<sop>') or '<|assistant|>' not in text:
        raise ValueError('Official template is not the expected GLM chat format')
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    ImmutableSandboxedEnvironment(extensions=['jinja2.ext.loopcontrols']).parse(text)
    destination.mkdir(parents=True, exist_ok=True)
    temp = destination / 'chat_template.jinja.tmp'
    temp.write_bytes(template)
    temp.replace(current)
    receipt.update(sha256=hashlib.sha256(template).hexdigest(),
                   checked_utc=datetime.now(timezone.utc).isoformat(), offline=offline)
    manifest.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--bundled', type=Path, default=Path('/opt/glm53/data/chat_template.jinja'))
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    print(json.dumps(refresh(args.destination, args.bundled, args.offline), indent=2))
