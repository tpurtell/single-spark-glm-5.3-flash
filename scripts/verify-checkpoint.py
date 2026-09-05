#!/usr/bin/env python3
"""Validate shard headers and EXL3 metadata without importing Torch or CUDA."""
import argparse
import collections
import hashlib
import json
import struct
from pathlib import Path


def inspect(root):
    config = json.loads((root / 'config.json').read_text())
    quant = json.loads((root / 'quantization_config.json').read_text())
    index = json.loads((root / 'model.safetensors.index.json').read_text())
    weight_map = index['weight_map']
    headers = {}
    shard_bytes = 0
    errors = []
    for name in sorted(set(weight_map.values())):
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError(f'Unsafe shard path: {name}')
        path = root / rel
        with path.open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            if not 0 < size < 100_000_000:
                raise ValueError(f'Invalid safetensors header size in {path}')
            header = json.loads(stream.read(size))
        header.pop('__metadata__', None)
        actual_size = path.stat().st_size
        expected_size = 8 + size + max(t['data_offsets'][1] for t in header.values())
        if actual_size != expected_size:
            errors.append(f'{name}: size {actual_size} != header extent {expected_size}')
        shard_bytes += actual_size
        for key, entry in header.items():
            if weight_map.get(key) != name:
                errors.append(f'{key}: index does not point to {name}')
            headers[key] = entry
    if set(weight_map) != set(headers):
        errors.append('Index/header tensor names differ')
    projections = sorted(k.removesuffix('.trellis') for k in headers if k.endswith('.trellis'))
    stored = quant.get('tensor_storage', {})
    missing = sorted(set(projections) - set(stored))
    extra = sorted(set(stored) - set(projections))
    for name in projections:
        for suffix in ('suh', 'svh', 'mcg', 'trellis'):
            if name + '.' + suffix not in headers:
                errors.append(f'{name}: missing {suffix}')
    return {
        'snapshot': str(root),
        'architecture': config['architectures'],
        'shards': len(set(weight_map.values())),
        'shard_bytes': shard_bytes,
        'tensors': len(headers),
        'routed_projections': len(projections),
        'manifest_projections': len(stored),
        'manifest_missing_count': len(missing),
        'manifest_missing_examples': missing[:5],
        'manifest_extra_count': len(extra),
        'manifest_bits': dict(collections.Counter(str(t.get('bits_per_weight')) for t in stored.values())),
        'config_sha256': hashlib.sha256((root / 'config.json').read_bytes()).hexdigest(),
        'quantization_config_sha256': hashlib.sha256((root / 'quantization_config.json').read_bytes()).hexdigest(),
        'errors': errors,
        'shards_complete': not errors,
        'manifest_complete': not missing and not extra,
        'verification_scope': 'Header extents, index coverage, projection families; not full tensor-content hashes',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    args = parser.parse_args()
    result = inspect(args.snapshot)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['shards_complete'] and result['manifest_complete'] else 1)
