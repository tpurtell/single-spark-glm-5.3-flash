#!/usr/bin/env python3
"""Build complete EXL3 reader metadata in a derived view of a pinned snapshot.

Only safetensors headers and four-byte MCG markers are read. Source weights and
the downloaded Hugging Face snapshot remain untouched.
"""
import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path

DTYPES = {'I16': 'torch.int16', 'I32': 'torch.int32', 'F16': 'torch.float16'}


def prepare(source, destination, chat_template=None):
    source = source.resolve(strict=True)
    destination = destination.absolute()
    if destination == source or destination.is_relative_to(source):
        raise ValueError('Derived view must be outside the source snapshot')
    index = json.loads((source / 'model.safetensors.index.json').read_text())
    quant_path = source / 'quantization_config.json'
    quant = json.loads(quant_path.read_text())
    if quant.get('quant_method') != 'exl3' or quant.get('codebook') != 'mcg':
        raise ValueError('This preparation supports only EXL3 MCG checkpoints')
    tensors = {}
    markers = {}
    for shard in sorted(set(index['weight_map'].values())):
        rel = Path(shard)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError(f'Unsafe shard name: {shard}')
        with (source / rel).open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            if not 0 < length < 100_000_000:
                raise ValueError(f'Invalid safetensors header: {shard}')
            header = json.loads(stream.read(length))
            header.pop('__metadata__', None)
            extent = max(t['data_offsets'][1] for t in header.values())
            if os.fstat(stream.fileno()).st_size != 8 + length + extent:
                raise ValueError(f'Incomplete shard: {shard}')
            for key, entry in header.items():
                if index['weight_map'].get(key) != shard:
                    raise ValueError(f'Index/header disagreement: {key}')
                tensors[key] = entry
                if key.endswith('.mcg'):
                    if entry['dtype'] != 'I32' or entry['shape'] != [1]:
                        raise ValueError(f'Invalid MCG marker: {key}')
                    stream.seek(8 + length + entry['data_offsets'][0])
                    markers[key] = struct.unpack('<I', stream.read(4))[0]
    if set(index['weight_map']) != set(tensors):
        raise ValueError('Index/header tensor coverage differs')
    storage = {}
    for name in sorted(k.removesuffix('.trellis') for k in tensors if k.endswith('.trellis')):
        stored = {}
        for suffix in ('mcg', 'suh', 'svh', 'trellis'):
            key = f'{name}.{suffix}'
            entry = tensors[key]
            expected_dtype = {'mcg': 'I32', 'suh': 'F16', 'svh': 'F16', 'trellis': 'I16'}[suffix]
            if entry['dtype'] != expected_dtype:
                raise ValueError(f'Unexpected dtype for {key}: {entry["dtype"]}')
            stored[key] = {
                'dtype': DTYPES[entry['dtype']], 'shape': entry['shape'],
                'n_bytes': entry['data_offsets'][1] - entry['data_offsets'][0],
            }
        elements = math.prod(tensors[name + '.suh']['shape']) * math.prod(tensors[name + '.svh']['shape'])
        bits, remainder = divmod(stored[name + '.trellis']['n_bytes'] * 8, elements)
        if remainder or bits not in (2, 3, 4, 5, 6):
            raise ValueError(f'Cannot derive integral K2–K6 layout for {name}')
        storage[name] = {
            'bits_per_weight': bits, 'mcg_multiplier': markers[name + '.mcg'],
            'quant_format': 'exl3', 'stored_tensors': stored,
        }
    original = quant.get('tensor_storage', {})
    for name, entry in original.items():
        if name not in storage or entry != storage[name]:
            raise ValueError(f'Existing manifest disagrees with tensor storage: {name}')
    quant['tensor_storage'] = storage
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name == 'quantization_config.json' or (chat_template and item.name == 'chat_template.jinja'):
            continue
        target = destination / item.name
        if not target.is_symlink() and target.exists():
            raise ValueError(f'Refusing to replace a non-derived file: {target}')
        if target.is_symlink():
            if target.readlink() == item:
                continue
            raise ValueError(f'Derived view points to another snapshot: {target}')
        target.symlink_to(item)
    target = destination / 'quantization_config.json'
    if target.is_symlink():
        raise ValueError('Derived metadata must not be a symlink')
    temp = destination / 'quantization_config.json.tmp'
    temp.write_text(json.dumps(quant, separators=(',', ':')) + '\n')
    temp.replace(target)
    if chat_template:
        chat_target = destination / 'chat_template.jinja'
        if chat_target.is_symlink():
            if chat_target.readlink() != source / 'chat_template.jinja':
                raise ValueError('Unexpected chat-template symlink in derived view')
            chat_target.unlink()
        chat_temp = destination / 'chat_template.jinja.tmp'
        chat_temp.write_bytes(chat_template.read_bytes())
        chat_temp.replace(chat_target)
    receipt = {
        'source': str(source), 'destination': str(destination),
        'original_manifest_projections': len(original),
        'derived_manifest_projections': len(storage),
        'original_manifest_sha256': hashlib.sha256(quant_path.read_bytes()).hexdigest(),
        'derived_manifest_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'weights_modified': False,
        'chat_template_sha256': hashlib.sha256((destination / 'chat_template.jinja').read_bytes()).hexdigest(),
    }
    (destination / 'preparation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--chat-template', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination, args.chat_template), indent=2))
