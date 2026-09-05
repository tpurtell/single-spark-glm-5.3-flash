#!/usr/bin/env python3
"""Extend image-private mixed-Trellis descriptors for GLM's 288 experts.

The old descriptor had eight tier-local index bits. GLM can put more than 256
gate/up/down projections in a single tier. Update every producer, consumer,
and validation boundary together to nine bits (512 local slots). Descriptor
tables are generated in process, never loaded from checkpoint files.
"""
import sys
from pathlib import Path

root = Path(sys.argv[1]) / 'b12x/moe'
path = root / '_shared/kernels/w4a16/mixed_trellis.py'
source = path.read_text()
if '# GLM53_DESCRIPTOR_ABI_9BIT' not in source:
    replacements = {
        'tier0.num_experts > 256 or tier1.num_experts > 256': 'tier0.num_experts > 512 or tier1.num_experts > 512',
        'eight bits': 'nine bits',
        'descriptor >> Int32(8)': 'descriptor >> Int32(9)',
        'descriptor & Int32(0xFF)': 'descriptor & Int32(0x1FF)',
        '_MAX_TIER_EXPERTS = 256': '_MAX_TIER_EXPERTS = 512',
        'len(tier0_ids) > 256 or len(tier1_ids) > 256': 'len(tier0_ids) > 512 or len(tier1_ids) > 512',
        '(1 << 8) | i': '(1 << 9) | i',
        '(tier << 8) |': '(tier << 9) |',
        'value < 0 or value > 256 for value in slots': 'value < 0 or value > 512 for value in slots',
        'local > 0xFF': 'local > 0x1FF',
        'encoded_tiers = live >> 8': 'encoded_tiers = live >> 9',
        '1..256 slots': '1..512 slots',
        'at most 256 experts': 'at most 512 experts',
        '[0, 256]': '[0, 512]',
        'exceeds 256 experts': 'exceeds 512 experts',
        'low 8 bits': 'low 9 bits',
        "descriptor's 8-bit": "descriptor's 9-bit",
    }
    for old, new in replacements.items():
        if old not in source:
            raise RuntimeError(f'Mixed descriptor source drift: {old}')
        source = source.replace(old, new)
    source += '\n# GLM53_DESCRIPTOR_ABI_9BIT\n'
    compile(source, str(path), 'exec')
    path.write_text(source)
path = root / 'fused_moe/_impl.py'
source = path.read_text()
old = '''            if int(weight_E) > 256:
                raise ValueError(
                    "projection-mixed Trellis supports at most 256 experts"
                )'''
new = old.replace('256', '512')
if new not in source:
    if source.count(old) != 1:
        raise RuntimeError('Projection workspace expert admission source drift')
    path.write_text(source.replace(old, new))
print('GLM mixed-Trellis descriptors: nine-bit local slots, up to 512 experts')
