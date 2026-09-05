#!/usr/bin/env python3
"""Reserve one block per direct mixed-Trellis route, not per packed block."""
import sys
from pathlib import Path

path = Path(sys.argv[1]) / 'b12x/moe/fused_moe/_impl.py'
source = path.read_text()
replacements = [(
    '''            route_blocks_capacity = (
                route_slots_capacity + block_size_m - 1
            ) // block_size_m
            sms = max(1, int(get_num_sm(device)))''',
    '''            route_blocks_capacity = max(
                (route_slots_capacity + block_size_m - 1) // block_size_m,
                min(routed_capacity, _MIXED_TRELLIS_DIRECT_ROUTE_LIMIT),
            )
            sms = max(1, int(get_num_sm(device)))'''), (
    '''    route_blocks = (route_slots + block_size_m - 1) // block_size_m
    rotation_input_dtype = _w4a16_element_dtype(core_plan.dtype)''',
    '''    route_blocks = max(
        (route_slots + block_size_m - 1) // block_size_m,
        min(capacity_rows, _MIXED_TRELLIS_DIRECT_ROUTE_LIMIT),
    )
    rotation_input_dtype = _w4a16_element_dtype(core_plan.dtype)'''),
]
for old, new in replacements:
    if new in source:
        continue
    if source.count(old) != 1:
        raise RuntimeError('Projection mixed-Trellis capacity source drift')
    source = source.replace(old, new)
compile(source, str(path), 'exec')
path.write_text(source)
print('Mixed-Trellis direct-route workspace capacity repaired')
