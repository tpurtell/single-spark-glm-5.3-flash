#!/usr/bin/env python3
"""CPU-only aggregation of Torch Chrome-trace CUDA kernel durations."""
import argparse
import collections
import gzip
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('trace', type=Path)
args = parser.parse_args()
opener = gzip.open if args.trace.suffix == '.gz' else open
with opener(args.trace, 'rt') as stream:
    trace = json.load(stream)
durations, counts = collections.Counter(), collections.Counter()
for event in trace['traceEvents']:
    if event.get('cat') != 'kernel' or 'dur' not in event:
        continue
    durations[event['name']] += event['dur']
    counts[event['name']] += 1
total = durations.total()
print(json.dumps({'trace': str(args.trace), 'kernel_duration_sum_ms': total / 1000,
                  'kernels': [{'name': name, 'sum_ms': value / 1000, 'calls': counts[name],
                               'mean_us': value / counts[name], 'fraction': value / total}
                              for name, value in durations.most_common(40)]}, indent=2))
