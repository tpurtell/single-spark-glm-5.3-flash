#!/usr/bin/env python3
"""Summarize a trusted, locally generated PyTorch memory diagnostic (CPU only)."""
import collections
import json
import pickle
import sys


def stack_key(frames):
    return tuple(f"{f['filename']}:{f['line']}:{f['name']}" for f in frames[:8])


def groups(entries):
    totals = collections.Counter()
    counts = collections.Counter()
    for entry in entries:
        key = stack_key(entry.get('frames', []))
        totals[key] += entry.get('requested_size', entry['size'])
        counts[key] += 1
    return [dict(bytes=size, count=counts[stack], stack=stack)
            for stack, size in totals.most_common(25)]


def summarize(snapshot):
    allocations = [e for trace in snapshot['device_traces'] for e in trace
                   if e['action'] == 'alloc']
    active = [b for s in snapshot['segments'] for b in s['blocks']
              if b['state'] == 'active_allocated']
    return dict(
        total_active_bytes=sum(b['requested_size'] for b in active),
        active_by_stack=groups(active),
        profile_allocation_churn_by_stack=groups(allocations),
        largest_profile_allocations=sorted(allocations, key=lambda e: e['size'],
                                           reverse=True)[:12],
    )


if __name__ == '__main__':
    # pickle is not safe for third-party files. This input must be the trace
    # produced by this recipe's opt-in GLM53_MEMORY_TRACE on our own Spark.
    with open(sys.argv[1], 'rb') as stream:
        print(json.dumps(summarize(pickle.load(stream)), indent=2))
