# Active-request fit versus reusable prefix capacity

Inspected source: published image
`sha256:b5ae51f7229f51d0afbb217461811411f9ea39f0cd167b893e5e59843139c3e9`.
Relevant exact excerpts are in `20260905-published-cache-headroom-source.txt`.

The printed request-equivalent capacity comes from
`vllm/v1/core/kv_cache_utils.py:get_max_concurrency_for_kv_cache_config`:
pool block count divided by the sum of each cache group's maximum active
request demand. `MambaSpec.max_memory_usage_bytes` in `align` mode budgets
`2 + num_speculative_blocks` state pages, not a retained state snapshot at
every historical prefix boundary. `BlockPool.get_usage` additionally excludes
the reserved null block from usable capacity.

`HybridKVCacheCoordinator.find_longest_cache_hit` reconciles the group hits
to a common boundary: any group may reduce the reusable length. Therefore
retaining attention blocks alone is insufficient if the matching recurrent
state or another participating group's required blocks are unavailable.
This is a property of the inspected implementation, not evidence of corrupted
weights or incorrect cached state.

Observed serving evidence:

- The earlier 1.17x-capacity .85 run passed all four strict near-1M requests
  with 3,072,000 cache hits and approximately 29-second identical replay.
- The published 1.01x-capacity .85 run passed cold retrieval but had zero
  additional hits on identical replay. It had zero preemptions, remained
  healthy, and recovered after the deliberate client interruption.
- The .86 comparison admits 1.13x capacity on dodo; its full test is running.

Inference: the near-zero spare capacity is a plausible reason that active
1M serving succeeded while replay failed. We have not captured per-group
eviction traces, so the precise missing group/block and the root cause are
not established. Cross-host observations do not isolate all host effects.
No fixed minimum replay-headroom percentage is proven by these samples.

The runtime also contains optional sparse-retention machinery, but it is
**not enabled or qualified by this recipe**. Reducing retention would change
which prefix boundaries remain reusable; it is not silently substituted for
the tested dense-retention behavior. The explicit .86 comparison changes
only the utilization override and stays below the .87 hard cap.
