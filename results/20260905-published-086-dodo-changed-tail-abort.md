# 0.86 comparison: identical replay passes, changed-tail reuse misses

Published image `b5ae51f7…`, otherwise-default 1M/C6/APC-on configuration,
utilization 0.86, 7.63 GiB KV / 1.13 request-equivalents.

Two completed requests passed exact six-record answers and clean termination:
cold TTFT 1,353.016 seconds; identical replay TTFT 29.437 seconds with
1,029,120 cached tokens. The changed-tail request then admitted with no
additional prefix hits. Its metrics recorded 3,166,833 cumulative queried
tokens and the same 1,029,120 hit count, with zero preemptions.

The exact owned client PID 2872582 was checked and interrupted with SIGINT
around 16:30 UTC after this miss was established. The pipeline exited 130.
The changed-tail answer and unstarted original-after-change pass are unscored;
the partial JSON remains `complete: false`. The queued post-long grammar and
throughput job correctly stopped at its prerequisite check (exit 1), rather
than treating the partial result as a pass.

The separate post-abort cancellation/recovery run passed all 12 pairs.
The final environment recorded zero post-ready JIT events, no OOM/restart,
9.885 GB host available memory, and 8 KiB less swap usage than before the
test. The dodo server was then stopped. Raw receipts share
`20260905-published-086-dodo-` and include the partial JSON, miss metrics,
interruption log, full serving log, and post-abort recovery/environment.

This establishes useful identical replay, **not** full cached changed-prefix
isolation qualification at this capacity. The recipe now tests the allowed
0.87 upper-bound fallback on emu. Default utilization remains 0.85.

The v3 test only required positive hits somewhere across the run. The new v4
checker records a hit delta per pass and requires zero cold hits plus positive
hits on every warm pass, alongside all four exact-answer/termination checks.
Four CPU regression tests cover missing hits on each warm pass, incomplete
runs, wrong answers, contaminated cold runs, and reset counters. Older v3
receipts are retained unchanged; stronger scoring is not applied retroactively.
