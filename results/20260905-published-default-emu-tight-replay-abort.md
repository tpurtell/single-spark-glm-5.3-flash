# Tight-fit 1M replay probe: observed miss, then interrupted

At 2026-09-05 16:01:37 UTC, the owned HTTP client PID 2821578 was interrupted
with SIGINT after confirming its exact command and recording server metrics.
The client/pipeline exited with status 130. This was a deliberate stop after
an observed prefix-cache miss, not a polling timeout or an engine failure.

The published image, unmodified launcher defaults, utilization 0.85, and
1.01x request-equivalent KV capacity had passed cold near-1M retrieval:
1,048,320 prompt tokens, all six exact records, clean stop, TTFT 1,393.874 s.
The identical replay admitted one running request, zero waiting requests,
and zero additional prefix-cache hits; it was still recomputing over nine
minutes later. The global hit counter stayed at 46,080 (from the earlier
32K probe), with zero preemptions. The four-pass run is therefore **not a
passing prefix-reuse qualification**.

The interrupted second completion and the two unstarted isolation passes
have no final-answer score. `20260905-published-default-emu-prefix1m.json`
intentionally remains `complete: false`, with only its completed cold pass.
Neither its pass count nor elapsed time was fabricated. The partial log
retains the interruption traceback.

Evidence: `20260905-published-default-emu-tight-replay-inflight.metrics`,
`20260905-published-default-emu-tight-replay-before-abort.metrics`, the
in-flight serving log, and the original partial JSON/log. After interruption,
the server had zero running/waiting requests and zero preemptions. A separate
post-abort cancellation/recovery check and final environment capture follow;
their outcome is recorded in their own receipts.

A same-image, otherwise-default 0.86 comparison on dodo starts with 7.63 GiB
KV and 1.13 request-equivalents. Its four-pass near-1M test is ongoing. This
does not change the default from 0.85 and is not yet a qualified fallback.
