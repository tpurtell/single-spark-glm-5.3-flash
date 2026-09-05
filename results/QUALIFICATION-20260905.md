# Development and release qualification, 2026-09-05

Historical and release evidence are distinguished below. Baseline image:
`sha256:51390ff99614e19cdb8c880cd50eafff0fe5f02f383a1e2fdf7daf18cfaf11e8`.
The convolution-window correction was subsequently built on ostrich as
`sha256:e3b750a15e20cc8c07ee1d397b9a8d77a47a0a7b4f813721b96320e8baccd3f3`;
its 36-request rolling serving retest passed; later images completed the other gates.
The correction plus C6 strict-tool startup warmup was independently rebuilt
on dodo as
`sha256:513492a7c54be04163e2a850dabae5ffdb224c75d505f6b887ff67c647108020`
and transferred to emu for native-MTP depth tests. No weights are in the image.
The final XGrammar termination backport was subsequently built as
`sha256:94711456c8e9f17b849a9294fbb021245fc8a65b55d9e76a67fa8a0c77309095`.
This corrects the observed FP8 ignore-EOS HTTP 500; CPU acceptance passed,
followed by 145/145 live grammar cases on both NVFP4 and FP8. Matched serving
qualification is summarized in the README. Its metadata-only release is now published;
see [the runtime identities](../PROVENANCE.md#published-runtime-identity).
Source/model/template pins are in [PROVENANCE.md](../PROVENANCE.md).
No workstation GPU was used. Brandon's recipe and the B12x source checkout
were not modified.

## Initial serving measurements

The current-image matched results and promoted 1M default are summarized in
[README.md](../README.md#measured-performance-and-quality). The older tables
below are historical development records, not the selected deployment profile.

Current image `94711456…`, NVFP4 K5, 1M/.85/prefix-on/batch512/prefill512:
all four exact near-1M passes succeeded, then 6/6 low-thinking rolling C6
requests and both cancellation/recovery phases. C1/C6 decode medians were
29.04/84.15 tok/s. Before/after host available memory was 10.17/9.45 GB;
swap usage was 364,277,760/364,171,264 bytes (no net growth between snapshots).
Cache capacity was 1,221,641 token-equivalents, or 1.17x the 1M request limit.
The post-run environment audit recorded zero post-ready JIT compilations.
Receipts share the `20260905-final-1m-nvfp4-dodo-` prefix.

The optional K7 image passed 145/145 grammar cases. Its 32K prefix test had
four correct answers but zero cache hits, so the strict replay gate failed.
Inspection found allocator blocks of 18,432 tokens (K5: 15,360); speculative
lookback removes the final matching block. At 64K, the identical probe passed
4/4 with 110,592 hits and replay TTFT 31.38 s versus 71.00 s cold. This is
a coarse-reuse limitation, not evidence that all prefix lengths are reusable.
Both raw 32K and 64K results are retained.

Code-agent prompt, two runs per point, text-only 262,144-token model limit,
six scheduler slots, utilization 0.85. Rates below use the **batch window**
from first-any to last-any streamed token, not summed per-sequence rates.
These measurements preceded the frozen-image matched sweeps, varied-content
tests, and runtime stress qualification. Host and cache-admission differences make
these development comparisons, not isolated-kernel causal measurements.

| Speculation | KV | Recurrent rollback | Spark | C1 tok/s | C6 aggregate tok/s |
| --- | --- | --- | --- | ---: | ---: |
| Native MTP, 1 draft token (**untuned K1 control**) | NVFP4 | Compact | emu | 16.95 | 67.62 |
| Native MTP, 3 draft tokens (fixed-depth, corrected image) | NVFP4 | Compact | emu | 19.86 | 78.29 |
| Native MTP, 5 draft tokens (fixed-depth, corrected image) | NVFP4 | Compact | emu | 20.74 | 71.11 |
| Native MTP, 7 draft tokens (fixed-depth, corrected image) | NVFP4 | Compact | emu | 18.63 | 59.48 |
| DFlash2, 5 draft tokens | NVFP4 | Full | ostrich | 27.25 | 55.65 |
| DFlash2, 5 draft tokens | FP8 | Full | dodo | 26.43 | 54.56 |
| DFlash2, 5 draft tokens | NVFP4 | Compact | emu | 29.10 | 87.19 |
| DFlash2, 5 draft tokens | FP8 | Compact | dodo | 28.31 | 88.36 |
| DFlash2, 7 draft tokens | NVFP4 | Compact | ostrich | 29.16 | 95.12 |
| DFlash2, 5 draft tokens (**corrected compact image**) | NVFP4 | Compact | ostrich | 26.75 | 90.98 |

Raw receipts: `20260905-code-agent-initial-mtp1-emu.json`,
`20260905-code-agent-cg36-nvfp4-ostrich.json`,
`20260905-code-agent-cg36-fp8-dodo.json`,
`20260905-code-agent-replayssm-nvfp4-emu.json`, and
`20260905-code-agent-replayssm-k7-nvfp4-ostrich.json`.
The native-MTP run's startup predated completion of image transfers; its
memory-capacity measurement is not a clean-release control. All later
explicit-CG36 and compact-DFlash2 starts occurred without image transfers.

The K1 row is not a best-native-MTP baseline and does not establish DFlash2's
advantage over tuned native MTP. A fixed-depth native-MTP K3/K5/K7 sweep
(adaptation disabled, no external draft model) is complete at the user's
request. Its results are separate from DFlash2 draft-depth measurements.
K3 on corrected image `513492a7…` passed 145/145 grammar/termination checks;
C1/C6 draft acceptance was about 66.9%/74.2%. Its two-run throughput receipt
is `20260905-native-mtp-k3-nvfp4-emu-code-agent.json`. K5 also passed 145/145;
C1/C6 acceptance was 57.6%/56.5%, with rates 20.74/71.11. K7 passed 145/145
and measured 18.63/59.48 tok/s, with acceptance 44.1%/41.8%.
Higher native-MTP depth is not uniformly faster under concurrency.

Corrected DFlash2 K5's three-run C1/C2/C4/C6 medians were
26.75/45.59/76.13/90.98 tok/s. The seven-content blend's raw outputs and
per-sequence structural checks are also retained: C1 6/7 and C6 34/42
individual contracts passed. At C6, the six math failures are the inherited `25`
literal check; two C6 fables were below the requested word range. Generated
code is not executed by this contract harness. These remain development
results before the final grammar backport/image freeze.

The older v3 receipt's prose called summed per-sequence rates aggregate;
that wording is wrong under staggered admission. This table exclusively uses
`median_batch_window_decode_tokens_per_second`. Harness v4 makes that primary
definition explicit and retains the old field only for compatibility.

## Correctness and quality so far

| Gate | Development result and scope |
| --- | --- |
| Per-projection adapter | Full GLM H4096/I2048/E288/top-k8, independently assigned gate/up/down K2/K3 and K3/K4: 12/12 oracle/graph/stable-allocation cases. Small geometry K4/K5 and K5/K6: 24/24. See projection receipt. Not a future full mixed-checkpoint quality result. |
| Wide sparse MLA | 6/6 FP8/NVFP4 focused cases, 2,051 live candidates including tail candidates, empty rows, graph replay, and page offsets beyond 2 GiB. |
| 128K retrieval | Strict official-chat-framed test: exact 131,072 prompt tokens; all six final-answer records correct; clean stop, 74 output tokens. TTFT 155.238 s. `20260905-multi-needle-128k-chat-nvfp4-kiwi.json`. |
| 512K retrieval | Same full-rollback long-context profile, exact 524,288 prompt tokens; all six records correct and clean stop. TTFT 643.716 s. `20260905-multi-needle-512k-chat-nvfp4-kiwi.json`. |
| Near-1M retrieval | Exact 1,048,320 prompt tokens plus a 256-token output allowance reaches the 1,048,576 model envelope. All six records correct, clean stop after 74 output tokens. TTFT 1,369.284 s (22.82 min). Full rollback, NVFP4, text-only, utilization 0.87, prefix cache off, batch/prefill 512. `20260905-multi-needle-1m-chat-nvfp4-kiwi.json`. |
| **Near-1M with prefix ON at 0.85** | Corrected compact NVFP4 K5, text-only, batch/prefill 512: cold, identical replay, changed final fact, original-after-change all passed exact six-line answers and clean stops. Exact 1,048,320 prompt tokens each; TTFT 1,353.38 / 29.62 / 51.77 / 29.25 seconds. 3,072,000 prefix-cache hits. `20260905-fixed-replayssm-1m-prefix-nvfp4-dodo.json`. |
| Prefix replay/isolation | Compact DFlash2 K5, 32,768 exact prompt tokens: cold, identical replay, changed final record, then original replay all passed. 46,080 prefix-cache hits recorded. `20260905-prefix-replayssm-32k-nvfp4-emu.json`. |
| Rolling 32K C6 stress | **Failed on the uncorrected image:** compact NVFP4 K5, 36 requests, three loops detected. Five of 12 low-thinking requests hit the output limit; the v2 pass count of 33/36 did not reject all such truncations. V3 additionally requires a complete once-only final answer for the low-thinking lane. Same fixtures and seeds are retained for the corrected-image retest. `20260905-replayssm-k5-stress-nvfp4-emu/summary.json`. |
| Full-rollback rolling control | Same NVFP4 K5, 32K prompt, C6, seed and low-thinking fixture: 6/6 completed, no loops or token-limit finishes. `20260905-full-rollback-control-nvfp4-dodo/summary.json`. |
| Corrected compact rolling retest | NVFP4 K5, 32K, C6, same fixtures/seeds: 36/36 across low-thinking shared-prefix, max-thinking shared-prefix, and max-thinking unique-prefix phases. No loops, transport errors, marker misses, or token-limit finishes in any phase. `20260905-fixed-replayssm-k5-stress-nvfp4-ostrich/summary.json`. |
| Corrected compact tools/prefix/cancellation | NVFP4 K5 passed 145/145 tool/termination checks, all four strict 32K prefix/isolation passes, and both C6 cancellation/recovery phases. See `20260905-fixed-replayssm-k5-{grammar,prefix32k,cancellation}-nvfp4-ostrich.json`. |
| FP8 termination canary | Corrected compact FP8 K5 passed 144/145; one explicit ignore-EOS JSON request returned HTTP 500 after XGrammar termination. All normal required/named tools passed. The preserved receipt is `20260905-fixed-replayssm-k5-grammar-fp8-kiwi.json`; this is not a passing release gate. |
| Grammar/termination canary | Compact NVFP4 K7: 145/145 HTTP requests survived, but only 114/145 passed the stricter output checks. C6 had 28 wrong argument values and 3 incomplete tool calls. Sequential tools and controls passed. `20260905-grammar-replayssm-k7-nvfp4-ostrich.json`. This is a failed release gate, not a pass based on server survival. |
| Full-rollback grammar control | NVFP4 K5: all 145/145 passed the same strict output and termination checks, including C6 required/named tool calls. `20260905-grammar-full-rollback-k5-nvfp4-dodo.json`. |
| Post-1M tools | Kiwi's full-rollback long-context profile passed 145/145 strict tools/termination cases after the near-1M retrieval. `20260905-post-1m-grammar-full-rollback-kiwi.json`. |
| Post-1M cancellation | Six prefill streams and six decode streams disconnected; both queues drained, all 12 subsequent exact recovery answers passed, endpoint remained healthy. `20260905-post-1m-cancellation-full-rollback-kiwi.json`. |
| RULER-lite | Same Kiwi profile: 12/12 at exact 8K, 32K, 128K rendered prompt lengths, covering single/multi-key retrieval, variable tracking, and common-word extraction. Gold inclusion and clean stop only; facts are early in these synthetic tasks, not official RULER or distributed-depth retrieval. The model added some incorrect, unrequested word-frequency counts despite finding the correct common-word sets; raw outputs are retained. `20260905-ruler-full-rollback-1m-nvfp4-kiwi.json`. |
| Seven content contracts | Initial NVFP4/full 5/7; FP8/full 7/7; NVFP4/compact K5 6/7. Structural checks only: generated Python is not executed by this harness. |
| Tool Eval Bench | Initial FP8/full C6: 123/138 points, rounded score 89, 57 pass / 9 partial / 3 fail. Full mock-tool traces retained in JSON and Markdown, with SQLite persistence through the benchmark CLI. Not a serial or compact-rollback quality comparison. |

Content-check limitations matter: the correct math answer using `0.75`
instead of literal `25` fails the inherited lexical check. The FP8 code case
passed the structural checker but contained an incorrect example assertion
merging non-overlapping intervals. No semantic-code-correctness claim follows
from its 7/7 structural score. The original raw-completion 128K retrieval hit
the output limit and repeated part of the answer; it is superseded for the
quality claim by the strict chat-framed receipt, not silently discarded.

Tool failures were prompt-injection content leakage (TC-34), an empty required
search query (TC-43), and not attempting the requested analysis script
(TC-61). Partial scores and full traces remain in
[the tool report](tool-eval-reports/2026/09/2026-09-05T11-58-36.102200Z_1a2e6995.md).

## Memory and optimization findings

Historical full-rollback text-only 1M startup did **not** fit at 0.85. At 0.87, NVFP4/full rollback,
prefix caching off, and batch/prefill capacity 512, it allocated 8.91 GiB KV
and reported 1,291,157 token-equivalents / 1.23x the 1M request limit. This is
capacity planning, not six simultaneous 1M requests. The exact near-1M
six-record retrieval passed; post-long-context tool/stability checks later
passed as recorded above. Host available memory
after the near-1M request was 7.91 GB. Swap usage grew by about 0.2 MiB during
the long-context ladder (7.42 MB to 7.63 MB), so this is not a literal zero-swap-
growth claim. No OOM or engine restart occurred.

Prefix-off was a memory-fit experiment, **not a deployment requirement or a
benchmark-cache shortcut**. Utilization, rollback layout, and chunk sizes also
changed, so it does not isolate prefix caching's memory cost. The corrected
compact image's prefix-off 1M startup fits at 0.85 with batch/prefill 512
(7.76 GiB KV, 1.19x request-equivalent capacity). The matched **prefix-on**
startup also fits at 0.85 (7.68 GiB KV, 1,191,100 token-equivalents / 1.14x
the 1M request limit). All four strict near-1M retrieval/replay/isolation passes
then succeeded. This establishes that prefix-off is not necessary for the
corrected compact profile's 1M fit or these retrieval tests.
Deployment/default qualification keeps prefix caching on. Following the
current-image retest above, launcher defaults are text-only 1M, prefix-on,
compact rollback, five DFlash2 drafts, 512-token batch/prefill capacities,
and utilization 0.85. Other diagnostic profiles retain 262K defaults.

The sparse-memory patch removed an over-conservative 40-sequence indexer
workspace reservation and an unreachable dense-MLA profiling allocation on
the B12x path. The actual B12x workspaces remain profiled. All launchers reject
utilization above 0.87.

The rolling failure exposed a concrete missing GLM ReplaySSM convolution
port: eager mixed batches have one compact state-table column, but GLM used
that column count for `max_query_len`. The convolution updates its actual
live token count internally, while its preplanned state-write extent still
uses the supplied capacity. This leaves later convolution history entries
stale. Pure-decode graph metadata broadcasts the column into a K+1-width
buffer, hiding the defect in short throughput tests. The upstream series
already corrects this for Qwen GDN/Mamba2; the GLM caller was missed.

`port-glm-replayssm-conv-window.py` now supplies K+1 for compact rollback.
Focused GPU oracle: four cases (5/7 draft tokens; 128/24,576 channels; mixed
live lengths and strided state IDs) passed independent output/state checks,
CUDA graphs, and stable allocations. The one-column negative control left
640–171,839 stale state elements. This proves the defect and local correction,
not by itself resolution of every observed serving failure. The subsequent
36-request rolling retest passed; strict-tool requalification subsequently
passed 145/145 on both final cache profiles. Receipt:
`20260905-replayssm-conv-window-oracle-ostrich.jsonl`.

Torch profiling identified BF16 dense GEMMs and Trellis MoE as the main decode
costs. A 32-shape cuBLAS/cuBLASLt graph probe passed FP32-oracle checks
(max relative error 0.001706, min cosine 0.9999985) but did not establish a
general speed benefit, so no global BLAS switch was adopted. Raw diagnostic:
`20260905-dense-blas-probe-ostrich.jsonl`. These synthetic timings are not
serving throughput measurements.

The date-overridden tool runs are explicitly
superseded; see `20260905-tool-eval-reference-date-misconfiguration.md`.

Update: C6 standard-default Tool Eval Bench is complete (NVFP4 124/138 points,
FP8 121/138; both retain safety-gate failures), target-only and K7 tuning are
complete, and the metadata-only image is published. Serial scores are
NVFP4 124/138 and FP8 120/138. Target-only C1/C6:
10.36/48.46 tok/s; K7 C1/C2/C4/C6: 29.65/48.59/72.87/85.27. The K7 serving
audit found one post-ready `_kpool_tail_seed_kernel` compile. K5 1M and
target-only post-run audits found none.

## Published-image validation

The registry pull on emu completed and the unmodified 1M/.85/C6/prefix-on
launcher defaults reached healthy startup. Its KV capacity is 6.85 GiB,
1,058,756 token-equivalents / 1.01x. It passed 145/145 grammar canaries,
four strict 32K prefix passes, exact 128K and 512K six-record retrieval,
and 12 cancellation/recovery pairs. The 512K cold TTFT was 652.520 seconds.
The same process subsequently passed cold near-1M retrieval but missed
identical prefix reuse; the deliberate interruption is recorded below.
Receipts: `20260905-published-default-emu-*`.

A preceding dodo restart with the same default command failed KV admission:
6.72 GiB available versus 6.75 GiB required. It exited, without an OOM or
automatic utilization/context adjustment. The preserved failed startup,
inspect, and state files use `20260905-published-default-dodo-failed-*`.
A diagnostic retry with `GLM53_MEMORY_TRACE=1` fits at 6.96 GiB / 1.03x.
The memory profiler accounts for changes in global free memory on UMA;
the source of this startup variance is not established by these snapshots.
The successful earlier 1.17x measurement is not a universal startup guarantee.

The diagnostic dodo deployment-profile Tool Eval Bench C6 run completed at
122/138 points (88), 56 pass / 10 partial / 3 fail. Failures are TC-34,
TC-43, and TC-61, including the two safety-gate failures. Its fixed benchmark
reference date and complete traces are retained in
`20260905-published-default-dodo-tool-eval-c6.json` and
[the Markdown report](tool-eval-reports/2026/09/2026-09-05T15-07-22.307025Z_b6f0ea50.md).
Its full 36-request rolling stress test passed all three phases (12/12 each),
with zero loops, transport errors, marker misses, or token-limit finishes.
The subsequent 12 cancellation/recovery pairs passed too. Available host
memory was 10.331 GB before / 10.194 GB after; swap usage decreased by 8 KiB
between snapshots. No OOM or restart occurred. The post-ready JIT audit
**failed its strict zero-event check**: one
`_kpool_softmax_rotate_write_cache_kernel` compilation occurred during the
tool suite. This is a documented warmup coverage gap, not a zero-JIT pass.
See `20260905-published-default-dodo-{post-environment,jit,cancellation}.json`,
the full serving log, and `20260905-published-default-dodo-stress/summary.json`.
The completed dodo test server was stopped before the later 0.86 comparison.

The five-repeat C1 content comparisons completed on fresh matched 262K
starts (ostrich NVFP4 / kiwi FP8), with 30/35 structural contracts passing
on each. NVFP4's five math failures preserve the inherited lexical-input
check; FP8's five fables were 179 words, outside 140–170. Per-content rates
are in the README. The separate repeated-orchid diagnostic measured
44.196 / 43.322 tok/s, but both produced 101 instead of 100 words in every
timed run (0/5 exact contracts each, all clean stops). Both complete sessions
had zero post-ready JIT events and zero restarts. Before/after environment
receipts and full serving logs share `20260905-content-repeat-*` prefixes.
The repetition client has a CPU-only regression test proving warmup exclusion,
strict word/count/finish scoring, unique nonces, and unresolved-rate exclusion.

Final kernel reruns close the earlier-image provenance gap: the published
image passes all 36 projection-mixed cases, and its equivalent frozen runtime
passes six wide-MLA plus four convolution-state oracle cases. Exact image
IDs and matching script hashes are in
`20260905-published-projection-oracle-dodo.log` and
`20260905-frozen-wide-conv-oracle-ostrich.log`. The two one-shot test jobs
exited successfully and removed their own temporary containers.

The published emu 1.01x-capacity startup passed cold near-1M retrieval
(TTFT 1,393.874 s), but the identical replay recorded zero additional prefix
hits and recomputed. This does not qualify tight-fit fast replay, unlike
the earlier 1.17x-capacity run. The original four-pass client was interrupted
after the miss was established; its partial JSON is explicitly `complete: false`.
The second answer and two unstarted isolation passes are unscored. In-flight metrics and
serving state/logs are preserved as `20260905-published-default-emu-tight-replay-inflight.*`.
Additional cache headroom is a hypothesis, not a proven root cause yet.
An explicit 0.86 utilization comparison was run on dodo, using the
same published image and otherwise-default launch settings. It does not
change the launcher default of 0.85 or bypass the 0.87 hard cap.

The emu client exited 130 following SIGINT at 16:01:37 UTC, not an observation
timeout. Its queue drained with zero preemptions; the separate post-abort
cancellation/recovery test passed all 12 pairs. Final host available memory
was 10.060 GB; swap usage decreased by 12 KiB from the pre-suite snapshot.
There was no OOM/restart. One post-ready
`_kpool_softmax_rotate_write_cache_kernel` compilation occurred during the
earlier grammar suite. The emu server was then stopped. See
`20260905-published-default-emu-tight-replay-abort.md`, the full serving log,
and `20260905-published-default-emu-post-abort-cancellation.json`.

The published 0.86 dodo start reached healthy status with 7.63 GiB KV,
1,180,920 token-equivalents / 1.13x, and 10.557 GB host available memory.
No default was changed. Cold and identical replay passed (TTFT 1,353.016 /
29.437 s, 1,029,120 cached tokens), but changed-tail reuse missed. The owned
client was deliberately interrupted after confirming that miss; its last
two answers are unscored and its partial receipt remains incomplete. The
follow-up grammar/throughput prerequisite guard correctly exited 1.
The separate cancellation/recovery check passed 12 pairs; post-ready JIT
count was zero, no OOM/restart occurred, available host memory was 9.885 GB,
and swap usage decreased 8 KiB. Dodo was then stopped. Receipts:
`20260905-published-086-dodo-*`, especially `changed-tail-abort.md`.

The explicit 0.87 upper-bound fallback ran on emu, with otherwise
identical published-image defaults. Its prefix checker is v4: positive cache
hits are required independently on each of the three warm passes, plus zero
cold hits and all four exact answers/clean stops. Five CPU scoring regressions
pass, including main-loop counter accounting and failed-report retention.
Older v3 receipts retain their original scoring and are not overwritten.
The .85 default and .87 hard ceiling remain unchanged.

At 9.17 GiB KV / 1,415,068 token-equivalents / 1.35x, **all four v4 passes
succeeded**. TTFT: 1,387.216 / 29.820 / 52.370 / 30.115 seconds; per-pass hits:
0 / 1,029,120 / 1,013,760 / 1,029,120. Each prompt was exactly 1,048,320 tokens,
each answer contained the exact six records, and each stopped after 74 output
tokens. Post-long cancellation/recovery passed all 12 pairs. Final grammar
passed 145/145. Three-run code-agent medians were C1 26.2015 / C6 84.5537
batch-window tok/s; median draft acceptance 62.54% / 61.98%. This C1 result
is lower than the earlier .85 measurements and is retained without substitution.
Available host memory was 8.307 GB before / 7.488 GB after; swap usage stayed
at 233,562,112 bytes. No OOM/restart occurred. One post-ready
`_kpool_softmax_rotate_write_cache_kernel` compilation occurred during grammar.
The pipeline exited successfully after its final environment/serving-log
capture at 17:07 UTC. The qualified explicit-fallback server remains on emu;
the launcher default remains .85.
Receipts: `20260905-published-087-emu-*`.
