# Release scope audit, 2026-09-05

Status: the requested recipe, published image, and evaluation deliverables
are complete, with the limitations below. The explicit .87/1.35x fallback
passed all four near-1M requests under the stronger v4 per-pass cache-hit
gate, 12 cancellation/recovery pairs, 145 grammar checks, and the C1/C6
throughput recheck. Default remains .85; no automatic utilization increase
or context reduction was added. This is not a blanket assertion that every
quality, startup, or zero-JIT check passed.

| User requirement | Evidence and scope |
| --- | --- |
| One Spark, official GLM base, current pinned B12x fork | Dockerfile pins official ARM64 base and B12x `c90cd008…`; `20260905-conv-grammar-build-ostrich.log` ends in successful ARM64 image export. Final environment receipts identify one NVIDIA GB10 and tensor parallel size 1. |
| K2 target plus Inco DFlash2 | Launcher pins both snapshots; all four `20260905-checkpoint-*.json` receipts verify shard/index coverage. Served environment receipts record source and repaired manifests, read-only weights, and actual speculative config. Full tensor-content hashing is a separate scope. |
| Uniform and per-projection Trellis | Uniform K2 is exercised throughout real serving. Published-image synthetic loader/adapter retest passes 36/36, covering independent gate/up/down assignments, adjacent tier pairs K2/3 through K5/6, dense reconstruction, graph replay, and stable allocation. Receipt: `20260905-published-projection-oracle-dodo.log`; a future mixed GLM checkpoint needs model-level qualification. |
| B12x primitives and GLM-specific fixes | `20260905-frozen-wide-conv-oracle-ostrich.log` passes 6 wide MLA and 4 convolution-state cases on frozen image `94711456…`, including >2 GiB offsets and saved-state negative controls. Real serving covers Trellis MoE, sparse indexing, and attention. Dense BLAS probe did not justify a global switch. |
| DFlash2 performance above controls | Final code-agent receipts: 1M DFlash2 29.04/84.15 tok/s at C1/C6 versus target-only 10.36/48.46. Native MTP fixed 3/5/7 sweep completed; best respective rates 20.74/78.29. Native tests precede the grammar-only fix and are labeled accordingly. |
| NVFP4 default and FP8 comparison | README reports matched 262K C1/C2/C4/C6, prefill, varied content, acceptance, memory, RULER-lite, tools, and stability. Cache profiles share model/draft/image/settings; host effects and unequal admitted KV pools are disclosed. |
| Memory utilization .85, hard maximum .87 | `start.sh` defaults .85 and rejects non-finite or >.87 values; seven CPU launcher tests pass. The serving command is present in each environment receipt. No automatic increase is implemented. |
| Ideally 1M, six concurrency slots | Frozen runtime's four near-1M requests passed at .85/1.17x with APC on, followed by C6 stress/recovery. Published .87/1.35x passes v4 with 0 / 1,029,120 / 1,013,760 / 1,029,120 per-pass hits (`20260905-published-087-emu-prefix1m.json`). Exact prompt 1,048,320 plus 256 allowance reaches 1,048,576. Six slots share one pool, not six million-token requests. Lower-headroom replay misses are retained. |
| Prefix caching enabled | Launcher default on; 32K and near-1M receipts prove cache hits and changed-prefix isolation. Coarse hybrid block granularity is disclosed; K7 32K zero-hit failure is retained alongside its passing 64K retest. |
| Fresh official chat template | Auto-refresh resolves immutable official revision, validates template, and uses verified cache/bundle fallback. Environment records actual served hash and official revision. Template CPU tests pass; real tool/grammar tests use the override. |
| MIA-style content and quality coverage | Seven content categories, five-repeat C1 diagnostic, separate repeated-word test, cold prefill, RULER-lite, context ladder, structured/tool canaries, Tool Eval Bench serial/C6, rolling stress and cancellation are reported with raw outputs. These are not execution-based code evaluation or official RULER. |
| Frozen/published serving stability | Published dodo passes 36/36 rolling requests and 12 cancellation/recovery pairs, zero loops/truncations/OOM/restarts, no net swap growth. Published .87 emu also passes post-1M recovery and 145/145 grammar, with no OOM/restart or net swap growth. Its final C1/C6 rates are 26.20/84.55 tok/s. One post-ready JIT event in each broad published session fails strict zero-JIT checks and remains documented. |
| Build/run scripts and published container | Executable `build.sh`, `download.sh`, `start.sh`, `stop.sh`; shell syntax and CPU client/launcher checks pass. Immutable GHCR digest `1e91406e…` is in push receipt, emu clean-pull receipt, and launcher default. Seven metadata-promotion equivalence checks pass. Weights are not bundled; owner controls public visibility. |
| Do not use workstation GPUs or alter Brandon | Build and GPU test commands run over SSH on named Sparks; local clients are CPU HTTP clients. Runtime changes are confined to this repository/container patch stack. Brandon and the B12x source checkout were not edited. |

## Explicit limitations retained

- One clean 1M/.85 startup failed admission (6.72 GiB versus 6.75 required).
  Other starts succeeded with 1.01–1.17 request-equivalents. Startup fit depends
  on host/runtime memory admission; the cause of the variance is not proven.
- At .85/1.01x, identical replay had no hits; at .86/1.13x, identical replay
  worked but changed-tail reuse missed. Those probes were interrupted after
  the misses were established; their uncompleted answers remain unscored.
  The passing .87/1.35x fallback is explicit, not a universal minimum
  headroom guarantee or a replacement for the .85 default.
- Published 1M Tool Eval Bench scored 122/138, including three failures and
  a failed safety gate. Matched NVFP4/FP8 serial and C6 scores and failures are
  separate; neither result establishes safe autonomous deployment.
- Both five-repeat content suites passed 30/35 structural contracts. Both
  repeated-word diagnostics failed exact count 0/5 while stopping cleanly.
- One post-ready kernel compile occurred in each broad 262K test session,
  and one in each broad published 1M session. Zero-JIT results from
  narrower sessions must not be generalized.
- Per-projection support is one adjacent tier pair per layer, not arbitrary
  three-tier mixtures. Vision is opt-in and unqualified. Neither is disguised
  as model-level quality evidence for the current uniform K2 checkpoint.

Completed Tool Eval Bench runs have full-trace JSON/Markdown plus SQLite
persistence through the benchmark CLI. Latest history receipt:
`20260905-tool-eval-sqlite-history-final.txt`. The wrong-reference-date runs
and earlier runtime failures are preserved and explicitly superseded, not
rescored or deleted.

At handoff, emu retains the healthy published-image .87 fallback server on
port 8001. Our serving containers on the other Sparks are stopped; one-shot
oracle containers removed themselves. No serving test remains queued.
