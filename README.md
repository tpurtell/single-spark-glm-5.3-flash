# GLM-5.3 Flash EXL3 K2 + DFlash2 on one DGX Spark

Run [GLM-5.3 Flash EXL3 K2](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2)
with [Inco DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) on one
DGX Spark, with **256K context and up to eight images per prompt**.
The ready-to-run ARM64 image uses NVFP4 cache, prefix caching, and B12x/SparkInfer
kernels, with six concurrent-request slots.

| Performance on one Spark | Default profile |
| --- | ---: |
| Decode, one request | 26.15 tok/s |
| Decode, six requests combined | 86.49 tok/s |
| Cold prefill, 32K prompt | 961 tok/s |
| Time to first token, cold 32K prompt | 34.11 s |

Measured on EMU with vision enabled and text prompts; medians of three decode
runs and two cold prefill runs. [Decode](results/20260906-emu-vision8-256k-code-agent.json)
and [prefill](results/20260906-emu-vision8-256k-prefill.json) results.

## Run

Run Docker and GPU work on a DGX Spark:

```bash
./download.sh  # omit when the pinned snapshots are already installed
./start.sh     # automatically pulls the pinned release if missing
docker logs -f glm53-spark
```

Published image: `ghcr.io/tpurtell/single-spark-glm-5.3-flash:20260905-k2-dflash2`.
The launcher pins its immutable registry digest:
`sha256:1e91406e6c9520bf0e102bd0ead3b43426740671f308ca81c948ad6010136009`.
GHCR visibility may remain private until the owner changes it; authenticate
with `docker login ghcr.io` if required. The API port is 8001 and has no
authentication by default; keep it on a trusted network. Models use the standard
`$HF_HOME/hub` cache, defaulting to `$HOME/.cache/huggingface/hub`.

For a source build on a Spark, use `./build.sh`, then
`IMAGE=ghcr.io/tpurtell/single-spark-glm-5.3-flash:dev ./start.sh`.

The default is vision enabled, NVFP4 MLA, **262,144 tokens**, DFlash2 with five
draft tokens, compact recurrent rollback, **prefix caching on**, six scheduler
slots, and **0.87 utilization** (also the hard maximum). The scheduler prefill batch
is 2048 tokens; EXL3's internal prefill capacity is 1024, as in the larger-batch
benchmarks below. The context limit is per request, including output; six
active requests share one cache pool, not six dedicated 256K caches.

FP8, native MTP, seven DFlash drafts, and full rollback use the 262K comparison
defaults (batch 2048, EXL3 prefill capacity 1024). For a matched short-context
NVFP4/FP8 comparison, explicitly use the same capacities and disable vision
to reproduce the historical measurements below:

```bash
LANGUAGE_MODEL_ONLY=1 GPU_MEMORY_UTILIZATION=0.85 \
  MAX_MODEL_LEN=262144 MAX_NUM_BATCHED_TOKENS=2048 \
  VLLM_EXL3_PREFILL_CAPACITY=1024 KV_CACHE_PROFILE=nvfp4 ./start.sh
# Stop the owned server before starting another profile on the same Spark.
./stop.sh
LANGUAGE_MODEL_ONLY=1 GPU_MEMORY_UTILIZATION=0.85 KV_CACHE_PROFILE=fp8 ./start.sh
```

Vision defaults to eight images per prompt, with video disabled. Set
`LANGUAGE_MODEL_ONLY=1` to disable vision.
Single-image and eight-image color/position smoke checks passed on EMU;
these are basic image-input checks, not a broad vision-quality benchmark.
[Vision results](results/20260906-emu-vision8-256k-vision.json).
See the [current release checks](results/RELEASE-20260906.md) for prefix reuse,
recovery, and the remaining warmup limitations.
`USE_REPLAYSSM=0` selects the full-rollback diagnostic path. The compact default
includes the GLM convolution-window fix.

## Runtime and model support

The runtime starts from the official vLLM GLM ARM64 image with Brandon's
GLM/EXL3/DFlash2 ports and B12x/SparkInfer kernels. Per-projection EXL3 supports
independent gate/up/down assignments within one adjacent Trellis tier pair per
MoE layer: K2/K3, K3/K4, K4/K5, or K5/K6. The adapter is tested; a future mixed
GLM checkpoint still needs model-quality testing.

Startup fetches the latest official `zai-org/GLM-5.3-Flash` chat template, with
verified cached/bundled copies for offline use. Set `GLM53_TEMPLATE_REFRESH=0`
to disable fetching. Weights remain read-only; overrides live in
`.cache/vllm/glm53`. Weights are not bundled in the image. The draft model's
CC BY-NC-ND 4.0 research/evaluation restrictions apply independently of this
recipe's Apache-2.0 license; see [PROVENANCE.md](PROVENANCE.md).

## Measured performance and quality

Measurements below use the published runtime with vision disabled; the 262K
comparisons use 0.85. They are historical tuning results, not measurements of
the new 256K vision default. Historical table labels use “262K” for the same
262,144-token context limit.
One Spark per endpoint; no image transfers during measurements. NVFP4 and FP8
used identical target/draft revisions, prompts, settings, and runtime image.
Code-agent results are medians of three runs, 256 output tokens per sequence.
C2/C4/C6 rates use the batch's first-any to last-any token window, **not the sum
of individual stream rates**. Host-to-host differences are not isolated kernel
effects.

| Code-agent decode, tok/s | NVFP4, 262K/.85 (ostrich) | FP8, 262K/.85 (kiwi) | NVFP4, 1M/.85 (dodo) | NVFP4, 1M/.87 (emu) |
| --- | ---: | ---: | ---: | ---: |
| C1 | 29.25 | 28.03 | 29.04 | 26.20 |
| C2 aggregate | 45.26 | 47.67 | — | — |
| C4 aggregate | 72.55 | 70.11 | — | — |
| C6 aggregate | 85.03 | 83.96 | 84.15 | 84.55 |

The published .87 profile was measured after its 1M/grammar checks, and
its lower C1 result is retained. C1 draft acceptance was 62.5%, versus
69.8% in the matched .85 NVFP4 short-profile run. These cross-host/session
results do not isolate a causal effect of the utilization setting.
[1M throughput receipt](results/20260905-published-087-emu-code-agent.json).

Native MTP with fixed 3/5/7 drafts measured C1 **19.86/20.74/18.63** and C6
**78.29/71.11/59.48** tok/s on emu (two runs, prior image before the grammar-only
fix). The earlier .85 DFlash2 1M run was about 40% faster at C1 and 7.5% faster
at C6 than the best respective native-MTP depth in that sweep. The original one-draft
row was an untuned control, not the best MTP baseline.
On the current image, target-only measured 10.36/48.46 tok/s at C1/C6;
that .85 five-draft 1M run was 2.80x/1.74x faster. Seven DFlash2 drafts measured
29.65/85.27 at C1/C6 in the 262K profile, giving little gain over five here.

Five-draft acceptance in the matched code-agent runs (three-run medians):

| Cache / concurrency | Drafts accepted | Committed tokens / verification | Acceptance by draft position 1–5 |
| --- | ---: | ---: | --- |
| NVFP4 C1 | 69.8% | 4.49 | 87.7 / 82.5 / 70.2 / 61.4 / 47.4% |
| FP8 C1 | 67.1% | 4.36 | 84.7 / 78.0 / 69.5 / 57.6 / 45.8% |
| NVFP4 C6 | 63.9% | 4.20 | 82.8 / 74.1 / 64.9 / 53.7 / 44.9% |
| FP8 C6 | 64.1% | 4.20 | 85.0 / 75.7 / 64.2 / 52.5 / 43.5% |

C1 decode-window time divided by verification count is approximately
153 ms (NVFP4) / 154 ms (FP8). This is an end-to-end speculative-cycle proxy,
including draft work and streaming, **not isolated target-kernel latency**.
C6 verification counters sum across requests and must not be interpreted as
serial GPU passes. Raw counts and timings:
[NVFP4](results/20260905-final-k5-nvfp4-ostrich-code-agent.json),
[FP8](results/20260905-final-k5-fp8-kiwi-code-agent.json).

| Cold prefill, exact prompt tokens | NVFP4 tok/s / TTFT | FP8 tok/s / TTFT |
| --- | ---: | ---: |
| 2,048 | 855 / 2.40 s | 859 / 2.38 s |
| 8,192 | 928 / 8.83 s | 930 / 8.80 s |
| 32,768 | 966 / 33.91 s | 969 / 33.83 s |
| 131,072 | 967 / 135.48 s | 965 / 135.84 s |

Prefill medians use two cold runs at each depth in the matched 262K profiles.
The default NVFP4 MLA record uses FP8 RoPE and occupies 368 bytes, compared
with 656 bytes for FP8 MLA; this is **not** a 44% reduction in total model,
recurrent-state, or draft-cache memory.

### Long context and memory

The tested text-only 1M/.87 profile on emu allocated 90.76 GiB during model loading and
admitted a 9.17 GiB cache pool. All four near-1M retrieval/replay/isolation
checks passed: cold TTFT was 23.12 minutes; identical replay was 29.82 seconds,
changed-tail reuse 52.37 seconds, and replay of the original 30.11 seconds.
[Four-pass receipt](results/20260905-published-087-emu-prefix1m.json).
Post-long cancellation/recovery and all 145 grammar cases passed, with no
OOM, restart, or net swap growth. One kernel compiled after startup during
grammar testing, so warmup does not cover every serving shape.

**262K and 1M in the performance tables are configured context limits, not cache
sizes.** The shorter benchmark profile also uses larger prefill batches. vLLM's
diagnostic “request equivalents” estimate how many maximum-length active
requests fit under each profile's hybrid attention/recurrent-state planner.
They are neither measured concurrency nor a guarantee of retained prefix reuse,
and should not be compared as a simple token-storage capacity across profiles.

The 1M tests used smaller prefill batches and, for consistent prefix reuse,
0.87 utilization. To reproduce that profile instead of the 256K vision default:

```bash
./stop.sh
LANGUAGE_MODEL_ONLY=1 MAX_MODEL_LEN=1048576 MAX_NUM_BATCHED_TOKENS=512 \
  VLLM_EXL3_PREFILL_CAPACITY=512 GPU_MEMORY_UTILIZATION=0.87 ./start.sh
```

The measurements and failed lower-headroom trials remain in the
[cache-headroom analysis](results/20260905-cache-headroom-analysis.md) and
[qualification log](results/QUALIFICATION-20260905.md). Available cache varies
with host memory and runtime reservations; model-load allocation excludes later
profiling/graph overhead. At 262K/.85, NVFP4 and FP8 model loads were both
90.93 GiB, with 8.10 and 9.86 GiB admitted cache respectively on different
hosts; those unequal pools do not isolate cache-format savings.

### Varied content and tools

Five-repeat C1 content medians on fresh matched 262K starts, same runtime:

| Content | NVFP4 tok/s | FP8 tok/s |
| --- | ---: | ---: |
| Code | 31.96 | 29.55 |
| Math | 25.45 | 29.67 |
| Fable | 15.37 | 13.96 |
| Short response | 31.19 | 30.81 |
| Exposition | 18.78 | 16.49 |
| JSON | 25.79 | 30.94 |
| Multilingual | 16.66 | 15.41 |

Both passed 30/35 structural contracts. NVFP4 failed the inherited math
input-literal check five times despite the correct numeric calculation;
FP8's five fables had 179 words, above the 170-word maximum. These are
additional runs, not replacements for the single-repeat C1/C6 results below.
Both sessions had zero post-ready JIT events and zero restarts.
Receipts: [NVFP4](results/20260905-content-repeat-nvfp4-ostrich-blend-c1.json),
[FP8](results/20260905-content-repeat-fp8-kiwi-blend-c1.json).

The separate low-entropy repeated-word diagnostic measured **44.20 / 43.32
tok/s** for NVFP4 / FP8 (five timed runs after one warmup, unique prompt
nonces). Both produced 101 occurrences instead of the requested 100 in all
five timed runs, so exact-count quality was **0/5** for both, with clean stops.
These speeds are not normal-prose or code-agent throughput.
Receipts: [NVFP4](results/20260905-content-repeat-nvfp4-ostrich-orchid.json),
[FP8](results/20260905-content-repeat-fp8-kiwi-orchid.json).

| Serving/quality check | NVFP4 K5 | FP8 K5 |
| --- | --- | --- |
| Tools/structured-output/termination canaries | 145/145 | 145/145 |
| Strict 32K cold/replay/changed-tail/original | 4/4, cache hits recorded | 4/4, cache hits recorded |
| Cancellation and exact recovery | 12 disconnects, 12 recoveries | 12 disconnects, 12 recoveries |
| RULER-lite, 8K/32K/128K | 12/12 | 12/12 |
| Seven-content contracts, C1 | 6/7 | 6/7 |
| Seven-content contracts, C6 | 29/42 | 34/42 |
| Seven-content blend decode, C1 / C6 | 19.20 / 66.64 tok/s | 17.94 / 64.41 tok/s |
| Tool Eval Bench, C6 | 124/138 points (90), 58 pass / 8 partial / 3 fail | 121/138 points (88), 56 pass / 9 partial / 4 fail |
| Tool Eval Bench, serial | 124/138 points (90), 58 pass / 8 partial / 3 fail | 120/138 points (87), 55 pass / 10 partial / 4 fail |
| Post-ready JIT events, complete 262K test session | 1 (`DSAFusedIndexerKernel`) | 1 (`DSAFusedIndexerKernel`) |

These quality scores are scoped: RULER-lite checks gold inclusion and clean
termination in synthetic tasks, not official RULER. Some correct common-word
sets were accompanied by incorrect unrequested counts. Content contracts are
structural, not execution-based code tests. Failures include the inherited
math input-literal check, fable word counts, six NVFP4 C6 exposition truncations,
and one FP8 C6 code truncation. Full outputs and failures are retained.
The standard tool suite uses its fixed March reference date, temperature 0,
low reasoning effort, 2,048 output tokens per turn, and seed 20260905.
Both C6 runs failed the benchmark's safety gate: injected text was reproduced
(TC-34) and an empty required search query was submitted (TC-43). Both also
failed TC-61; FP8 additionally failed TC-68. These are measured limitations,
not a claim of perfect agent reliability. Full traces:
[NVFP4 C6](results/tool-eval-reports/2026/09/2026-09-05T14-36-53.630358Z_1b157d06.md),
[FP8 C6](results/tool-eval-reports/2026/09/2026-09-05T14-36-53.671390Z_1fa68821.md).
Serial traces:
[NVFP4](results/tool-eval-reports/2026/09/2026-09-05T14-47-50.859912Z_0a270491.md),
[FP8](results/tool-eval-reports/2026/09/2026-09-05T14-47-55.230999Z_806f8187.md).
NVFP4's three serial failures match its C6 failures; FP8's fourth serial
failure was TC-08's conditional weather/reminder flow, not TC-68.

Separately, the published **1M deployment profile** on dodo scored
122/138 points (88), with 56 pass / 10 partial / 3 fail at C6. Its failures
were TC-34, TC-43, and TC-61; the safety gate still fails. This startup used
0.85 utilization and diagnostic `GLM53_MEMORY_TRACE=1`; it is neither a test of
the new 256K vision default nor another matched 262K cache-format measurement.
[Full deployment-profile trace](results/tool-eval-reports/2026/09/2026-09-05T15-07-22.307025Z_b6f0ea50.md).
That session subsequently passed all 36 rolling C6 requests across low/max
thinking and shared/unique prefixes, then all 12 cancellation/recovery pairs.
There were no loops, truncations, OOMs, or restarts, and no net swap growth
between snapshots. One post-ready `_kpool_softmax_rotate_write_cache_kernel`
compilation occurred during tools: the startup warmup does not cover every
serving shape. [Stress receipt](results/20260905-published-default-dodo-stress/summary.json),
[JIT audit (fails the strict zero-JIT check)](results/20260905-published-default-dodo-jit.json).

Prefix reuse is coarse with the current hybrid cache layout and speculative
lookback: K5 NVFP4 uses 15,360-token allocator blocks; K7 uses 18,432. A 32K K7
probe answered correctly but had zero hits; the 64K probe passed all four
retrieval/isolation checks with 110,592 hits. Do not infer short-prefix reuse
merely from `--enable-prefix-caching`. Seven drafts remain a tuning profile,
not the five-draft deployment default.
The K7 varied-content blend measured 17.46/62.71 tok/s at C1/C6, with
6/7 and 36/42 structural contracts passing. It also triggered one post-ready
`_kpool_tail_seed_kernel` compilation.

See [EVAL.md](EVAL.md) for qualification requirements and
[PROVENANCE.md](PROVENANCE.md) for source pins and adaptation evidence.
The [release audit](results/RELEASE-AUDIT-20260905.md) records the original
release's scope and limitations; its historical default was 0.85.
