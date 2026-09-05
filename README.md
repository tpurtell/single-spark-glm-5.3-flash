# GLM-5.3 Flash EXL3 K2 + DFlash2 on one DGX Spark

Release qualification is finishing: the ARM64 image is published to GHCR;
the published-default stress/context checks are still running. Measurements
are in [the qualification log](results/QUALIFICATION-20260905.md); they are not
frozen-release guarantees.

**Startup qualification caveat:** one clean 1M/0.85 restart was rejected by
the KV admission check (6.72 GiB available versus 6.75 GiB required), despite
the identical command/environment passing earlier. A diagnostic retry fits
at 0.85; a fresh registry pull on emu also starts with unmodified defaults,
but with only 1.01 request-equivalents of cache. Its 128K/512K retrieval and
cancellation checks passed. Its cold near-1M answer also passed, but the
identical replay has recorded **zero cache hits** and is recomputing the
prompt; this tight-fit startup is not qualified for fast 1M replay. The earlier
1.17x-capacity run did reuse the prefix. The launcher does not silently
raise utilization or reduce context after a failed admission.

Target: [vcruz305/GLM-5.3-Flash-EXL3-K2](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2).
Draft: [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2).
The production profile uses NVFP4 MLA cache, one GB10 GPU, six scheduler
slots, and 0.85 GPU memory utilization. The launcher rejects utilization above
0.87. A text-only 1,048,576-token model limit has passed near-1M six-record
retrieval, identical replay, and changed-prefix isolation at 0.85 with prefix
caching on. Post-long-context qualification and the complete release matrix
are still running.

The runtime starts from the official vLLM GLM ARM64 image and carries the
GLM/EXL3/DFlash2 patch stack from Brandon's two-RTX recipe. B12x/SparkInfer
supplies Trellis MoE, sparse attention/indexing, and admitted dense primitives.
FP8 is retained for controlled cache-format comparisons, and native MTP for
the performance floor.

Per-projection EXL3 is included: gate, up, and down are independently assigned
within one adjacent pair of MCG Trellis tiers per MoE layer (K2/K3, K3/K4,
K4/K5, or K5/K6). The checkpoint-wide
average does not override tensor bit widths. Qualification of the GPU adapter
is separate from model-quality testing of a future mixed-bitrate GLM quant.

For the current vcruz checkpoint, startup automatically resolves the latest
official `zai-org/GLM-5.3-Flash` revision and fetches its chat template. A verified
bundled copy supports first boot offline; a verified cache is reused if the
network is unavailable. Set `GLM53_TEMPLATE_REFRESH=0` to disable fetching.
The downloaded weights remain read-only; overrides live in `.cache/vllm/glm53`.
Weights are not bundled in the container. The draft model card specifies
CC BY-NC-ND 4.0 for research/evaluation; its restrictions apply independently
of this recipe's Apache-2.0 license. See [PROVENANCE.md](PROVENANCE.md).

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
with `docker login ghcr.io` if required. The API port is 8001. Models use the standard
`$HF_HOME/hub` cache, defaulting to `$HOME/.cache/huggingface/hub`.

For a source build on a Spark, use `./build.sh`, then
`IMAGE=ghcr.io/tpurtell/single-spark-glm-5.3-flash:dev ./start.sh`.

The default is text-only, NVFP4 MLA, **1,048,576 tokens**, DFlash2 with five
draft tokens, compact recurrent rollback, **prefix caching on**, six scheduler
slots, and **0.85 utilization**. Batch and EXL3 prefill capacities are 512.
This does not imply six concurrent million-token requests: the measured shared
cache capacity was 1.17 request-equivalents at the 1M limit.

The current image passed all four near-1M retrieval/replay/isolation requests,
then C6 rolling stress and cancellation/recovery. Cold TTFT was 22.5 minutes;
identical replay was 28.8 seconds. No post-ready JIT compilations or engine
restarts were observed in that run; host swap usage did not grow between the
before/after snapshots. Published-default stress/context checks remain open.
An explicit 0.86 fallback is being tested for additional 1M cache headroom;
it is not the default or a qualified recommendation yet.

FP8, native MTP, seven DFlash drafts, and full rollback use the 262K comparison
defaults (batch 2048, EXL3 prefill capacity 1024). For a matched short-context
NVFP4/FP8 comparison, explicitly use the same capacities:

```bash
MAX_MODEL_LEN=262144 MAX_NUM_BATCHED_TOKENS=2048 \
  VLLM_EXL3_PREFILL_CAPACITY=1024 KV_CACHE_PROFILE=nvfp4 ./start.sh
# Stop the owned server before starting another profile on the same Spark.
./stop.sh
KV_CACHE_PROFILE=fp8 ./start.sh
```

Vision is opt-in with `LANGUAGE_MODEL_ONLY=0`; its memory/quality profile is not
yet qualified, and it defaults to one image per prompt.
`USE_REPLAYSSM=0` selects the full-rollback diagnostic path. The compact default
includes the GLM convolution-window fix: its serving retest passed 36/36 rolling
requests without loops or truncation. Historical failures remain in the log.

## Measured performance and quality

Current qualification image: `sha256:94711456c8e9f17b849a9294fbb021245fc8a65b55d9e76a67fa8a0c77309095`.
One Spark per endpoint; no image transfers during measurements. NVFP4 and FP8
used identical target/draft revisions, prompts, settings, and runtime image.
Code-agent results are medians of three runs, 256 output tokens per sequence.
C2/C4/C6 rates use the batch's first-any to last-any token window, **not the sum
of individual stream rates**. Host-to-host differences are not isolated kernel
effects.

| Code-agent decode, tok/s | NVFP4, 262K (ostrich) | FP8, 262K (kiwi) | NVFP4, 1M default (dodo) |
| --- | ---: | ---: | ---: |
| C1 | 29.25 | 28.03 | 29.04 |
| C2 aggregate | 45.26 | 47.67 | — |
| C4 aggregate | 72.55 | 70.11 | — |
| C6 aggregate | 85.03 | 83.96 | 84.15 |

Native MTP with fixed 3/5/7 drafts measured C1 **19.86/20.74/18.63** and C6
**78.29/71.11/59.48** tok/s on emu (two runs, prior image before the grammar-only
fix). DFlash2's 1M profile is about 40% faster at C1 and 7.5% faster at C6 than
the best respective native-MTP depth in that sweep. The original one-draft
row was an untuned control, not the best MTP baseline.
On the current image, target-only measured 10.36/48.46 tok/s at C1/C6;
the five-draft 1M profile is 2.80x/1.74x faster. Seven DFlash2 drafts measured
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

Observed memory admission for the measured sessions (all utilization 0.85):

| Profile / host | Model-load allocation | Available KV pool | Token-equivalent capacity | Request equivalents |
| --- | ---: | ---: | ---: | ---: |
| NVFP4 262K / ostrich | 90.93 GiB | 8.10 GiB | 424,259 | 1.62x |
| FP8 262K / kiwi | 90.93 GiB | 9.86 GiB | 453,597 | 1.73x |
| NVFP4 1M / dodo, qualified runtime | 90.76 GiB | 7.87 GiB | 1,221,641 | 1.17x |
| NVFP4 1M / emu, published defaults | 90.76 GiB | 6.85 GiB | 1,058,756 | 1.01x |

These are hybrid-cache planner token-equivalents, not pure MLA byte division.
Host baseline memory and runtime reservations affect admission; the unequal
pool sizes do **not** isolate cache-format savings. The model-load allocation
also excludes later profiling/graph overhead. Fresh starts can admit less
memory, including the failed 1M attempt described above.

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
diagnostic `GLM53_MEMORY_TRACE=1` and had 1.03 request-equivalents of cache;
it is not another matched 262K cache-format measurement.
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
not the 1M deployment default.
The K7 varied-content blend measured 17.46/62.71 tok/s at C1/C6, with
6/7 and 36/42 structural contracts passing. It also triggered one post-ready
`_kpool_tail_seed_kernel` compilation, unlike the qualified K5 1M run.

See [EVAL.md](EVAL.md) for qualification requirements and
[PROVENANCE.md](PROVENANCE.md) for source pins and adaptation evidence.
