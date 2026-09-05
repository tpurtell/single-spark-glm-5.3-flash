# GLM-5.3 Flash EXL3 K2 + DFlash2 on one DGX Spark

Work in progress: the ARM64 image is built and serving on the Sparks. Release
qualification and container publication are unfinished. Initial measurements
are in [the qualification log](results/QUALIFICATION-20260905.md); they are not
frozen-release guarantees.

Target: [vcruz305/GLM-5.3-Flash-EXL3-K2](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2).
Draft: [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2).
The production profile will use NVFP4 MLA cache, one GB10 GPU, six scheduler
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

## Development commands

Run Docker and GPU work on a DGX Spark:

```bash
./build.sh
./download.sh  # omit when the pinned snapshots are already installed
./start.sh
docker logs -f glm53-spark
```

The development image is `ghcr.io/tpurtell/single-spark-glm-5.3-flash:dev`;
it is not published yet. The API port is 8001. Models use the standard
`$HF_HOME/hub` cache, defaulting to `$HOME/.cache/huggingface/hub`.

The default is text-only, NVFP4 MLA, **1,048,576 tokens**, DFlash2 with five
draft tokens, compact recurrent rollback, **prefix caching on**, six scheduler
slots, and **0.85 utilization**. Batch and EXL3 prefill capacities are 512.
This does not imply six concurrent million-token requests: the measured shared
cache capacity was 1.17 request-equivalents at the 1M limit.

The current image passed all four near-1M retrieval/replay/isolation requests,
then C6 rolling stress and cancellation/recovery. Cold TTFT was 22.5 minutes;
identical replay was 28.8 seconds. No post-ready JIT compilations or engine
restarts were observed in that run; host swap usage did not grow between the
before/after snapshots. Full tool-quality scoring and publication remain open.

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

| Serving/quality check | NVFP4 K5 | FP8 K5 |
| --- | --- | --- |
| Tools/structured-output/termination canaries | 145/145 | 145/145 |
| Strict 32K cold/replay/changed-tail/original | 4/4, cache hits recorded | 4/4, cache hits recorded |
| Cancellation and exact recovery | 12 disconnects, 12 recoveries | 12 disconnects, 12 recoveries |
| RULER-lite, 8K/32K/128K | 12/12 | 12/12 |
| Seven-content contracts, C1 | 6/7 | 6/7 |
| Seven-content contracts, C6 | 29/42 | 34/42 |
| Seven-content blend decode, C1 / C6 | 19.20 / 66.64 tok/s | 17.94 / 64.41 tok/s |
| Tool Eval Bench, C6 and serial | Running | Running |

These quality scores are scoped: RULER-lite checks gold inclusion and clean
termination in synthetic tasks, not official RULER. Some correct common-word
sets were accompanied by incorrect unrequested counts. Content contracts are
structural, not execution-based code tests. Failures include the inherited
math input-literal check, fable word counts, six NVFP4 C6 exposition truncations,
and one FP8 C6 code truncation. Full outputs and failures are retained.

Prefix reuse is coarse with the current hybrid cache layout and speculative
lookback: K5 NVFP4 uses 15,360-token allocator blocks; K7 uses 18,432. A 32K K7
probe answered correctly but had zero hits; the 64K probe passed all four
retrieval/isolation checks with 110,592 hits. Do not infer short-prefix reuse
merely from `--enable-prefix-caching`. Seven drafts remain a tuning profile,
not the 1M deployment default.

See [EVAL.md](EVAL.md) for qualification requirements and
[PROVENANCE.md](PROVENANCE.md) for source pins and adaptation evidence.
