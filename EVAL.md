# Qualification matrix

Use the MIA image's critical eval coverage as the reference. Run NVFP4 and FP8
with the same target revision, runtime image, sampling settings, request
concurrency, prompts, and measurement definitions. Keep raw JSON receipts,
startup logs, image identity, launch settings, and hardware telemetry.

| Gate | Required evidence |
| --- | --- |
| Runtime and weights | ARM64 build probes; complete checkpoint mapping; real successful generation |
| Per-projection EXL3 | Independent gate/up/down K2/K3 and K3/K4 assignments through the GLM loader; dense reconstruction oracle; decode/prefill tails, graph replay and stable allocation. Synthetic evidence is separate from full mixed-checkpoint qualification. |
| Memory | 0.85 default and never >0.87; measured weights, runtime overhead, cache capacity, host available memory, no swap growth/OOM |
| DFlash2 benefit | Target-only and native-MTP controls; K1–K7 sweep where useful; C1/C6 rates, target-pass cost, total and per-position acceptance |
| Decode | Code-agent and varied content; C1/C2/C4/C6; first-to-last-token decode separately from TTFT; usage token counts, not SSE event counts |
| Prefill | Cold 2K/8K/32K/128K and longer context ladder; server-tokenized sizes, TTFT, prompt tok/s |
| Content quality | Code, math, creative prose, short response, exposition, JSON, multilingual; repeated-word diagnostic reported separately |
| RULER-lite | Single/multi-key retrieval, variable tracking, common-word aggregation at multiple context depths |
| Long context | Cold retrieval/garble/tool ladder to 1,048,576 total tokens where memory permits; exact rendered/tokenized lengths |
| Prefix cache | Identical-prefix reuse and changed-prefix isolation, including recurrent state |
| Tools | Single, complex, parallel, forced, thinking, multi-turn, truncation; Tool Eval Bench standard suite with raw failure details |
| Structured output | Repeated grammar/termination canaries under speculation and concurrency |
| Stability | Mixed prefill/decode rolling C6 requests; cancellation/recovery; post-long-context soak, no corruption or restarts |
| Release | Frozen-image suite, clean startup, warmup/JIT audit, documented measured defaults, published image digest and pull validation |

Six scheduler slots means six active sequences sharing one cache pool. It does
not establish six simultaneous million-token requests. Report both the tested
per-request limit and the total request-equivalent cache capacity.

No borrowed DeepSeek or two-RTX performance result counts as evidence for this
single-Spark GLM image. Report model-behavior failures as failures rather than
modifying the test or scoring away truncation and malformed outputs.

## Reproduce the HTTP checks

Start the desired profile on a Spark using the README. Run one client at a
time per endpoint; a second benchmark would contaminate global speculative
counters and timing. These Python clients do not use local GPUs. Use a fresh
output directory per run and capture the server environment before and after
with `scripts/capture-release-environment.py` **on the serving Spark**.

```bash
GLM_EVAL_URL=http://ostrich:8001
GLM_EVAL_MODEL=vcruz305/GLM-5.3-Flash-EXL3-K2
GLM_EVAL_OUT=$(mktemp -d results/recheck.XXXXXXXX)

python3 scripts/benchmark-dflash2-vllm.py --base-url "$GLM_EVAL_URL" \
  --suite code-agent --draft-tokens 5 --kv-cache nvfp4_ds_mla \
  --concurrency 1 2 4 6 --runs 3 --output "$GLM_EVAL_OUT/code-agent.json"
python3 scripts/benchmark-dflash2-vllm.py --base-url "$GLM_EVAL_URL" \
  --suite blend --draft-tokens 5 --kv-cache nvfp4_ds_mla \
  --concurrency 1 --runs 5 --output "$GLM_EVAL_OUT/content-c1.json"
python3 scripts/benchmark-dflash2-vllm.py --base-url "$GLM_EVAL_URL" \
  --suite blend --draft-tokens 5 --kv-cache nvfp4_ds_mla \
  --concurrency 6 --runs 1 --output "$GLM_EVAL_OUT/content-c6.json"
python3 scripts/benchmark-repetition.py --base-url "$GLM_EVAL_URL" \
  --kv-cache nvfp4_ds_mla --repeats 5 --output "$GLM_EVAL_OUT/orchid.json"
python3 scripts/benchmark-prefill.py --base-url "$GLM_EVAL_URL/v1" \
  --model "$GLM_EVAL_MODEL" --profile nvfp4 --runs 2 \
  --prompt-tokens 2048 8192 32768 131072 \
  --output "$GLM_EVAL_OUT/prefill.json"
python3 scripts/ruler-lite.py --base-url "$GLM_EVAL_URL/v1" \
  --lengths 8192,32768,131072 --output "$GLM_EVAL_OUT/ruler.json"
python3 scripts/verify-issue136-xgrammar-live.py --base-url "$GLM_EVAL_URL/v1" \
  --output "$GLM_EVAL_OUT/grammar.json"
python3 scripts/test-prefix-replay-vllm.py --base-url "$GLM_EVAL_URL" \
  --tokens 32768 --output "$GLM_EVAL_OUT/prefix32k.json"
python3 scripts/test-cancellation-vllm.py --base-url "$GLM_EVAL_URL" \
  --output "$GLM_EVAL_OUT/cancellation.json"
python3 scripts/test-replayssm-stress.py --base-url "$GLM_EVAL_URL" \
  --model "$GLM_EVAL_MODEL" --requests-per-phase 12 --concurrency 6 \
  --target-tokens 32768 --max-tokens 512 --timeout 900 \
  --output "$GLM_EVAL_OUT/stress"
```

For FP8, change both the server profile and client cache labels (`fp8_ds_mla`,
or `--profile fp8`). Keep the server capacities matched as shown in the README.
Benchmark flags label the actual server; they do not reconfigure it. Native
MTP controls need `SPECULATIVE_METHOD=mtp ADAPTIVE_MTP=0 MTP_TOKENS=3` at
startup and `--speculative-method mtp --draft-tokens 3` in the client; repeat
for 5/7. Target-only uses `SPECULATIVE_METHOD=none` and client
`--speculative-method none --draft-tokens 0`.

On the 1M NVFP4 profile, additionally run `test-multi-needle-vllm.py` at
`--tokens 131072` and `524288`, then `test-prefix-replay-vllm.py` at
`--tokens 1048320` (each with `--base-url` and a distinct `--output`). The latter
reserves 256 output tokens within the 1,048,576 limit and performs four
retrieval/replay/isolation passes. Repeat grammar/cancellation or rolling
stress afterward. Cold near-1M prefill takes about 22 minutes; a quiet client
is not evidence that the job has stopped.

Tool Eval Bench requires the separate `../tool-eval-bench` checkout and its
project virtual environment. Use its CLI, standard 69-scenario suite, serial
and C6, temperature 0, seed 20260905, low reasoning, and 2,048 output tokens.
Leave its fixed reference date unchanged; archive both SQLite history and
the full-trace Markdown report. See the raw run `config` objects for exact
settings. Do not use a current-date override against the fixed-date scorer.
