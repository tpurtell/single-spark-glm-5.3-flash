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
