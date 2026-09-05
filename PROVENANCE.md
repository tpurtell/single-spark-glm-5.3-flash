# Inputs and adaptation evidence

| Component | Pinned identity |
| --- | --- |
| Official GLM ARM64 runtime | `vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce` |
| vLLM in runtime | `0.1.dev20051+g487ecf187` |
| Torch / CUDA / CUTLASS DSL | `2.13.0+cu130` / CUDA 13 / `4.6.2` |
| Brandon recipe source | `4714ae05d85630674e4c4d283be4541bf92e1ff0` in `../brandon-glm-5.3-flash/recipe` |
| B12x fork | `tpurtell/sparkinfer-glmrt@c90cd0080274b90da4721f8ab7536bca29cae720` (remote HEAD checked 2026-09-05) |
| Streaming loader | `instanttensor==0.1.9` (native `copy=True`, plus MIA-derived pressure-triggered allocator trimming) |
| EXL3 and B12x Python adapter source | `ghcr.io/tpurtell/deepseek-v4-flash-0731-exl3-k2-spark@sha256:86c8c1054f9c24454949e37031ce6165c007963aa0c0ef30fa884f6d4170af32` |
| Upstream DFlash2 delta | `vllm-project/vllm@b389ac29465b33f9e9c534df221ea3c129e9793f` |
| Target | `vcruz305/GLM-5.3-Flash-EXL3-K2@ca0bcdae265f7df1e346c57a2b53b8b8f632ee0b` |
| Draft | `incoai/GLM-5.3-Flash-DFlash2@bf582e4eacc1810f76656d1811693ff6c6737d2a` |

The EXL3 source stage is amd64 and supplies four Python files only; no command
executes in that stage, and no amd64 binary enters the ARM64 runtime.
The official ARM64 runtime's vLLM, Torch, and CUTLASS versions were checked
inside a container on ostrich and match Brandon's runtime versions.

The MIA DeepSeek recipe is the quality/evaluation reference. The newer
`ds4fv-vllm-28-sm12x` recipe is an implementation reference, not a quality baseline:
its README records a model-quality regression relative to the MIA derivative.

## K2 manifest repair

The pinned K2 snapshot contains 120 shards totalling 97,728,721,536 bytes,
150,226 tensors, and 37,152 routed projections. Header extents and index
coverage are complete. Its external EXL3 manifest describes only 4,180
projections, leaving 32,972 absent. This is a metadata issue in the pinned
snapshot, not an incomplete transfer.

`scripts/prepare-model.py` reconstructs a complete external manifest from the
shard headers and stored four-byte MCG markers. It derives each projection's
integral bitrate from its actual packed size and rotation dimensions, checks
every existing manifest record for exact agreement, and creates a derived
view with symlinks to the original tensors. It never requantizes weights or
edits the downloaded snapshot. Full tensor-content hashes remain a separate
verification scope from this header/index audit.

## Single-Spark integration fixes

The recipe carries narrow patches on top of the pinned B12x fork:

- Admit GLM's 2176-wide physical selection table in FP8/NVFP4 prefill and
  size graph-stable logical-to-physical metadata from the model-owned buffer.
  All 2051 selected candidates remain present; padding is masked.
- Restore the uniform Trellis FP16 prepared-weight contract while keeping
  projection-mixed live activations BF16, following the MIA integration.
- Reserve enough direct-route workspace for small mixed-Trellis expert sets.
- Extend the in-process mixed-Trellis descriptor from eight to nine local
  index bits so one projection tier can hold GLM's 288 experts. Both descriptor
  builders and GPU consumers use the new encoding; no checkpoint is rewritten.
- Bound GLM's shared indexer gather capacity by the scheduler's maximum
  sequence count, consistently in the actual allocation and chunk planner.
  The inherited 40-context constant allocated 5.16 GiB at 1M context on C6.
- Profile B12x's actual sparse MLA workspace instead of an unreachable dense
  context up-projection. Keep vLLM's memory accounting and graph reservation.
- Correct the GLM ReplaySSM convolution window to K+1 tokens independently of
  its one-column compact state table. In mixed/eager batches, the old column
  count left saved convolution history stale; pure-decode graph padding hid
  this from short throughput tests. The focused mixed-length GPU oracle and
  one-column negative control are in
  `results/20260905-replayssm-conv-window-oracle-ostrich.jsonl`. The corrected
  image then passed all 36 rolling C6 requests with no loops or truncation;
  the subsequent grammar retest passed 145/145 on both NVFP4 and FP8.
- Carry vLLM [#52805](https://github.com/vllm-project/vllm/pull/52805), already
  present in MIA issue #136, onto GLM's still-unfixed XGrammar methods. Accept
  and validation batches stop at grammar termination, later advances are
  no-ops, and reset clears the cached termination flag. The exact pre-patch
  GLM file SHA256 is `3fd606dc2b8e950fe9b49f28cf1c030be78beaaf7c78b457b5942a0909d3457f`.
  CPU tests extracted from the installed methods pass 7/7; the uncorrected
  image fails three of the same tests. Image `94711456c8e9…` then passed all
  145 live grammar/termination cases on both NVFP4 and FP8, including the
  formerly failing ignore-EOS request. Receipts:
  `results/20260905-final-k5-{nvfp4-ostrich,fp8-kiwi}-grammar.json`.

`scripts/qualify-wide-mla.py` compares static/dynamic NVFP4 and FP8 to a dense
quantized reference, including dominant final-three candidates, empty rows,
high recycled page IDs above 2 GiB byte offsets, and graph replay. Raw initial
evidence is in `results/20260905-wide-mla-kiwi.jsonl`; this is kernel evidence,
not an end-to-end quality or throughput claim.

`scripts/qualify-projection-mixed.py` exercises the GLM adapter with independent
gate/up/down rates, a dense reconstructed-weight oracle, 4- and 288-expert
layouts, and decode/prefill graph replay. The 2026-09-05 full-geometry run on emu
passed 12 cases at H4096/I2048, 288 experts, top-k 8, M1/6/8/36/65/128, for
K2/K3 and K3/K4. Packed payloads are shared within each projection/bitrate test
tier; unique expert rotations and boundary routes exercise the descriptor map.
Its NumPy oracle is vectorized across tiles and checked against the original
scalar-tile reconstruction. See `results/20260905-projection-glm-geometry-emu.log`.
A future full mixed GLM checkpoint still needs its own loading and model-level
quality qualification; these synthetic checks do not substitute for that.

## Credits and licenses

Recipe code is Apache-2.0, inherited with attribution to the Brandon recipe,
tpurtell's MIA derivative, B12x/SparkInfer contributors, and vLLM contributors.
Thanks to Z.ai for GLM, vcruz305 for K2, and Inco AI/Z-Lab for DFlash2.
Checkpoint licenses apply independently: the target is MIT; the draft's
[model card](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) specifies
CC BY-NC-ND 4.0 for research/evaluation. Weights are not bundled in the image.
