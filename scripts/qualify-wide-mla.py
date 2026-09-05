#!/usr/bin/env python3
"""Validate all 2051 GLM candidates at width 2176, including high pool IDs."""
import json
import sys

sys.path.insert(0, '/opt/b12x')
import torch
from b12x.attention._shared.mla.prefill import run_unified_prefill
from b12x.attention._shared.mla.traits import ScaleFormat
from b12x.attention._shared.mla.reference import pack_mla_kv_cache_reference, unpack_mla_kv_cache_reference
from tests._reference.helpers import pack_dsv4_nvfp4_record_reference, dequantize_nvfp4_mla_nope


@torch.inference_mode()
def qualify(high_pid, cache_format):
    torch.manual_seed(20260905)
    device = torch.device('cuda:0')
    page_size, width, live = 64, 2176, 2051
    record_bytes = 656 if cache_format == 'fp8' else 368
    stride = page_size * record_bytes + 8192  # packed GLM slab padding
    first_page = (2**31 // stride + 1) if high_pid else 0
    blocks = first_page + 35
    backing = torch.empty(blocks * stride, dtype=torch.uint8, device=device)
    cache = torch.as_strided(backing, (blocks, page_size, 1, record_bytes),
                            (stride, record_bytes, record_bytes, 1))
    values = torch.randn((35 * page_size, 512), dtype=torch.bfloat16, device=device) / 4
    # Final three candidates dominate the softmax. Dropping them MUST fail.
    values[:live - 3, 0] = -4
    values[live - 3:live, 0] = 4
    if cache_format == 'fp8':
        records = pack_mla_kv_cache_reference(values, torch.zeros((values.shape[0], 64), dtype=torch.bfloat16, device=device))[:, 0]
        dequant = unpack_mla_kv_cache_reference(records)[:, 0, :512]
    else:
        packed = pack_dsv4_nvfp4_record_reference(values)
        records = torch.cat((packed[:, :304], torch.zeros((values.shape[0], 64), dtype=torch.uint8, device=device)), dim=1)
        dequant, _ = dequantize_nvfp4_mla_nope(records)
        if cache_format == 'nvfp4-dynamic':
            outer = torch.linspace(0.75, 1.25, values.shape[0], device=device)
            records[:, 292:296] = outer.view(torch.uint8).reshape(-1, 4)
            dequant *= outer[:, None]
    cache[first_page:].copy_(records.view(35, page_size, 1, record_bytes))
    q = torch.zeros((3, 64, 576), dtype=torch.bfloat16, device=device)
    q[:, :, :512] = torch.randn((3, 64, 512), dtype=torch.bfloat16, device=device) / 4
    q[0, :, 0] = 64
    indices = torch.full((3, width), -1, dtype=torch.int32, device=device)
    indices[:, :live] = first_page * page_size + torch.arange(live, dtype=torch.int32, device=device)
    lengths = torch.tensor([live, 1537, 0], dtype=torch.int32, device=device)
    output = torch.empty((3, 64, 512), dtype=torch.bfloat16, device=device)
    lse = torch.empty((3, 64), dtype=torch.float32, device=device)

    def run():
        return run_unified_prefill(
            q=q, kv_cache=cache, topk_indices=indices, topk_length=lengths,
            sm_scale=256**-0.5, page_block_size=page_size,
            scale_format=ScaleFormat.ARBITRARY_FP32 if cache_format == 'fp8' else ScaleFormat.NVFP4_E4M3,
            fp8_rope=cache_format != 'fp8', latent_scale_per_token=cache_format == 'nvfp4-dynamic',
            output=output, lse_out=lse)

    run()
    torch.cuda.synchronize()
    metrics = []
    for row, count in enumerate((live, 1537)):
        logits = q[row, :, :512].float() @ dequant[:count].T * 256**-0.5
        expected = logits.softmax(-1) @ dequant[:count]
        actual = output[row].float()
        cosine = float(torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0))
        max_abs = float((actual - expected).abs().max())
        assert torch.isfinite(actual).all() and actual.norm() > 0
        assert cosine > 0.995 and max_abs < 0.03, (high_pid, row, cosine, max_abs)
        if row == 0:
            assert logits.softmax(-1)[:, -3:].sum(-1).min() > 0.99
        metrics.append(dict(row=row, live=count, cosine=cosine, max_abs=max_abs))
    assert output[2].count_nonzero() == 0
    assert torch.isneginf(lse[2]).all()
    eager = output.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(eager, output)
    print(json.dumps(dict(cache_format=cache_format, high_pid=high_pid, first_page=first_page,
                          byte_offset=first_page * stride, width=width,
                          metrics=metrics, zero_length=True, graph_replay=True)), flush=True)


if __name__ == '__main__':
    assert torch.cuda.get_device_capability() == (12, 1), 'Run this qualification on a Spark'
    for cache_format in ('nvfp4-static', 'nvfp4-dynamic', 'fp8'):
        for high_pid in (False, True):
            qualify(high_pid, cache_format)
