#!/usr/bin/env python3
"""GPU qualification of the image's GLM EXL3 projection adapter.

Run only on a Spark, inside the built image. The oracle reconstructs each
projection independently using the pinned B12x test reference, then evaluates
the dense rotated MoE. This is synthetic kernel/adapter evidence, not a model
quality result for the still-unavailable mixed-bitrate GLM checkpoint.
"""
import argparse
import json
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, '/opt/b12x')
import torch
from b12x.moe import fused_moe
from tests.moe.test_fused_moe_trellis import (
    _decode_lane, _reconstruct_native, _reference_full_rotation_decoded,
)
from vllm.model_executor.layers.quantization.exl3 import Exl3Config, Exl3MoEMethod


def reconstruct_vectorized(trellis):
    """Same independent NumPy MCG oracle, vectorized over all 16x16 tiles."""
    native = trellis.detach().cpu().numpy()
    kt, nt, width = native.shape
    bits = width // 16
    packed = native.view(np.uint16).reshape(kt, nt, 8 * bits, 2)
    words = packed[..., 0].astype(np.uint32) | (packed[..., 1].astype(np.uint32) << np.uint32(16))
    tiles = np.empty((kt, nt, 16, 16), dtype=np.float16)
    for lane in range(32):
        values = _decode_lane(words, lane, bits)
        row0 = lane % 4 * 2
        rows = (row0, row0 + 1, row0 + 8, row0 + 9)
        col0, parity = lane // 8, (lane >> 2) & 1
        for weight in range(8):
            tiles[:, :, rows[weight % 4], 2 * (col0 if weight < 4 else col0 + 4) + parity] = values[..., weight]
    return torch.from_numpy(tiles.transpose(0, 2, 1, 3).reshape(kt * 16, nt * 16))


@torch.inference_mode()
def qualify(bits, experts=4, glm_geometry=False):
    torch.manual_seed(20260905 + bits)
    device = torch.device('cuda:0')
    hidden, intermediate = (4096, 2048) if glm_geometry else (128, 128)
    topk = 8 if glm_geometry else 2
    # No projection shares the same membership; gate/up counts are unequal.
    rates = ((bits, bits + 1, bits), (bits + 1, bits, bits + 1),
             (bits, bits + 1, bits + 1), (bits, bits, bits + 1))
    if experts == 288:
        rates = tuple((bits if e < 287 else bits + 1,
                       bits + 1 if e else bits, bits + e % 2) for e in range(experts))
    storage = {}
    raw = {}
    templates = {}
    for expert, triple in enumerate(rates):
        for projection, shard, rate in zip(
                ('gate_proj', 'up_proj', 'down_proj'), ('w1', 'w3', 'w2'), triple):
            name = f'model.language_model.layers.0.mlp.experts.{expert}.{projection}'
            storage[name] = {
                'bits_per_weight': rate, 'quant_format': 'exl3',
                'stored_tensors': {f'{name}.{s}': {} for s in ('suh', 'svh', 'trellis', 'mcg')},
            }
            k, n = (intermediate, hidden) if shard == 'w2' else (hidden, intermediate)
            if glm_geometry:
                # Shared native payloads bound independent oracle cost; unique
                # per-expert rotation scales still expose route-map mistakes.
                if (shard, rate) not in templates:
                    templates[shard, rate] = torch.randint(
                        -32768, 32767, (k // 16, n // 16, 16 * rate), dtype=torch.int16, device=device)
                raw[expert, shard] = templates[shard, rate]
            else:
                raw[expert, shard] = torch.randint(
                    -32768, 32767, (k // 16, n // 16, 16 * rate), dtype=torch.int16, device=device)
    config = Exl3Config(bits=bits + 0.5, codebook='mcg', tensor_storage=storage)
    config._configure_standard_fused_moe(SimpleNamespace(model_type='glm5_next', num_hidden_layers=1))
    assert config.standard_layer_projection_bitrates('model.language_model.layers.0', experts) == rates

    def scales(width):
        return (0.875 + 0.25 * torch.rand((experts, width), device=device)).half()

    gate_suh, up_suh, gate_svh, up_svh, down_suh, down_svh = [
        scales(width) for width in (hidden, hidden, intermediate, intermediate, intermediate, hidden)]
    layer = SimpleNamespace(
        local_num_experts=experts, exl3_hidden_size=hidden,
        exl3_intermediate_size_per_partition=intermediate, exl3_projection_bitrates=rates,
        exl3_params_dtype=torch.bfloat16, activation=SimpleNamespace(value='silu'),
        layer_name='model.language_model.layers.0.mlp.experts', use_ep=False,
    )
    for prefix, shards in (('w13', ('w1', 'w3')), ('w2', ('w2',))):
        for suffix in ('suh', 'svh', 'trellis', 'mcg', 'mul1'):
            param = SimpleNamespace(exl3_tensors={}, exl3_backing=None, exl3_shard_ids=shards)
            if suffix == 'trellis':
                param.exl3_tensors = {k: v for k, v in raw.items() if k[1] in shards}
            setattr(layer, f'{prefix}_{suffix}', param)
    layer.w13_suh.exl3_backing = torch.stack((gate_suh, up_suh))
    layer.w13_svh.exl3_backing = torch.stack((gate_svh, up_svh))
    layer.w2_suh.exl3_backing = down_suh
    layer.w2_svh.exl3_backing = down_svh
    method = object.__new__(Exl3MoEMethod)
    method.quant_config = config
    method._prepare_projection_mixed_weights(layer)
    weights = layer.exl3_trellis_weights
    prepared = weights.representation_for('w4a16')
    assert prepared.tier_bits == (bits, bits + 1)
    assert prepared.gate_counts != prepared.up_counts
    if glm_geometry:
        for rate in (bits, bits + 1):
            tiny = templates['w1', rate][:2, :2].contiguous()
            assert torch.equal(reconstruct_vectorized(tiny), _reconstruct_native(tiny))
        decoded_templates = {key: reconstruct_vectorized(value).to(device)
                             for key, value in templates.items()}
        decoded = [torch.stack([decoded_templates[s, rates[e][p]] for e in range(experts)])
                   for p, s in enumerate(('w1', 'w3', 'w2'))]
    else:
        decoded = [torch.stack([_reconstruct_native(raw[e, s]) for e in range(experts)]).to(device)
                   for s in ('w1', 'w3', 'w2')]
    rotations = torch.cat((gate_svh, up_svh, down_suh), dim=1)
    results = []
    for capacity in (8, 128):
        plan = fused_moe.plan(fused_moe.Caps(
            max_tokens=capacity, num_topk=topk, route_num_experts=0, device=device,
            weight_plan=weights.plan, quant_mode='w4a16', w4a16_block_size_m=8,
            mixed_trellis_route_id_dtypes=(torch.int32,),
            mixed_trellis_broadcast_suh=(False,), mixed_trellis_broadcast_svh=(False,),
            full_rotation_output_dtype=torch.float32,
        ))
        spec = plan.scratch_specs()[0]
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        for rows in ((1, 6, 8) if capacity == 8 else (36, 65, 128)):
            x = (torch.randn((rows, hidden), device=device) * 1e-3).bfloat16()
            ids = torch.arange(rows * topk, device=device, dtype=torch.int32).reshape(rows, topk) % experts
            if experts == 288:
                boundary_ids = torch.tensor([255, 256, 287, 0, 254, 286, 1, 257], dtype=torch.int32, device=device)
                ids = boundary_ids[ids.long() % boundary_ids.numel()]
            routing = torch.linspace(1, 0.25, topk, device=device)
            routing = (routing / routing.sum()).expand(rows, topk).contiguous()
            binding = fused_moe.bind(plan, scratch=scratch, a=x, experts=weights,
                                     topk_weights=routing, topk_ids=ids)
            actual = binding.run().clone()
            expected = _reference_full_rotation_decoded(
                x, ids, routing, *decoded, gate_suh, up_suh, rotations, down_svh)
            torch.cuda.synchronize()
            assert torch.isfinite(actual).all() and actual.norm() > 0
            relative = float((actual - expected).norm() / expected.norm())
            cosine = float(torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0))
            assert relative < 0.02, (bits, rows, relative)
            assert cosine > 0.999, (bits, rows, cosine)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = binding.run()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(captured, actual)
            before = torch.cuda.memory_allocated()
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == before
            result = dict(tiers=[bits, bits + 1], experts=experts, rows=rows, capacity=capacity,
                          hidden=hidden, intermediate=intermediate, topk=topk,
                          relative_error=relative, cosine=cosine, graph_replay=True,
                          stable_allocation=True)
            print(json.dumps(result), flush=True)
            results.append(result)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--glm-geometry', action='store_true')
    parser.add_argument('--bits', type=int, nargs='+', choices=(2, 3, 4, 5), default=[2, 3])
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (12, 1), 'Run this qualification on a Spark'
    print(json.dumps({'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__}), flush=True)
    for experts in ((288,) if args.glm_geometry else (4, 288)):
        for bits in args.bits:
            qualify(bits, experts, args.glm_geometry)
