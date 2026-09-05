#!/usr/bin/env python3
"""Spark GPU oracle for compact-state versus speculative convolution width.

The negative control deliberately uses one compact state column as the token
capacity. Its first outputs can look correct while its saved history is stale.
"""
import json
import platform

if platform.machine() != 'aarch64':
    raise SystemExit('Run on a Spark only')
import torch
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

assert torch.cuda.get_device_capability() == (12, 1)
torch.manual_seed(20260905)


@torch.inference_mode()
def case(drafts, dim):
    maximum, width = drafts + 1, 4
    lengths = [1, maximum, 3]
    ids = torch.tensor([[1, -1], [3, -1], [4, -1]], device='cuda', dtype=torch.int32)[:, 0]
    starts = torch.tensor([0, 1, 1 + maximum, 4 + maximum], device='cuda', dtype=torch.int32)
    accepted = torch.tensor([1, 3, maximum], device='cuda', dtype=torch.int32)
    x = torch.randn((sum(lengths), dim), device='cuda', dtype=torch.bfloat16) * 0.1
    state = torch.randn((5, dim, width - 1 + drafts), device='cuda', dtype=torch.bfloat16) * 0.1
    weight = torch.randn((dim, width), device='cuda', dtype=torch.bfloat16) * 0.1
    expected = torch.empty_like(x)
    expected_state = state.clone()
    begin = 0
    for slot, count, offset in zip((1, 3, 4), lengths, (0, 2, maximum - 1)):
        history = torch.cat((state[slot, :, offset:offset + width - 1], x[begin:begin + count].T), dim=1)
        for token in range(count):
            value = (history[:, token:token + width].float() * weight.float()).sum(1)
            expected[begin + token] = torch.nn.functional.silu(value).bfloat16()
        expected_state[slot, :, :width - 2 + count] = history[:, 1:]
        begin += count

    def run(capacity, current_state, out):
        causal_conv1d_update(x, current_state, weight, activation='silu',
                            conv_state_indices=ids, num_accepted_tokens=accepted,
                            query_start_loc=starts, max_query_len=capacity, out=out)

    good_state, bad_state = state.clone(), state.clone()
    good, bad = torch.empty_like(x), torch.empty_like(x)
    run(maximum, good_state, good)
    run(1, bad_state, bad)
    torch.cuda.synchronize()
    relative = float((good.float() - expected.float()).norm() / expected.float().norm())
    assert relative < 0.004, relative
    assert torch.equal(good_state, expected_state)
    wrong_state_elements = int((bad_state != expected_state).sum())
    assert wrong_state_elements > 0, 'negative control did not reproduce stale history'
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        good_state.copy_(state)
        run(maximum, good_state, good)
    graph.replay()
    torch.cuda.synchronize()
    saved = good.clone()
    allocated = torch.cuda.memory_allocated()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(good, saved) and torch.equal(good_state, expected_state)
    assert torch.cuda.memory_allocated() == allocated
    print(json.dumps({'drafts': drafts, 'dim': dim, 'lengths': lengths,
                      'state_index_stride': ids.stride(0), 'relative_output_error': relative,
                      'correct_saved_state': True, 'negative_control_stale_elements': wrong_state_elements,
                      'graph_replay': True, 'stable_allocation': True}), flush=True)


for drafts in (5, 7):
    for dim in (128, 24576):
        case(drafts, dim)
