#!/usr/bin/env python3
"""Spark-only BF16 cuBLAS/cuBLASLt graph A/B on observed GLM dense shapes.

Diagnostic synthetic data, not a serving-performance or model-quality claim.
Uses unchanged BF16 weights and chunked FP32 oracle, never weight quantization.
"""
import gc
import json
import platform
import statistics
import subprocess

if platform.machine() != 'aarch64':
    raise SystemExit('Run only on a DGX Spark, not the workstation')

import torch

assert torch.cuda.get_device_capability() == (12, 1)
torch.manual_seed(20260905)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def snapshot():
    return subprocess.check_output([
        'nvidia-smi', '--query-gpu=uuid,name,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_event_reasons.active',
        '--format=csv,noheader'], text=True).strip()


print(json.dumps({'scope': 'synthetic BF16 dense graph diagnostic',
                  'torch': torch.__version__, 'gpu': snapshot()}), flush=True)


@torch.inference_mode()
def case(m, n, k):
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16) * 0.1
    w = torch.randn((n, k), device='cuda', dtype=torch.bfloat16) * 0.1
    expected = torch.empty((m, n), device='cuda', dtype=torch.float32)
    # FP32 without TF32; keep the oracle's temporary weight well below 64 MiB.
    for start in range(0, n, 512):
        expected[:, start:start + 512] = torch.nn.functional.linear(x.float(), w[start:start + 512].float())
    graphs, outputs, errors = {}, {}, {}
    for backend in ('cublas', 'cublaslt'):
        torch.backends.cuda.preferred_blas_library(backend)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(4):
                y = torch.nn.functional.linear(x, w)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = torch.nn.functional.linear(x, w)
        graph.replay()
        torch.cuda.synchronize()
        relative = float((y.float() - expected).norm() / expected.norm())
        cosine = float(torch.nn.functional.cosine_similarity(y.float().flatten(), expected.flatten(), dim=0))
        assert torch.isfinite(y).all() and y.norm() > 0
        assert relative < 0.004 and cosine > 0.99998, (m, n, k, backend, relative, cosine)
        saved = y.clone()
        before = torch.cuda.memory_allocated()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(y, saved)
        assert torch.cuda.memory_allocated() == before
        graphs[backend], outputs[backend] = graph, y
        errors[backend] = {'relative_error': relative, 'cosine': cosine,
                           'graph_replay': True, 'stable_allocation': True}
    samples = {key: [] for key in graphs}
    before_gpu = snapshot()
    # Balanced AB/BA ordering, 20 graph launches per timed sample.
    for repetition in range(8):
        order = ('cublas', 'cublaslt') if repetition % 2 == 0 else ('cublaslt', 'cublas')
        for backend in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(20):
                graphs[backend].replay()
            end.record()
            end.synchronize()
            samples[backend].append(start.elapsed_time(end) * 1000 / 20)
    medians = {key: statistics.median(value) for key, value in samples.items()}
    print(json.dumps({'m': m, 'n': n, 'k': k, 'correctness': errors,
                      'samples_us': samples, 'median_us': medians,
                      'speedup_cublas_over_cublaslt': medians['cublas'] / medians['cublaslt'],
                      'gpu_before': before_gpu, 'gpu_after': snapshot()}), flush=True)


for n, k in ((24896, 4096), (4096, 8192), (8192, 128), (4096, 4096),
             (4096, 2048), (4096, 16384), (4096, 20480), (154880, 4096)):
    for m in (1, 6, 36, 148):
        case(m, n, k)
        gc.collect()
        torch.cuda.empty_cache()
