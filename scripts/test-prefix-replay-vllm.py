#!/usr/bin/env python3
"""Cold, replay, changed-tail isolation, then replay the original chat prompt."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path


def load_long_context():
    path = Path(__file__).with_name("test-multi-needle-vllm.py")
    spec = importlib.util.spec_from_file_location("glm53_long_context", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prefix_hits(base_url: str) -> float:
    with urllib.request.urlopen(base_url.rstrip("/") + "/metrics", timeout=300) as response:
        body = response.read().decode("utf-8", errors="replace")
    total = 0.0
    for line in body.splitlines():
        if line.startswith("vllm:prefix_cache_hits_total{"):
            total += float(line.rsplit(" ", 1)[-1])
    return total


def passes_qualify(passes: list[dict]) -> bool:
    """Correct answers alone do not prove cached-state isolation."""
    return (len(passes) == 4 and all(item['passed'] for item in passes)
            and passes[0]['prefix_cache_hit_delta'] == 0
            and all(item['prefix_cache_hit_delta'] > 0 for item in passes[1:]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--model", default="vcruz305/GLM-5.3-Flash-EXL3-K2"
    )
    parser.add_argument("--tokens", type=int, default=128_000)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    long_context = load_long_context()
    nonce = f"prefix-replay-{uuid.uuid4().hex}"
    cache_salt = uuid.uuid4().hex
    prompt, _, _, _ = long_context.render_exact_prompt(
        args.base_url, args.model, args.tokens, nonce
    )
    original_facts = long_context.FACTS
    # Change the last fact only: most of the prefix remains byte-identical,
    # while scoring must reject state leaked from the previous completion.
    long_context.FACTS = original_facts[:-1] + (
        (original_facts[-1][0], original_facts[-1][1], "amethyst-3926"),)
    changed_prompt, _, _, _ = long_context.render_exact_prompt(
        args.base_url, args.model, args.tokens, nonce)
    changed_facts = long_context.FACTS
    long_context.FACTS = original_facts
    before_hits = prefix_hits(args.base_url)
    passes = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'schema': 'glm53-prefix-replay.v4', 'model': args.model, 'base_url': args.base_url,
        'passed': False, 'complete': False, 'target_prompt_tokens': args.tokens,
        'cache_salt': cache_salt, 'fixture_nonce': nonce,
        'original_prompt_ids_sha256': hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
        'changed_prompt_ids_sha256': hashlib.sha256(json.dumps(changed_prompt).encode()).hexdigest(),
        'prefix_cache_hits_before': before_hits, 'passes': passes,
        'scope': 'Exact answers, zero-hit cold retrieval, then positive cache hits on each of three replay/isolation passes',
    }
    for pass_index, (name, tokens, facts) in enumerate((
        ('cold', prompt, original_facts), ('replay', prompt, original_facts),
        ('changed-tail', changed_prompt, changed_facts),
        ('original-after-changed-tail', prompt, original_facts),
    )):
        pass_hits_before = prefix_hits(args.base_url)
        output, usage, ttft, elapsed, finish_reason = long_context.stream_completion(
            args.base_url,
            {
                "model": args.model,
                "prompt": tokens,
                "add_special_tokens": False,
                "max_tokens": 256,
                "temperature": 0,
                "seed": 20260901,
                "cache_salt": cache_salt,
                'stream': True, 'stream_options': {'include_usage': True},
            },
            args.timeout,
        )
        expected_lines = [f"{key}={value}" for _, key, value in facts]
        actual_lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
        usage = usage or {}
        pass_hits_after = prefix_hits(args.base_url)
        passes.append(
            {
                "pass": pass_index + 1,
                "name": name,
                "passed": actual_lines == expected_lines and finish_reason == 'stop'
                          and usage.get('prompt_tokens') == args.tokens,
                "expected_lines": expected_lines,
                "finish_reason": finish_reason,
                "request_seconds": elapsed,
                'ttft_seconds': ttft,
                "usage": usage,
                "output": output,
                "prefix_cache_hit_delta": pass_hits_after - pass_hits_before,
            }
        )
        report['prefix_cache_hits_after'] = pass_hits_after
        report['prefix_cache_hit_delta'] = report['prefix_cache_hits_after'] - before_hits
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps({'pass': name, 'passed': passes[-1]['passed'],
                          'prefix_cache_hit_delta': passes[-1]['prefix_cache_hit_delta'],
                          'ttft_seconds': ttft, 'request_seconds': elapsed}), flush=True)
    after_hits = prefix_hits(args.base_url)
    report.update({
        "passed": passes_qualify(passes),
        'complete': True,
        "target_prompt_tokens": args.tokens,
        "cache_salt": cache_salt,
        "prefix_cache_hits_before": before_hits,
        "prefix_cache_hits_after": after_hits,
        "prefix_cache_hit_delta": after_hits - before_hits,
        "passes": passes,
        "high_recycled_page_rationale": (
            "The identical second pass exercises the production allocator/cache replay "
            "path that previously exposed 32-bit pool-offset overflow; focused B12x "
            "tests separately force page IDs past the exact 2^31/stride boundary."
        ),
    })
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
