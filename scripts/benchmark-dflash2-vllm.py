#!/usr/bin/env python3
"""Benchmark DFlash2, native MTP, or target-only with shared serving metrics."""

from __future__ import annotations

import argparse
import http.client
import importlib.util
import json
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from types import ModuleType
from urllib.parse import urlparse


CODE_AGENT_PROMPT = """You are editing an async Python task runner. Fix the cancellation and
exception-handling bugs in this implementation, preserve result ordering, and add precise type
hints. Return only the complete replacement Python module.

```python
import asyncio

async def run_all(factories, limit=8):
    sem = asyncio.Semaphore(limit)
    results = []
    async def one(factory):
        async with sem:
            results.append(await factory())
    tasks = [asyncio.create_task(one(factory)) for factory in factories]
    try:
        await asyncio.gather(*tasks)
    except Exception:
        for task in tasks:
            task.cancel()
    return results
```
"""


def load_contracts() -> ModuleType:
    path = Path(__file__).with_name("test-content-vllm.py")
    spec = importlib.util.spec_from_file_location("glmrt_vllm_contracts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load content contracts from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def post_json(base_url: str, path: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def metrics(base_url: str, timeout: float) -> dict[str, float]:
    with urllib.request.urlopen(
        base_url.rstrip("/") + "/metrics", timeout=timeout
    ) as response:
        text = response.read().decode("utf-8", errors="replace")
    wanted = {
        "vllm:spec_decode_num_drafts_total": "target_verification_passes",
        "vllm:spec_decode_num_draft_tokens_total": "draft_tokens",
        "vllm:spec_decode_num_accepted_tokens_total": "accepted_tokens",
    }
    values = {name: 0.0 for name in wanted.values()}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric_name = line.split("{", 1)[0].split(" ", 1)[0]
        destination = wanted.get(metric_name)
        if destination is not None:
            values[destination] += float(line.rsplit(" ", 1)[-1])
        if metric_name == 'vllm:spec_decode_num_accepted_tokens_per_pos_total':
            position = re.search(r'position="(\d+)"', line)
            if position:
                key = 'position_' + position.group(1)
                values[key] = values.get(key, 0.0) + float(line.rsplit(' ', 1)[-1])
    return values


def render_prompt(
    base_url: str, model: str, prompt: str, close_think: list[int], timeout: float
) -> list[int]:
    rendered = post_json(
        base_url,
        "/v1/chat/completions/render",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": "low",
        },
        timeout,
    )
    return rendered["token_ids"] + close_think


def stream_completion(
    base_url: str,
    model: str,
    prompt_tokens: list[int],
    concurrency: int,
    output_tokens: int,
    timeout: float,
    *,
    force_length: bool,
    seed: int,
    temperature: float,
) -> dict:
    parsed = urlparse(base_url)
    connection_type = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = connection_type(parsed.hostname, parsed.port, timeout=timeout)
    payload = {
        "model": model,
        "prompt": prompt_tokens,
        "add_special_tokens": False,
        "n": concurrency,
        "max_tokens": output_tokens,
        "temperature": temperature,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "cache_prompt": False,
    }
    if force_length:
        payload.update({"min_tokens": output_tokens, "ignore_eos": True})
    if temperature == 0 and concurrency > 1:
        # vLLM rejects n>1 for greedy sampling. A batch of independent
        # prompts with n=1 preserves greedy settings and flattened indices.
        payload['prompt'] = [prompt_tokens] * concurrency
        payload['n'] = 1

    before = metrics(base_url, timeout)
    started_epoch = time.time()
    started = time.perf_counter()
    connection.request(
        "POST",
        parsed.path.rstrip("/") + "/v1/completions",
        body=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    if response.status != 200:
        error = response.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {response.status}: {error}")

    token_times: list[list[float]] = [[] for _ in range(concurrency)]
    content = ["" for _ in range(concurrency)]
    finish_reasons: list[str | None] = [None for _ in range(concurrency)]
    usage = None
    saw_done = False
    while True:
        raw_line = response.readline()
        if not raw_line:
            break
        observed = time.perf_counter()
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == '[DONE]':
            saw_done = True
            break
        if not data:
            continue
        event = json.loads(data)
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        for choice in event.get("choices", []):
            index = choice.get("index")
            if not isinstance(index, int) or not 0 <= index < concurrency:
                continue
            content[index] += choice.get("text") or ""
            if choice.get("finish_reason") is not None:
                finish_reasons[index] = choice["finish_reason"]
            token_ids = choice.get("token_ids")
            if isinstance(token_ids, list):
                token_times[index].extend([observed] * len(token_ids))
    connection.close()
    ended = time.perf_counter()
    after = metrics(base_url, timeout)

    nonempty = [times for times in token_times if times]
    if len(nonempty) != concurrency:
        raise RuntimeError(
            f"only {len(nonempty)} of {concurrency} streams exposed token IDs"
        )
    streamed_tokens = sum(map(len, token_times))
    if usage is None or usage.get('completion_tokens') != streamed_tokens:
        raise RuntimeError(f'Stream/usage token count mismatch: {streamed_tokens}, {usage}')
    if not saw_done or any(reason is None for reason in finish_reasons):
        raise RuntimeError('stream ended without DONE or a finish reason for every sequence')
    first_token = min(times[0] for times in nonempty)
    last_token = max(times[-1] for times in nonempty)
    decode_tokens = sum(max(0, len(times) - 1) for times in token_times)
    batch_window_decode_seconds = last_token - first_token
    sequence_decode_seconds = [times[-1] - times[0] for times in token_times]
    sequence_decode_tokens_per_second = [
        max(0, len(times) - 1) / seconds if seconds > 0 else None
        for times, seconds in zip(
            token_times, sequence_decode_seconds, strict=True
        )
    ]
    resolved_rates = [rate for rate in sequence_decode_tokens_per_second if rate is not None]
    pure_decode_tokens_per_second = sum(resolved_rates) if resolved_rates else None
    drafted = after["draft_tokens"] - before["draft_tokens"]
    accepted = after["accepted_tokens"] - before["accepted_tokens"]
    target_passes = (
        after["target_verification_passes"]
        - before["target_verification_passes"]
    )
    rejected = drafted - accepted
    return {
        "concurrency": concurrency,
        "prompt_tokens": len(prompt_tokens),
        "completion_tokens": sum(len(times) for times in token_times),
        "completion_tokens_by_sequence": [len(times) for times in token_times],
        "decode_tokens": decode_tokens,
        "decode_seconds": batch_window_decode_seconds,
        "decode_tokens_per_second": pure_decode_tokens_per_second,
        "batch_window_decode_seconds": batch_window_decode_seconds,
        "batch_window_decode_tokens_per_second": (
            decode_tokens / batch_window_decode_seconds
            if batch_window_decode_seconds > 0 and resolved_rates else None
        ),
        'decode_timing_resolved': bool(resolved_rates),
        "sequence_decode_tokens_per_second": sequence_decode_tokens_per_second,
        "ttft_ms": (first_token - started) * 1000,
        "request_seconds": ended - started,
        "started_epoch_seconds": started_epoch,
        "sequence_ttft_ms": [
            (times[0] - started) * 1000 for times in token_times
        ],
        "sequence_decode_seconds": sequence_decode_seconds,
        "sequence_finish_offset_seconds": [
            times[-1] - started for times in token_times
        ],
        "draft_tokens": int(drafted),
        "accepted_draft_tokens": int(accepted),
        "rejected_draft_tokens": int(rejected),
        "accepted_draft_rate": accepted / drafted if drafted else 0.0,
        "per_position_acceptance": {
            key.removeprefix('position_'): (value - before.get(key, 0.0)) / target_passes
            for key, value in after.items()
            if key.startswith('position_') and target_passes
        },
        "rejected_draft_rate": rejected / drafted if drafted else 0.0,
        "target_verification_passes": int(target_passes),
        "committed_tokens_per_target_pass": (
            1.0 + accepted / target_passes if target_passes else 0.0
        ),
        "finish_reasons": finish_reasons,
        "content": content,
        "usage": usage,
    }


def summarize_runs(runs: list[dict]) -> dict:
    rates = [run["decode_tokens_per_second"] for run in runs
             if run["decode_tokens_per_second"] is not None]
    batch_rates = [run['batch_window_decode_tokens_per_second'] for run in runs
                   if run['batch_window_decode_tokens_per_second'] is not None]
    acceptances = [run["accepted_draft_rate"] for run in runs]
    target_pass_efficiencies = [
        run["committed_tokens_per_target_pass"] for run in runs
    ]
    return {
        "median_batch_window_decode_tokens_per_second": statistics.median(batch_rates) if batch_rates else None,
        "median_request_tokens_per_second": statistics.median(
            run['completion_tokens'] / run['request_seconds'] for run in runs),
        "median_decode_tokens_per_second": statistics.median(rates) if rates else None,
        "min_decode_tokens_per_second": min(rates) if rates else None,
        "max_decode_tokens_per_second": max(rates) if rates else None,
        'decode_timing_resolved_runs': len(batch_rates),
        "median_accepted_draft_rate": statistics.median(acceptances),
        "median_committed_tokens_per_target_pass": statistics.median(
            target_pass_efficiencies
        ),
        "draft_tokens": sum(run["draft_tokens"] for run in runs),
        "accepted_draft_tokens": sum(run["accepted_draft_tokens"] for run in runs),
        "rejected_draft_tokens": sum(run["rejected_draft_tokens"] for run in runs),
        "target_verification_passes": sum(
            run["target_verification_passes"] for run in runs
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--model", default="vcruz305/GLM-5.3-Flash-EXL3-K2"
    )
    parser.add_argument("--suite", choices=("code-agent", "blend"), required=True)
    parser.add_argument('--draft-tokens', '--dflash-tokens', dest='draft_tokens',
                        type=int, required=True,
                        help='Configured fixed draft depth; zero for target-only')
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 6])
    parser.add_argument('--kv-cache', choices=('nvfp4_ds_mla', 'fp8_ds_mla'), required=True)
    parser.add_argument('--speculative-method', choices=('dflash2', 'mtp', 'none'), default='dflash2')
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--inter-run-seconds",
        type=float,
        default=0.0,
        help="Optional idle interval after each request so benchmark samples do not overlap engine cleanup.",
    )
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.draft_tokens < 0 or (args.speculative_method == 'none') != (args.draft_tokens == 0):
        parser.error('use positive --draft-tokens for speculation, zero for target-only')
    if args.suite == 'blend' and len(args.concurrency) != 1:
        parser.error('blend measures one concurrency per invocation; pass --concurrency 1 or 6')

    close_think = post_json(
        args.base_url,
        "/tokenize",
        {"model": args.model, "prompt": "</think>", "add_special_tokens": False},
        args.timeout,
    )["tokens"]
    report: dict = {
        "schema": "glm53-serving-benchmark.v5",
        "model": args.model,
        "kv_cache": args.kv_cache,
        "speculative_method": args.speculative_method,
        'draft_tokens_per_step': args.draft_tokens,
        'dflash_tokens': args.draft_tokens if args.speculative_method == 'dflash2' else None,
        'mtp_tokens': args.draft_tokens if args.speculative_method == 'mtp' else None,
        "suite": args.suite,
        "primary_metric": "median_batch_window_decode_tokens_per_second",
        "method": (
            "primary aggregate decode is total post-first tokens divided by the "
            "first-any to last-any batch window; end-to-end request throughput "
            "includes TTFT. Legacy median_decode_tokens_per_second is the sum "
            "of per-sequence rates and must not be treated as aggregate serving "
            "capacity when requests are staggered or queued. Acceptance is the matching "
            "Prometheus counter delta; target-pass efficiency is one target "
            "bonus token plus accepted drafts per verification pass"
        ),
    }

    if args.suite == "code-agent":
        prompt_tokens = render_prompt(
            args.base_url, args.model, CODE_AGENT_PROMPT, close_think, args.timeout
        )
        points = []
        for concurrency in args.concurrency:
            for _ in range(args.warmup_runs):
                stream_completion(
                    args.base_url,
                    args.model,
                    prompt_tokens,
                    concurrency,
                    args.output_tokens,
                    args.timeout,
                    force_length=True,
                    seed=args.seed,
                    temperature=args.temperature,
                )
                if args.inter_run_seconds:
                    time.sleep(args.inter_run_seconds)
            runs = []
            for run_index in range(args.runs):
                result = stream_completion(
                    args.base_url,
                    args.model,
                    prompt_tokens,
                    concurrency,
                    args.output_tokens,
                    args.timeout,
                    force_length=True,
                    seed=args.seed,
                    temperature=args.temperature,
                )
                result.pop("content")
                runs.append(result)
                print(
                    f"{args.speculative_method} K{args.draft_tokens} C{concurrency} run "
                    f"{run_index + 1}/{args.runs}: "
                    f"{result['batch_window_decode_tokens_per_second']:.2f} batch-window tok/s, "
                    f"{result['accepted_draft_rate']:.1%} accepted",
                    flush=True,
                )
                if args.inter_run_seconds:
                    time.sleep(args.inter_run_seconds)
            points.append(
                {
                    "concurrency": concurrency,
                    **summarize_runs(runs),
                    "runs": runs,
                }
            )
        report.update(
            {
                "workload": CODE_AGENT_PROMPT,
                "output_tokens_per_sequence": args.output_tokens,
                "warmup_runs_per_point": args.warmup_runs,
                "runs_per_point": args.runs,
                "temperature": args.temperature,
                "inter_run_seconds": args.inter_run_seconds,
                "points": points,
            }
        )
    else:
        contracts = load_contracts()
        cases = []
        all_runs = []
        for case_id, case in contracts.CASES.items():
            prompt_tokens = render_prompt(
                args.base_url, args.model, case.prompt, close_think, args.timeout
            )
            runs = []
            for repeat in range(args.runs):
                result = stream_completion(
                    args.base_url,
                    args.model,
                    prompt_tokens,
                    args.concurrency[0],
                    case.max_tokens,
                    args.timeout,
                    force_length=False,
                    seed=args.seed,
                    temperature=0.0,
                )
                contents = result.pop('content')
                validations = [contracts.validate_case_content(case_id, content)
                               for content in contents]
                for validation, finish in zip(validations, result['finish_reasons'], strict=True):
                    if finish != 'stop':
                        validation['quality_contract_passed'] = False
                        validation['quality_contract_issues'].append(f'incomplete finish: {finish}')
                result['quality_contract_passed'] = all(
                    item['quality_contract_passed'] for item in validations)
                result['quality_by_sequence'] = validations
                result['content'] = contents
                result['content_preview'] = contents[0][:240].replace('\n', '\\n')
                result["repeat"] = repeat + 1
                runs.append(result)
                all_runs.append(result)
                rate = result['batch_window_decode_tokens_per_second']
                rate_text = f'{rate:.2f}' if rate is not None else 'unresolved (single output chunk)'
                print(
                    f"{args.speculative_method} K{args.draft_tokens} C{args.concurrency[0]} {case_id} repeat {repeat + 1}/{args.runs}: "
                    f"{rate_text} batch-window tok/s, "
                    f"{result['accepted_draft_rate']:.1%} accepted, "
                    f"quality={result['quality_contract_passed']}",
                    flush=True,
                )
            cases.append(
                {
                    "case": case_id,
                    "category": case.category,
                    **summarize_runs(runs),
                    "quality_passed": all(
                        run["quality_contract_passed"] for run in runs
                    ),
                    "runs": runs,
                }
            )
        timed_runs = [run for run in all_runs if run['decode_timing_resolved']]
        total_decode_tokens = sum(run["decode_tokens"] for run in timed_runs)
        total_decode_seconds = sum(run["decode_seconds"] for run in timed_runs)
        total_drafts = sum(run["draft_tokens"] for run in all_runs)
        total_accepted = sum(run["accepted_draft_tokens"] for run in all_runs)
        total_rejected = sum(run["rejected_draft_tokens"] for run in all_runs)
        total_target_passes = sum(
            run["target_verification_passes"] for run in all_runs
        )
        report.update(
            {
                "glmrt_standard_seven_case_blend": True,
                'concurrency': args.concurrency[0],
                'batch_scope': 'Each batch contains n completions of one content prompt; seven prompt types across batches',
                "repeats_per_case": args.runs,
                "cases": cases,
                "aggregate": {
                    "decode_tokens_per_second": total_decode_tokens
                    / total_decode_seconds if total_decode_seconds else None,
                    'decode_timing_resolved_runs': len(timed_runs),
                    'decode_timing_unresolved_runs': len(all_runs) - len(timed_runs),
                    "draft_tokens": total_drafts,
                    "accepted_draft_tokens": total_accepted,
                    "rejected_draft_tokens": total_rejected,
                    "accepted_draft_rate": total_accepted / total_drafts
                    if total_drafts
                    else 0.0,
                    "rejected_draft_rate": total_rejected / total_drafts
                    if total_drafts
                    else 0.0,
                    "target_verification_passes": total_target_passes,
                    "committed_tokens_per_target_pass": (
                        1.0 + total_accepted / total_target_passes
                        if total_target_passes
                        else 0.0
                    ),
                    "quality_passed": all(
                        run["quality_contract_passed"] for run in all_runs
                    ),
                },
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
