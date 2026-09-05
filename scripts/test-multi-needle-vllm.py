#!/usr/bin/env python3
"""Cold multi-needle retrieval qualification at an exact prompt-token count."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.request
import uuid
from pathlib import Path


DEFAULT_MODEL = "vcruz305/GLM-5.3-Flash-EXL3-K2"
FACTS = (
    (0.05, "CINDER-05", "azurite-4831"),
    (0.25, "JUNIPER-25", "topaz-7614"),
    (0.50, "LANTERN-50", "cobalt-2097"),
    (0.75, "MARBLE-75", "saffron-6382"),
    (0.95, "ORBIT-95", "willow-1459"),
    (0.99, "QUARTZ-99", "indigo-8726"),
)


def post(base_url: str, path: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def stream_completion(
    base_url: str, payload: dict, timeout: float
) -> tuple[str, dict | None, float, float, str | None]:
    """Return output, usage, TTFT, and total request time from an SSE stream."""
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    output_parts: list[str] = []
    usage: dict | None = None
    finish_reason: str | None = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage") is not None:
                usage = event["usage"]
            for choice in event.get("choices") or ():
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                text = choice.get("text") or ""
                if text:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    output_parts.append(text)
    finished = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("stream completed without an output token")
    return (
        "".join(output_parts),
        usage,
        first_token_at - started,
        finished - started,
        finish_reason,
    )


def tokenize(base_url: str, model: str, prompt: str) -> list[int]:
    return post(
        base_url, "/tokenize", {"model": model, "prompt": prompt}, 600
    )["tokens"]


def detokenize(base_url: str, model: str, tokens: list[int]) -> str:
    return post(
        base_url, "/detokenize", {"model": model, "tokens": tokens}, 600
    )["prompt"]


def build_exact_prompt(
    base_url: str, model: str, target_tokens: int, nonce: str
) -> tuple[str, list[dict]]:
    prefix = (
        f"Cold long-context retrieval qualification {nonce}.\n"
        "The document contains six audit records. Remember every KEY=VALUE pair. "
        "All other prose is distractor material.\n"
    )
    suffix = (
        "\nEND OF DOCUMENT. Return all six audit records in document order, exactly "
        "as KEY=VALUE, one per line. Return nothing else.\nANSWER:\n"
    )
    filler = (
        "Ordinary archive prose describes quiet rivers, copper clocks, patient "
        "engineers, and slate valleys; it contains no audit key or value.\n"
    )
    prefix_ids = tokenize(base_url, model, prefix)
    suffix_ids = tokenize(base_url, model, suffix)
    filler_ids = tokenize(base_url, model, filler)
    fact_ids = [
        tokenize(
            base_url,
            model,
            f"\nAUDIT RECORD: {key}={value}\n",
        )
        for _, key, value in FACTS
    ]
    fixed = len(prefix_ids) + len(suffix_ids) + sum(map(len, fact_ids))
    if target_tokens <= fixed + len(FACTS):
        raise ValueError("target token count is too small for the qualification")

    # Place each record near its requested absolute depth, filling every gap
    # with a deterministic canonical token stream.
    result = list(prefix_ids)
    placements: list[dict] = []
    for (fraction, key, value), encoded in zip(FACTS, fact_ids):
        desired = round(target_tokens * fraction)
        gap = max(1, desired - len(result))
        repeats = math.ceil(gap / len(filler_ids))
        result.extend((filler_ids * repeats)[:gap])
        placements.append(
            {
                "fraction": fraction,
                "key": key,
                "value": value,
                "token_offset_before_record": len(result),
            }
        )
        result.extend(encoded)

    tail = target_tokens - len(result) - len(suffix_ids)
    if tail < 1:
        raise RuntimeError("record placement left no room for the question")
    # A repeated whole filler plus a neutral one-token pad gives the final
    # splice a stable boundary. Cutting arbitrary prose tokens can oscillate
    # between two canonical lengths and never produce an exact requested size.
    pad_ids = tokenize(base_url, model, " x")
    if len(pad_ids) != 1:
        raise RuntimeError("neutral padding must encode to one token")
    full_repeats = max(0, (tail - 128) // len(filler_ids))
    result.extend(filler_ids * full_repeats)
    padding_start = len(result)
    padding = target_tokens - padding_start - len(suffix_ids)
    result.extend(pad_ids * padding)
    result.extend(suffix_ids)

    # Decode/encode can canonicalize the two splice boundaries. Adjust only
    # the final distractor span until the HTTP-visible prompt is exact.
    for _ in range(8):
        prompt = detokenize(base_url, model, result)
        roundtrip = tokenize(base_url, model, prompt)
        delta = target_tokens - len(roundtrip)
        if delta == 0:
            break
        padding += delta
        if padding < 1:
            raise RuntimeError("canonicalization exhausted the final filler span")
        result = result[:padding_start] + pad_ids * padding + suffix_ids
    else:
        raise RuntimeError("could not canonicalize the exact prompt-token count")

    if len(roundtrip) != target_tokens:
        raise RuntimeError(
            f"prompt roundtrip has {len(roundtrip)} tokens, expected {target_tokens}"
        )
    for _, key, value in FACTS:
        record = f"{key}={value}"
        if prompt.count(record) != 1:
            raise RuntimeError(f"prompt does not contain exactly one {record!r}")
    return prompt, placements


def render_exact_prompt(base_url, model, target_tokens, nonce):
    # Qualify the actual official chat template. Close the generation-time
    # thinking prefix explicitly, as the throughput harness does, so retrieval
    # is scored from the final answer rather than hidden/repeated reasoning.
    close_think = post(base_url, "/tokenize", {
        "model": model, "prompt": "</think>", "add_special_tokens": False,
    }, 600)["tokens"]
    body_tokens = target_tokens - 64
    for _ in range(4):
        prompt, placements = build_exact_prompt(
            base_url, model, body_tokens, nonce
        )
        rendered = post(base_url, "/v1/chat/completions/render", {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": "low",
        }, 600)
        prompt_ids = rendered["token_ids"] + close_think
        delta = target_tokens - len(prompt_ids)
        if delta == 0:
            break
        body_tokens += delta
    else:
        raise RuntimeError("could not construct an exact chat-framed prompt")
    return prompt_ids, prompt, placements, body_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokens", type=int, default=1_000_000)
    parser.add_argument("--nonce", default="glm53-context-qualification")
    parser.add_argument("--cache-salt")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=14_400)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cache_salt = args.cache_salt or uuid.uuid4().hex
    started = time.perf_counter()
    prompt_ids, prompt, placements, body_tokens = render_exact_prompt(
        args.base_url, args.model, args.tokens, args.nonce)
    built = time.perf_counter()
    output, usage, ttft_seconds, request_seconds, finish_reason = stream_completion(
        args.base_url,
        {
            "model": args.model,
            "prompt": prompt_ids,
            "add_special_tokens": False,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "cache_salt": cache_salt,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        args.timeout,
    )
    finished = time.perf_counter()
    checks = [
        {
            "key": key,
            "expected": value,
            "passed": f"{key}={value}" in output,
        }
        for _, key, value in FACTS
    ]
    expected_lines = [f"{key}={value}" for _, key, value in FACTS]
    actual_lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
    exact_answer = actual_lines == expected_lines
    exact_usage = usage is not None and usage.get("prompt_tokens") == args.tokens
    report = {
        "schema": "glm53-cold-multi-needle.v2",
        "passed": all(check["passed"] for check in checks) and exact_answer
                  and exact_usage and finish_reason == "stop",
        "exact_answer": exact_answer,
        "exact_prompt_usage": exact_usage,
        "finish_reason": finish_reason,
        "framing": "server official chat template, reasoning low, explicit </think>",
        "target_prompt_tokens": args.tokens,
        "body_tokens": body_tokens,
        "placement_scope": "offsets within the user-message body before chat framing",
        "nonce": args.nonce,
        "cache_salt": cache_salt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_ids_sha256": hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest(),
        "placements": placements,
        "checks": checks,
        "usage": usage,
        "build_seconds": round(built - started, 3),
        "ttft_seconds": round(ttft_seconds, 3),
        "stream_after_first_token_seconds": round(
            request_seconds - ttft_seconds, 3
        ),
        "request_seconds": round(request_seconds, 3),
        "wall_seconds_after_build": round(finished - built, 3),
        "output": output,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
