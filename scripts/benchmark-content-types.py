#!/usr/bin/env python3
"""Run DS4RT's seven weighted content prompts plus cache-busted Orchid."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from glm53_api import visible_prompt


@dataclass(frozen=True)
class PromptCase:
    category: str
    prompt: str
    max_tokens: int


# Kept byte-for-byte aligned with ../ds4rt/python/tools/
# bench_real_full_mtp_acceptance.py's weighted corpus.
CASES = {
    "code": PromptCase(
        "code",
        "Write a Python function merge_intervals(intervals) that merges overlapping "
        "integer intervals. Include type hints, a short docstring, and three assert-based "
        "examples. Return only one Python code block.",
        320,
    ),
    "math": PromptCase(
        "reasoning",
        "A shop discounts a $240 jacket by 25%, then applies 8% sales tax to the "
        "discounted price. What is the final price? Show the calculation briefly.",
        128,
    ),
    "fable": PromptCase(
        "creative-prose",
        "Write a self-contained fable of 140 to 170 words about two parrots who disagree "
        "about sharing credit. End with a one-sentence moral.",
        256,
    ),
    "hello": PromptCase("short-response", "hi", 32),
    "topic": PromptCase(
        "exposition",
        "Explain virtual memory to a junior programmer in five concise bullet points, "
        "including paging, page faults, and the role of the TLB.",
        224,
    ),
    "structured-json": PromptCase(
        "structured-output",
        "Return only a JSON object describing a file edit with keys path, operation, "
        "line_start, line_end, and rationale. Use path src/cache.rs, operation replace, "
        "lines 41 through 47, and a one-sentence rationale about removing a redundant copy.",
        128,
    ),
    "multilingual": PromptCase(
        "multilingual",
        "請用繁體中文，以四個簡短條列解釋什麼是寫入時複製（copy-on-write），"
        "並包含一個行程 fork 後修改記憶體頁面的例子。",
        192,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--orchid-warmups", type=int, default=1)
    parser.add_argument("--orchid-count", type=int, default=100)
    parser.add_argument("--orchid-max-tokens", type=int, default=1500)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.orchid_count < 1 or args.orchid_max_tokens < 1:
        parser.error("repeats, orchid count, and orchid max tokens must be positive")
    if args.orchid_warmups < 0 or args.timeout <= 0:
        parser.error("orchid warmups must be non-negative and timeout must be positive")
    return args


def stream_completion(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    body = {
        "model": model,
        "prompt": visible_prompt(base_url, model, [{"role": "user", "content": prompt}], timeout),
        "add_special_tokens": False,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason = None
    usage = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            choices = event.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            content = (choices[0].get("text") or "") if choices else ""
            reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""
            if first is None and (content or reasoning):
                first = time.perf_counter()
            content_parts.append(content)
            reasoning_parts.append(reasoning)
            if choices and choices[0].get("finish_reason"):
                finish_reason = choices[0]["finish_reason"]
            if event.get("usage"):
                usage = event["usage"]
    finished = time.perf_counter()
    if usage is None:
        raise RuntimeError("stream ended without a usage record")
    completion_tokens = int(usage["completion_tokens"])
    first = first or finished
    decode_seconds = max(0.001, finished - first)
    return {
        "prompt_tokens": int(usage["prompt_tokens"]),
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "ttft_s": first - started,
        "elapsed_s": finished - started,
        "decode_s": decode_seconds,
        "decode_tok_s": max(0, completion_tokens - 1) / decode_seconds,
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
    }


def compact_record(
    *, case_id: str, category: str, repeat: int, timed: bool, raw: dict[str, Any]
) -> dict[str, Any]:
    content = raw.pop("content")
    reasoning = raw.pop("reasoning")
    record = {
        "case": case_id,
        "category": category,
        "repeat": repeat,
        "timed": timed,
        **raw,
        "content_chars": len(content),
        "content_words": len(re.findall(r"\b\w+\b", content, re.UNICODE)),
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "content_preview": content[:240].replace("\n", "\\n"),
        "reasoning_chars": len(reasoning),
    }
    if case_id == "orchid":
        occurrences = len(re.findall(r"\borchid\b", content, re.IGNORECASE))
        record["observed_orchid_count"] = occurrences
        record["exact_orchid_count"] = occurrences == 100
    return record


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["timed"]:
            by_case.setdefault(record["case"], []).append(record)
    summaries = {}
    for case_id, samples in by_case.items():
        rates = [float(sample["decode_tok_s"]) for sample in samples]
        total_tokens = sum(max(0, int(sample["completion_tokens"]) - 1) for sample in samples)
        total_seconds = sum(float(sample["decode_s"]) for sample in samples)
        summary = {
            "category": samples[0]["category"],
            "samples": len(samples),
            "aggregate_decode_tok_s": total_tokens / total_seconds,
            "median_decode_tok_s": statistics.median(rates),
            "min_decode_tok_s": min(rates),
            "max_decode_tok_s": max(rates),
            "completion_tokens": [int(sample["completion_tokens"]) for sample in samples],
            "finish_reasons": [sample["finish_reason"] for sample in samples],
        }
        if case_id == "orchid":
            summary["exact_repetitions"] = sum(
                bool(sample["exact_orchid_count"]) for sample in samples
            )
            summary["observed_orchid_counts"] = [
                int(sample["observed_orchid_count"]) for sample in samples
            ]
        summaries[case_id] = summary
    return summaries


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    run_id = args.run_id or f"run-{time.time_ns()}"
    report: dict[str, Any] = {
        "schema": "mia-kx-ds4rt-content-v1",
        "source": "../ds4rt/python/tools/bench_real_full_mtp_acceptance.py and bench_real_full_repeat_decode.py",
        "run_id": run_id,
        "base_url": args.base_url,
        "model": args.model,
        "repeats": args.repeats,
        "orchid_warmups": args.orchid_warmups,
        "thinking": "off",
        "temperature": 0,
        "prompt_cases": {case_id: asdict(case) for case_id, case in CASES.items()},
        "records": [],
    }

    for repeat in range(1, args.repeats + 1):
        for case_id, case in CASES.items():
            raw = stream_completion(
                base_url=args.base_url,
                model=args.model,
                prompt=case.prompt,
                max_tokens=case.max_tokens,
                timeout=args.timeout,
            )
            record = compact_record(
                case_id=case_id,
                category=case.category,
                repeat=repeat,
                timed=True,
                raw=raw,
            )
            report["records"].append(record)
            write_report(args.output, report)
            print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)

    orchid_samples = args.orchid_warmups + args.repeats
    for sample in range(orchid_samples):
        marker = f"{run_id}-orchid-{sample}"
        prompt = (
            f"orchid {marker}\n"
            f'Repeat only the single word "orchid" exactly {args.orchid_count} '
            "times, separated by spaces. Do not repeat the nonce or add any other text."
        )
        raw = stream_completion(
            base_url=args.base_url,
            model=args.model,
            prompt=prompt,
            max_tokens=args.orchid_max_tokens,
            timeout=args.timeout,
        )
        record = compact_record(
            case_id="orchid",
            category="cache-busted repetition",
            repeat=sample + 1,
            timed=sample >= args.orchid_warmups,
            raw=raw,
        )
        # compact_record's default is the DS4RT benchmark's 100-word contract.
        if args.orchid_count != 100:
            record["exact_orchid_count"] = (
                record["observed_orchid_count"] == args.orchid_count
            )
        report["records"].append(record)
        write_report(args.output, report)
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)

    report["summary"] = summarize(report["records"])
    write_report(args.output, report)
    print(json.dumps({"summary": report["summary"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
