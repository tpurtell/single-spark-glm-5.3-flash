#!/usr/bin/env python3
"""Capture and validate the exact server/GPU environment for release results."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Any


CAPACITY_RE = re.compile(
    r"GPU KV cache size: (?P<tokens>[0-9,]+) tokens, Maximum concurrency for "
    r"(?P<request>[0-9,]+) tokens per request: (?P<concurrency>[0-9.]+)x"
)
READY_TEXT = "GLM release startup warmup complete; container is ready."
JIT_TEXT = "JIT compilation during inference:"
ENV_PREFIXES = ("VLLM_", "B12X_", "NCCL_", "CUDA_", "KV_FP8_", "GLM53_", "TORCH_BLAS_")
ENV_EXACT = {
    "PYTORCH_CUDA_ALLOC_CONF",
    "HF_HUB_OFFLINE",
    "GLM53_STARTUP_WARMUP",
    "GLM53_REPLAYSSM_ACTIVE",
}


def command(*args: str) -> str:
    return subprocess.run(
        args,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def api_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError(f"API did not return an object: {url}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def mounted_model_root(inspect: dict[str, Any]) -> Path:
    command_line = inspect["Config"].get("Cmd") or []
    if not command_line or not isinstance(command_line[0], str):
        raise ValueError("container command has no model path")
    container_path = Path(command_line[0])
    candidates = []
    for mount in inspect.get("Mounts") or []:
        destination = Path(mount["Destination"])
        try:
            relative = container_path.relative_to(destination)
        except ValueError:
            continue
        candidates.append((len(destination.parts), Path(mount["Source"]) / relative))
    if not candidates:
        raise ValueError(f"model path is not covered by a host mount: {container_path}")
    return max(candidates, key=lambda item: item[0])[1].resolve(strict=True)


def gpu_environment() -> list[dict[str, Any]]:
    fields = (
        "index,name,uuid,pci.bus_id,driver_version,memory.total,power.limit,"
        "power.default_limit,pstate,clocks.current.sm,clocks.current.memory,"
        "temperature.gpu"
    )
    output = command(
        "nvidia-smi",
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    )
    names = fields.split(",")
    rows = []
    for line in output.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) != len(names):
            raise ValueError(f"unexpected nvidia-smi row: {line!r}")
        row = dict(zip(names, values, strict=True))
        row["index"] = int(row["index"])
        for name in ("memory.total", "power.limit", "power.default_limit"):
            row[name] = None if row[name] == '[N/A]' else float(row[name])
        rows.append(row)
    return rows


def runtime_overrides(inspect: dict[str, Any], model_root: Path) -> dict[str, Any]:
    """Record the served metadata overlay, not just the read-only source quant."""
    mounts = [mount for mount in inspect.get('Mounts', [])
              if mount['Destination'] == '/root/.cache']
    if len(mounts) != 1:
        raise ValueError('expected one mounted runtime cache')
    cache = Path(mounts[0]['Source']) / 'glm53'
    prepared = cache / f'prepared-{model_root.name}'
    preparation_path = prepared / 'preparation.json'
    template_path = cache / 'official-chat-template' / 'chat_template.json'
    preparation = json.loads(preparation_path.read_text())
    template = json.loads(template_path.read_text())
    template_hash = sha256_file(prepared / 'chat_template.jinja')
    manifest_hash = sha256_file(prepared / 'quantization_config.json')
    if template_hash != preparation['chat_template_sha256'] or template_hash != template['sha256']:
        raise ValueError('served template does not match its source receipts')
    if manifest_hash != preparation['derived_manifest_sha256']:
        raise ValueError('served quantization manifest does not match its receipt')
    if preparation.get('weights_modified') is not False:
        raise ValueError('preparation did not attest read-only weights')
    return {'prepared_host_root': str(prepared), 'preparation': preparation,
            'official_chat_template': template,
            'served_chat_template_sha256': template_hash,
            'served_quantization_manifest_sha256': manifest_hash}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", required=True)
    parser.add_argument("--power-limit", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.machine() != 'aarch64':
        raise SystemExit('Run environment capture on the serving Spark, not the workstation')

    inspect = json.loads(command("docker", "inspect", args.container))[0]
    if inspect["State"]["Status"] != "running":
        raise ValueError(f"container is not running: {inspect['State']['Status']}")
    health = urllib.request.urlopen(
        args.base_url.rstrip("/") + "/health", timeout=30
    ).status
    if health != 200:
        raise ValueError(f"health endpoint returned HTTP {health}")
    models = api_json(args.base_url.rstrip("/") + "/v1/models")
    served = [item.get("id") for item in models.get("data", [])]
    if args.model not in served:
        raise ValueError(f"expected served model {args.model!r}, observed {served!r}")

    gpus = gpu_environment()
    if len(gpus) != 1 or gpus[0]['name'] != 'NVIDIA GB10':
        raise ValueError(f"release requires one GB10, observed {gpus}")
    for gpu in gpus:
        if args.power_limit is not None and (gpu['power.limit'] is None or
                                             abs(gpu["power.limit"] - args.power_limit) > 0.05):
            raise ValueError(
                f"GPU {gpu['index']} power limit is {gpu['power.limit']} W, "
                f"expected {args.power_limit:.2f} W"
            )

    logs = command("docker", "logs", "--timestamps", args.container)
    capacity_matches = list(CAPACITY_RE.finditer(logs))
    if not capacity_matches:
        raise ValueError("server log has no GPU KV cache capacity line")
    capacity_match = capacity_matches[-1]
    lines = logs.splitlines()
    ready_offsets = [index for index, line in enumerate(lines) if READY_TEXT in line]
    if len(ready_offsets) != 1:
        raise ValueError(f"expected one release-ready marker, observed {len(ready_offsets)}")
    post_ready = lines[ready_offsets[0] + 1 :]

    selected_env: dict[str, str] = {}
    for item in inspect["Config"].get("Env") or []:
        name, separator, value = item.partition("=")
        if separator and (name in ENV_EXACT or name.startswith(ENV_PREFIXES)):
            selected_env[name] = value
    model_root = mounted_model_root(inspect)
    model_index = model_root / "model.safetensors.index.json"
    quantization_config = model_root / "quantization_config.json"
    if not quantization_config.is_file():
        quantization_config = model_root / "quantize_config.json"
    for path in (model_root / "config.json", model_index, quantization_config):
        if not path.is_file():
            raise FileNotFoundError(f"served model metadata is missing: {path}")
    capacity = {
        "tokens": int(capacity_match.group("tokens").replace(",", "")),
        "request_tokens": int(capacity_match.group("request").replace(",", "")),
        "maximum_concurrency": float(capacity_match.group("concurrency")),
        "log_line": capacity_match.group(0),
    }
    report = {
        "schema": "glm53-release-environment-v2",
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "api_health_status": health,
        "served_models": served,
        "container": {
            "name": args.container,
            "id": inspect["Id"],
            "image_reference": inspect["Config"]["Image"],
            "image_id": inspect["Image"],
            "image_labels": inspect["Config"].get("Labels") or {},
            "created": inspect["Created"],
            "command": inspect["Config"].get("Cmd") or [],
            "environment": selected_env,
            "ready_marker": lines[ready_offsets[0]],
            "post_ready_request_lines": sum('"POST ' in line for line in post_ready),
            "post_ready_jit_count": sum(JIT_TEXT in line for line in post_ready),
        },
        "gpus": gpus,
        "host_memory": command('free', '-b'),
        "host_meminfo": Path('/proc/meminfo').read_text(),
        "required_power_limit_watts": args.power_limit,
        "kv_cache": capacity,
        "runtime_overrides": runtime_overrides(inspect, model_root),
        "model_artifact": {
            "host_root": str(model_root),
            "config_sha256": sha256_file(model_root / "config.json"),
            "model_index_sha256": sha256_file(model_index),
            "quantization_config_name": quantization_config.name,
            "quantization_config_sha256": sha256_file(quantization_config),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
