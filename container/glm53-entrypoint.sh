#!/usr/bin/env bash
set -euo pipefail

READY_FILE=/tmp/glm53-release-ready
rm -f "${READY_FILE}"

# Reconstruct the target's partial EXL3 manifest from stored tensor metadata.
# The model repository is mounted read-only; the derived view lives in cache.
if [[ $# -gt 0 && -d "$1" && -f "$1/quantization_config.json" ]]; then
  model_source=$1
  shift
  prepared_model="/root/.cache/glm53/prepared-$(basename -- "$model_source")"
  template_cache=/root/.cache/glm53/official-chat-template
  template_args=()
  if [[ "${GLM53_TEMPLATE_REFRESH:-1}" == 0 ]]; then
    template_args+=(--offline)
  fi
  python3 /usr/local/bin/glm53-refresh-chat-template.py "$template_cache" "${template_args[@]}"
  python3 /usr/local/bin/glm53-prepare-model.py "$model_source" "$prepared_model" \
    --chat-template "$template_cache/chat_template.jinja"
  set -- "$prepared_model" "$@"
fi

vllm serve "$@" &
server_pid=$!

forward_term() {
  kill -TERM "${server_pid}" 2>/dev/null || true
}
trap forward_term TERM INT

if [[ "${GLM53_STARTUP_WARMUP:-1}" == 1 ]]; then
  set +e
  python3 /usr/local/bin/glm53-release-warmup.py \
    --server-pid "${server_pid}" \
    --base-url http://127.0.0.1:8001
  warmup_status=$?
  set -e
  if [[ ${warmup_status} -ne 0 ]]; then
    printf 'GLM release startup warmup failed with status %s\n' \
      "${warmup_status}" >&2
    forward_term
    wait "${server_pid}" 2>/dev/null || true
    exit "${warmup_status}"
  fi
elif [[ "${GLM53_STARTUP_WARMUP}" != 0 ]]; then
  echo "GLM53_STARTUP_WARMUP must be 0 or 1" >&2
  forward_term
  wait "${server_pid}" 2>/dev/null || true
  exit 2
fi

touch "${READY_FILE}"
printf 'GLM release startup warmup complete; container is ready.\n'

set +e
wait "${server_pid}"
server_status=$?
set -e
exit "${server_status}"
