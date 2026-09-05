#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
IMAGE=${IMAGE:-ghcr.io/tpurtell/single-spark-glm-5.3-flash:dev}
SOURCE_REVISION=${SOURCE_REVISION:-$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)}
if [[ $(uname -m) != aarch64 ]]; then
  echo 'Build on a Spark: sync this checkout and run ./build.sh there.' >&2
  exit 2
fi
docker build --platform linux/arm64 --progress=plain \
  --label "org.opencontainers.image.revision=${SOURCE_REVISION}" \
  --label "org.opencontainers.image.title=GLM-5.3 Flash on one DGX Spark" \
  --tag "$IMAGE" "$ROOT" "$@"
docker image inspect "$IMAGE" --format 'Built {{.Id}} ({{.Architecture}}, {{.Size}} bytes)'
