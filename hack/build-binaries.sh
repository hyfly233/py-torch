#!/usr/bin/env bash
# 预编译 kk-infra 控制面二进制到 /tmp，供 e2e-local-k8s.sh 使用。
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

for svc in modelregistry k8sadapter controlplane gateway; do
  echo "编译 $svc ..."
  (cd "$ROOT/services/$svc" && go build -o "/tmp/$svc" ./cmd)
done
echo "全部编译完成: /tmp/{modelregistry,k8sadapter,controlplane,gateway}"
