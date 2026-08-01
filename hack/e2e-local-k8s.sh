#!/usr/bin/env bash
# Carrot AI Infra —— 本机 Docker Desktop K8s 真实闭环验证脚本
# 前提：Docker Desktop K8s 已启用；已构建 kk-infra/inference:local 镜像
# 流程：起 modelregistry/k8sadapter/controlplane/gateway → 注册模型 → 创建部署
#       → K8s 部署真实 Pod → OpenAI 调用 → 扩缩容 → 删除
set -euo pipefail

export PATH="/opt/homebrew/bin:$PATH"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NS="tenant-default"   # controlplane 默认命名空间（tenant-<id>）
IMAGE="kk-infra/inference:local"
PIDS=()

log() { echo -e "\n\033[1;36m=== $* ===\033[0m"; }
fail() { echo -e "\033[1;31mFAIL: $*\033[0m"; exit 1; }

cleanup() {
  log "清理进程"
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT

# ---------- 1. 准备 ----------
log "准备 namespace 与镜像"
kubectl get ns "$NS" >/dev/null 2>&1 || kubectl create ns "$NS"
# 确认镜像可用
docker image inspect "$IMAGE" >/dev/null 2>&1 || fail "请先构建镜像: docker build -f services/inference/Dockerfile -t $IMAGE ."
# Docker Desktop 集群可直接使用本地镜像
kubectl -n "$NS" get deploy 2>/dev/null | grep -q qwen-demo && kubectl -n "$NS" delete deploy qwen-demo --wait || true
kubectl -n "$NS" get svc 2>/dev/null | grep -q qwen-demo && kubectl -n "$NS" delete svc qwen-demo || true
# 清理残留 port-forward
lsof -ti :18080 >/dev/null 2>&1 && lsof -ti :18080 | xargs kill 2>/dev/null || true

# ---------- 2. 启动控制面服务（本地二进制） ----------
# 预编译：hack/e2e-local-k8s.sh 之前先执行 hack/build-binaries.sh
log "启动 modelregistry :8081"
/tmp/modelregistry --addr :8081 &
PIDS+=($!)

log "启动 k8sadapter :8082（真实集群 + 虚拟 GPU）"
/tmp/k8sadapter --fake=false \
  --kubeconfig "$HOME/.kube/config" \
  --virtual-gpus "gpu-001:A100:8:81920:42:Healthy" \
  --deploy-image "$IMAGE" &
PIDS+=($!)

log "启动 controlplane :8080"
/tmp/controlplane --addr :8080 \
  --model-registry http://127.0.0.1:8081 --k8s-adapter http://127.0.0.1:8082 \
  --deployment-image "$IMAGE" &
PIDS+=($!)

log "启动 gateway :8083"
/tmp/gateway --addr :8083 &
PIDS+=($!)

# ---------- 3. 等待服务就绪 ----------
log "等待服务就绪"
for i in $(seq 1 30); do
  ok=0
  curl -sf http://127.0.0.1:8081/api/v1/models >/dev/null 2>&1 && ok=$((ok+1))
  curl -sf http://127.0.0.1:8082/v1/resources/gpus >/dev/null 2>&1 && ok=$((ok+1))
  curl -sf http://127.0.0.1:8080/api/v1/deployments >/dev/null 2>&1 && ok=$((ok+1))
  [ "$ok" -ge 3 ] && break
  sleep 1
done
[ "$ok" -ge 3 ] || fail "控制面服务未就绪 (ok=$ok)"
log "控制面服务已就绪"

# ---------- 4. 验证 GPU 发现 ----------
log "4.1 GPU 资源发现（虚拟池 + K8s 节点）"
GPU_JSON=$(curl -s http://127.0.0.1:8080/api/v1/resources/gpus)
echo "$GPU_JSON" | head -c 400; echo
echo "$GPU_JSON" | grep -q "A100" || fail "GPU 发现缺少 A100"

# ---------- 5. 注册模型 ----------
log "5.1 注册模型 qwen"
curl -s -X POST http://127.0.0.1:8081/api/v1/models -H 'Content-Type: application/json' \
  -d '{"name":"qwen","description":"Qwen 系列测试模型"}' >/dev/null
MODEL_ID=$(curl -s http://127.0.0.1:8081/api/v1/models | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")

log "5.2 注册版本 7b"
VER_JSON=$(curl -s -X POST "http://127.0.0.1:8081/api/v1/models/$MODEL_ID/versions" \
  -H 'Content-Type: application/json' \
  -d '{"version":"7b","artifactUri":"s3://models/qwen-7b","runtime":"vLLM","gpuType":"A100","gpuCount":1,"memoryMB":2048,"contextLength":8192}')
VERSION_ID=$(echo "$VER_JSON" | python3 -c "import sys,json;print(json.load(sys.stdin)['data']['id'])")

log "5.3 校验版本"
curl -s -X POST "http://127.0.0.1:8081/api/v1/versions/$VERSION_ID/validate" >/dev/null

# ---------- 6. 创建部署（真实 K8s） ----------
log "6.1 创建模型服务 qwen-demo（1 GPU × 1 副本）"
DEPLOY_JSON=$(curl -s -X POST http://127.0.0.1:8080/api/v1/deployments -H 'Content-Type: application/json' \
  -d "{\"idempotencyKey\":\"deploy-qwen-001\",\"name\":\"qwen-demo\",\"modelVersionId\":\"$VERSION_ID\",\"tenantId\":\"default\",\"replicas\":1}")
echo "$DEPLOY_JSON" | head -c 300; echo
DEPLOY_ID=$(echo "$DEPLOY_JSON" | python3 -c "import sys,json;print(json.load(sys.stdin)['data']['id'])")

# ---------- 7. 等待 Running ----------
log "6.2 等待部署进入 RUNNING（K8s 拉取镜像 + 就绪探针）"
READY=""
for i in $(seq 1 60); do
  STATUS=$(curl -s http://127.0.0.1:8080/api/v1/deployments/$DEPLOY_ID | python3 -c "import sys,json;print(json.load(sys.stdin)['data']['status'])")
  if [ "$STATUS" = "RUNNING" ]; then READY=1; break; fi
  sleep 2
done
[ -n "$READY" ] || fail "部署未进入 RUNNING，最终状态=$STATUS"
log "部署已 RUNNING"

log "6.3 K8s 侧确认 Deployment/Pod"
kubectl -n "$NS" get deploy,pods -l carrot.ai/deployment-id="$DEPLOY_ID" 2>/dev/null || \
  kubectl -n "$NS" get deploy,pods -l app=qwen-demo

# ---------- 8. 配置 gateway 路由 ----------
log "7.1 port-forward 暴露模型服务 + 注册 gateway 路由"
# 网关在宿主机运行，无法解析 K8s 内部 DNS；用 port-forward 暴露到宿主机
kubectl -n "$NS" port-forward svc/qwen-demo 18080:80 >/tmp/pf-qwen-demo.log 2>&1 &
PF_PID=$!
PIDS+=($PF_PID)
# 等待 port-forward 就绪
for i in $(seq 1 20); do
  curl -sf http://127.0.0.1:18080/health >/dev/null 2>&1 && break
  sleep 1
done
curl -s -X POST http://127.0.0.1:8083/internal/routes -H 'Content-Type: application/json' \
  -d "{\"model\":\"qwen-demo\",\"endpoint\":\"http://127.0.0.1:18080\",\"tenantId\":\"default\",\"deploymentId\":\"$DEPLOY_ID\"}" >/dev/null
log "路由已注册: qwen-demo → http://127.0.0.1:18080"

log "7.2 创建 API Key"
KEY_JSON=$(curl -s -X POST http://127.0.0.1:8083/api/v1/keys -H 'Content-Type: application/json' -d '{"tenantId":"default"}')
API_KEY=$(echo "$KEY_JSON" | python3 -c "import sys,json;print(json.load(sys.stdin)['data']['key'])")
echo "API Key: $API_KEY"

# ---------- 9. OpenAI 调用 ----------
log "8.1 非流式调用 /v1/chat/completions"
RESP=$(curl -s -X POST http://127.0.0.1:8083/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen-demo","messages":[{"role":"user","content":"你好，介绍一下你自己"}]}')
echo "$RESP" | head -c 400; echo
echo "$RESP" | grep -q "回复:" || fail "非流式调用失败"

log "8.2 流式调用"
STREAM_OUT=$(curl -s -N -X POST http://127.0.0.1:8083/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen-demo","messages":[{"role":"user","content":"讲个故事"}],"stream":true}' | head -c 500)
echo "$STREAM_OUT" | head -c 300; echo
echo "$STREAM_OUT" | grep -q "data:" || fail "流式调用失败"

# ---------- 10. 扩缩容 ----------
log "9.1 扩容到 2 副本"
curl -s -X POST http://127.0.0.1:8080/api/v1/deployments/$DEPLOY_ID/scale \
  -H 'Content-Type: application/json' -d '{"replicas":2}' >/dev/null
sleep 3
kubectl -n "$NS" get deploy qwen-demo -o jsonpath='{.spec.replicas}' | grep -q "2" || fail "扩容未生效"
log "扩容到 2 副本成功"

# ---------- 11. 删除 ----------
log "10.1 删除部署（释放 GPU）"
curl -s -X DELETE http://127.0.0.1:8080/api/v1/deployments/$DEPLOY_ID >/dev/null
sleep 2
kubectl -n "$NS" get deploy qwen-demo 2>/dev/null && fail "删除后 Deployment 仍存在" || true
log "部署已删除，K8s 资源已释放"

# ---------- 12. 无效 Key ----------
log "11.1 无效 API Key 校验"
CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://127.0.0.1:8083/v1/chat/completions \
  -H "Authorization: Bearer sk-carrot-invalid" -H 'Content-Type: application/json' \
  -d '{"model":"qwen-demo","messages":[{"role":"user","content":"hi"}]}')
[ "$CODE" = "401" ] || fail "无效 Key 应返回 401，实际 $CODE"

log ""
log "🎉 MVP 闭环验证全部通过：发现 GPU → 注册模型 → 创建 vLLM 服务 → OpenAI 调用（含流式）→ 扩容 → 删除释放"
log "（控制面重启恢复、指标查询属于 Fake 环境特性，真实 K8s 验证见 observability 服务）"
