# Carrot AI Infra 架构设计（V2：生产化单集群）

## 1. 总体结构

```text
Console / OpenAI Client
          │
   API Gateway（鉴权/限流/路由/用量）
          │
Control Plane（模型/部署/配额/审计/状态机）
      │                 │
 Model Registry     Postgres
      │
 Kubernetes Adapter + Reconciler
      │
 Kubernetes + GPU Operator + Kueue/Volcano + vLLM
      │
 Prometheus/DCGM ─── Observability API ─── Dashboard/Alert
```

## 2. 责任边界

- `controlplane`：唯一业务状态和期望状态来源；不直接依赖 Kubernetes SDK
- `modelregistry`：模型版本、artifact、校验和发布状态；不负责启动 Pod
- `k8sadapter`：把期望对象渲染为 K8s 资源，负责 informer、事件和实际状态
- `gateway`：只处理请求面；按租户和模型授权后路由到服务，异步上报用量
- `observability`：查询 Prometheus/DCGM 与推理指标，不承担部署状态机
- `inference`：仅本地 Mock；生产由 vLLM 工作负载替代

## 3. 一次部署的数据流

```text
POST /deployments
 → 校验 ModelVersion / TenantQuota
 → Postgres 写入期望状态和幂等键
 → Reconciler 创建 Namespace/Secret/Deployment/Service
 → K8s 调度 GPU，vLLM readiness 通过
 → Adapter 上报实际状态和事件
 → Control Plane 收敛 RUNNING + Endpoint
 → Gateway 注册路由
```

所有写操作由 `request_id`、`idempotency_key`、`generation` 关联；Reconcile 可重复执行，资源以 `carrot.ai/*` 标签定位。

## 4. 持久化与一致性

Postgres 表至少包括 `models`、`model_versions`、`deployments`、`deployment_events`、`tenant_quotas`、`api_keys`、`audit_logs`。API 只负责写期望状态，后台 Reconciler 负责最终一致；启动时按标签扫描 K8s 资源并修复本地状态。

## 5. 安全与可靠性基线

- 租户隔离：Namespace + RBAC + ResourceQuota + NetworkPolicy
- Secret/API Key 哈希存储，日志脱敏，artifact URI 不直接下发给无权用户
- Deployment/Service/PDB/Secret 使用统一 owner label，删除处理 NotFound
- 创建、扩缩、删除有超时、有限重试、失败诊断和孤儿回收
- Gateway 超时、限流、熔断；流式请求不能被普通重试重复提交

## 6. 指标契约

基础设施：`gpu_utilization`、`gpu_memory_used`、`gpu_temperature`；推理：`requests`、`errors`、`ttft_ms`、`tpot_ms`、`tokens_per_sec`、`queue_length`、`kv_cache_utilization`；维度统一为 `tenant_id/model_id/deployment_id/pod`。

## 7. 演进边界

先稳定单集群控制面和推理数据面；多集群通过新增 `ClusterAdapter` 与 placement policy 扩展，不改领域对象。训练、Prefill/Decode、AIOps 属于后续独立 bounded context，避免把当前控制面变成“大而全”的平台服务。
