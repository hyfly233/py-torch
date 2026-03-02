# kk-infra — Carrot AI Infra 分布式微服务平台 TODO（V2 路线）

> 当前路线与执行基线：`docs/ROADMAP-V2.md`、`docs/PRD-V2.md`、`docs/ARCHITECTURE-V2.md`、`docs/INTERACTION-PROTOTYPE-V2.md`
> 历史基线：`docs/PRD.md`、`docs/AI-INFRA-PLAN.md`、`docs/ARCHITECTURE.md`、`docs/INTERACTION-PROTOTYPE.md`
> 目标：从 MVP 闭环走向单集群生产化。执行基线：单集群、vLLM、OpenAI 兼容 API；Fake K8s 只用于开发验收，不能作为生产能力。
> 工具链：Go 1.26.5（PATH 默认 1.20.4，必须用 `/opt/homebrew/bin/go`）。

---

## 0. 当前状态（2026-08-01，V2 评估）

### 已完成 ✅（MVP 闭环）

| 阶段 | 内容 |
|---|---|
| P0 | lib 共享库：领域模型/状态机/错误码/中间件/API 类型 |
| P1 | modelregistry：模型/版本 CRUD/校验/唯一性 |
| P2 | k8sadapter：KubeClient 接口/Fake/vLLM renderer/真实 K8s REST 客户端 |
| P3 | controlplane：部署生命周期/状态机/GPU 资源/Reconciler |
| P4 | gateway + inference：OpenAI 兼容 API/API Key/路由/流式 |
| P5 | observability 服务 + 指标链路（gateway 埋点 → sink → 聚合 → 查询） |
| 前端 | 全部页面 + 角色权限 + 路由守卫 |
| E2E | `hack/e2e-test.sh`（Fake）+ `hack/e2e-local-k8s.sh`（真实 K8s 闭环已跑通） |

### V2 文档要求 vs 现状差距 ⚠️

| V2 要求 | 现状 | 差距 |
|---|---|---|
| 契约测试覆盖 controlplane/gateway/observability | 有单测/集成测试，无跨服务契约测试 | 需补 contract tests |
| E2E 流式断言稳定 | 真实 K8s 已通，Fake 脚本需确认 | 需固化 |
| Postgres 替换内存 repository | 全部内存 `sync.RWMutex` | 需 Postgres + migration |
| 真实 client-go/informer + 重启对账 | 自写 REST client（非 client-go），Fake 对账 | 需 client-go/informer |
| Prometheus + DCGM 指标 | observability 内存 + collector 从 controlplane 拉 GPU | 需 Prometheus adapter + DCGM |
| 租户/RBAC/配额/NetworkPolicy | 单租户 default，TenantQuota 模型存在未落地 | 需租户隔离完整实现 |
| API Key 授权模型、审计日志 | Key 有哈希/禁用/轮换，无模型授权/审计 | 需扩展 |
| RELEASED 发布状态 + 发布记录 | 仅 REGISTERED/VALIDATED，无 RELEASED/发布记录 | 需状态扩展 |
| 失败诊断/孤儿回收/超时重试 | 有基础诊断，无孤儿回收 | 需补齐 |
| 前端：租户与配额页/告警与审计页 | 无 | 需新增页面 |

### 已完成但需评审确认的点

- 前端 `useAuth` 角色判定已修复（模板解构 ref）
- 部署幂等 + 同名重建已修复
- 真实 K8s e2e 三处修复：selector matchLabels / PATCH merge-patch Content-Type / mock 镜像模型名

---

## 1. V2 路线规划

### R1：MVP 收口（当前，优先级最高）——含 Postgres 持久化

目标：让本地演示和代码契约稳定，为 R2 打地基。**本轮使用本机 docker Postgres（`/Users/flyhy/workspace/docker/postgresql`，localhost:5432）落地持久化。**

- [ ] **R1-0 Postgres 持久化**（用户指定用本机 pg）
  - [ ] 创建专用数据库 `carrot`（`docker exec postgres psql -U postgres -c "CREATE DATABASE carrot"`）
  - [ ] 引入 `pgx` 驱动；新增各服务 `data/postgres.go` repository 实现
  - [ ] migration：SQL 文件 + 启动时执行；表：`models`、`model_versions`、`deployments`、`deployment_events`、`tenant_quotas`、`api_keys`、`audit_logs`
  - [ ] repository 接口不变，Postgres 实现替换内存；`hack/dev-up.sh` 支持 `--storage=postgres|memory`（默认 postgres）
  - [ ] 控制面重启恢复：启动时从 Postgres 读期望状态 + 从 K8s 对账
- [ ] **R1-1 契约测试**：新增 `hack/contract-test.sh` 或 Go contract 测试，覆盖 controlplane↔modelregistry↔k8sadapter↔gateway↔observability 的跨服务请求/响应契约（用 httptest 起真实服务对拍）
  - 统一 API 错误、状态、标签、幂等语义断言
- [ ] **R1-2 E2E 流式断言固化**：修复 `e2e-test.sh` 中 `curl|head -c` 的 SIGPIPE 问题（用 `-N` + 完整读取或 python 断言）；确认 gateway 流式透传真实链路
- [ ] **R1-3 服务详情补齐**：事件、诊断、API 示例字段对齐 V2 交互原型（服务详情页 6 个 tab 的数据契约）
- [ ] **R1-4 文档一致性**：README/TODO 与 V2 文档对齐；删除 V1 文档引用或标注"历史"

### R2：单集群生产化（优先级最高）

目标：真实集群上可持续运行一个租户或多个基础租户。

#### R2-1 Postgres 持久化
- [ ] 引入 `pgx` 或 `database/sql + lib/pq`，新增 `lib/store/` 或各服务 `data/postgres.go`
- [ ] migration 工具（`golang-migrate` 或自写 SQL 文件 + 启动执行）
- [ ] 表：`models`、`model_versions`、`deployments`、`deployment_events`、`tenant_quotas`、`api_keys`、`audit_logs`（对齐 ARCHITECTURE-V2 §4）
- [ ] repository 接口不变，Postgres 实现替换内存；`hack/dev-up.sh` 支持 `--storage=postgres|memory`
- [ ] 审计事件持久化 + 查询 API
- [ ] 控制面重启恢复：启动时从 Postgres 读期望状态 + 从 K8s 对账

#### R2-2 真实 K8s adapter（client-go/informer）
- [ ] 引入 `k8s.io/client-go`，实现 `client-go` 版 KubeClient（替换/并存自写 REST client）
- [ ] informer：监听 Deployment/Pod/Event，事件驱动状态同步（替代轮询）
- [ ] 控制面重启后从 K8s 对账恢复：按 `carrot.ai/*` 标签扫描资源，修复本地状态
- [ ] 孤儿资源回收：本地无记录的 K8s 资源（owner label 校验）→ 删除或告警
- [ ] 删除清理：Deployment/Service/Secret/PDB 统一 owner label，NotFound 幂等

#### R2-3 Prometheus + DCGM 指标链路
- [ ] observability 增加 Prometheus adapter：直接查询 Prometheus HTTP API（`promql`）
- [ ] 部署 DCGM Exporter（K8s DaemonSet 或 helm chart），指标 `gpu_utilization/gpu_memory_used/gpu_temperature`
- [ ] 推理指标维度统一：`tenant_id/model_id/deployment_id/pod`（对齐 ARCHITECTURE-V2 §6）
- [ ] 指标查询 API 支持按 deployment/tenant/model 维度 + 时间范围

#### R2-4 租户隔离与安全基线
- [ ] 多租户：Namespace 隔离（`tenant-<id>`），跨租户不可见/不可调用
- [ ] RBAC：k8sadapter 按租户最小权限（ServiceAccount + Role）
- [ ] ResourceQuota：每租户 GPU/内存/CPU 配额
- [ ] NetworkPolicy：租户间默认拒绝
- [ ] API Key 扩展：按模型授权（key ↔ model 白名单）、租户绑定、日志脱敏
- [ ] 审计日志：部署/删除/扩缩容/API Key 操作全审计

#### R2-5 发布流程与可靠性
- [ ] 模型版本状态扩展：`REGISTERED → VALIDATED → RELEASED`，仅 RELEASED 可部署
- [ ] 发布记录：校验结果、基准指标、操作者、时间
- [ ] 部署超时/有限重试/失败诊断：STARTING 超时 → FAILED + 诊断原因
- [ ] 删除失败重试 + 孤儿资源回收

### R3：推理平台能力（R2 稳定后）

- [ ] 发布门禁：artifact 校验、启动探针、基准测试、人工确认
- [ ] 灰度/回滚、模型路由、限流 fallback
- [ ] 推理扩缩容（QPS/队列长度/TTFT/KV Cache 驱动）
- [ ] 成本估算、Token 用量、租户账单
- [ ] Dashboard/告警/SLO

### R4：平台扩展（不设前置，R2/R3 稳定后再做）

- 多运行时（Triton/TensorRT-LLM）、多集群调度、训练 Notebook/Pipeline、
  Prefill/Decode 分离、AIOps Agent（ROADMAP-V2 §2 R4 明确暂不做）

---

## 2. 近期迭代顺序（ROADMAP-V2 §3）

```text
1. E2E/契约测试与文档一致性     → R1（本轮）
2. Postgres repository + migration → R2-1
3. 真实 K8s adapter + informer/reconcile → R2-2
4. Prometheus/DCGM 指标链路      → R2-3
5. 租户/RBAC/配额/安全基线      → R2-4
6. 发布门禁与灰度回滚            → R3
```

每批完成即 `./hack/check.sh`（build + vet + test）验证，全部通过再进下一批。

---

## 3. 前端跟进（随 R2 推进）

- [ ] 新增"租户与配额"页：配额/已用/队列/服务数/Token 用量，管理员可调整配额
- [ ] 新增"告警与审计"页：审计日志查询、告警列表
- [ ] 部署向导第 4 步：确认页（服务名/租户/访问策略/幂等键）
- [ ] 服务详情页 6 tab：性能/GPU/日志/事件/配置/API 示例（对齐 INTERACTION-PROTOTYPE-V2 §3）
- [ ] GPU 资源页三级展开：Pool → 节点 → GPU 卡
- [ ] API Key 授权模型功能
- [ ] 异步操作"受理成功 + 轮询"反馈统一

---

## 4. 暂不做的决定（ROADMAP-V2 §4）

不引入 Kubeflow 全家桶、Karmada、自研 GPU scheduler、Agent AIOps。
多运行时、多集群、训练、Prefill/Decode 分离均属 R4，不在当前优先级。
