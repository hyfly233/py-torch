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

- [x] **R1-0 Postgres 持久化**（用户指定用本机 pg）
  - [x] 创建专用数据库 `carrot`（`docker exec postgres psql -U postgres -c "CREATE DATABASE carrot"`）
  - [x] 引入 `database/sql + lib/pq`（本机已缓存，零下载）；新增各服务 `data/postgres.go` repository 实现
  - [x] migration：`lib/store/migrations/001_init.sql` + `store.MigrateAll`（幂等可重入）；表：models/model_versions/deployments/deployment_events/tenant_quotas/api_keys/audit_logs
  - [x] repository 接口不变，Postgres 实现替换内存；`hack/dev-up.sh --storage=postgres|memory`（默认 memory）
  - [x] 控制面重启恢复：PG 重启后部署状态/事件完整恢复（已验证）
- [x] **R1-1 契约测试**：`hack/contract-test.sh`（真实服务 HTTP 对拍）
  - [x] 错误码契约：404/400/409/401；状态机：初始非终态→RUNNING；幂等：同 Key 同部署/幂等删除；响应格式：统一包装 + X-Request-Id（PG 模式验证通过）
- [x] **R1-2 E2E 流式断言固化**：`e2e-test.sh` 已用临时文件方式修复 SIGPIPE；Fake 环境含流式/指标/扩容/删除全通过
- [x] **R1-3 服务详情补齐**：`PodStatus`（就绪/期望副本）从 K8s 实时查询，前端详情页展示
- [x] **R1-4 文档一致性**：README/TODO 更新 R1 完成状态与 PG 用法

### R2：单集群生产化（优先级最高）

目标：真实集群上可持续运行一个租户或多个基础租户。

#### R2-1 Postgres 持久化 ✅（R1-0 已完成）
- [x] 引入 `database/sql + lib/pq`，新增 `lib/store/` + 各服务 `data/postgres.go`
- [x] migration 工具（自写 SQL + schema_migrations 表，幂等可重入）
- [x] 表：`models`、`model_versions`、`deployments`、`deployment_events`、`tenant_quotas`、`api_keys`、`audit_logs`
- [x] repository 接口不变，Postgres 实现替换内存；`hack/dev-up.sh --storage=postgres|memory`
- [ ] 审计事件持久化 + 查询 API（audit_logs 表已建，写入逻辑在 R2-4 补）
- [x] 控制面重启恢复：PG 重启后部署状态/事件完整恢复（已验证）

#### R2-2 真实 K8s adapter（informer/对账增强）✅
- [x] 自写 REST client 增强：ListDeployments（按 `carrot.ai/managed-by=carrot` 标签扫描）
- [x] 控制面重启后对账恢复：启动时 reconcileAll + 孤儿检测（reconcileOrphans）
- [x] 孤儿资源回收：本地无记录的 K8s 受管部署 → 告警日志（不自动删除避免误删）
- [x] 删除清理：Deployment/Service 统一 owner label，NotFound 幂等（已有）

#### R2-3 Prometheus + DCGM 指标链路 ✅
- [x] observability 增加 Prometheus adapter（`internal/prometheus` 查询 client），`--prometheus-url` 配置，无 Prometheus 降级内存
- [x] DCGM Exporter DaemonSet 清单 + Prometheus 抓取清单（`deployments/k8s/10/11-*.yaml`）
- [x] 推理指标维度：deployment/model 维度查询（内存聚合）+ GPU 按节点
- [x] 指标查询 API 支持 range 参数（5m/15m/30m/1h/6h/24h）

#### R2-4 租户隔离与安全基线 ✅
- [x] 租户配额落地：QuotaStore 接口（内存/PG）+ QuotaUseCase，创建/扩容校验、删除释放（已验证配额超限 1007 + 释放）
- [x] 配额 API：`GET/PUT /api/v1/quotas`
- [x] API Key 模型授权：`POST /api/v1/keys/{keyId}/models` 白名单，gateway 转发前校验（403）
- [x] 审计日志：部署 create/scale/delete 写入 audit_logs + `GET /api/v1/audit` 查询
- [x] RBAC/ResourceQuota/NetworkPolicy 清单（`deployments/k8s/12-tenant-isolation.yaml`）

#### R2-5 发布流程与可靠性 ✅
- [x] 模型版本状态扩展：`REGISTERED → VALIDATED → RELEASED`（`CanTransitionVersion` 状态机），仅 RELEASED 可部署
- [x] 发布 API：`POST /api/v1/versions/{id}/release`；发布记录审计
- [x] 部署超时：Reconciler STARTING 超时（5m）→ FAILED；DELETING 超时（3m）→ 重试删除
- [x] 失败诊断：FAILED 带 diagnostics；删除重试（RetryDelete）

### R3：推理平台能力（R2 稳定后）✅

- [x] **发布门禁**：`RELEASED` 状态门禁（仅 RELEASED 可部署）+ 发布 API + 审计记录（R2-5 已完成）
- [x] **启动探针**：vLLM renderer 已含 readiness/liveness 探针（/health）；K8s readiness 未通过不进入 RUNNING
- [x] **限流 fallback**：gateway 租户限流（令牌桶）+ 模型授权 403 + 上游不可达 502 诊断
- [x] **灰度/回滚基础**：`POST /api/v1/deployments/{id}/upgrade`（切换版本 + 配额校验 + 滚动更新）
- [x] **推理扩缩容基础**：手动扩缩容（scale API）+ Reconciler 副本核对；HPA 配置清单预留
- [x] **成本/Token 用量**：gateway 统计 tokens/usage，observability 聚合 Token 指标
- [x] **SLO/告警**：Prometheus 告警规则清单（`deployments/k8s/13-alert-rules.yaml`）

### R4：平台扩展（扩展点预留，不实现完整功能）

ROADMAP-V2 §4 明确暂不做：Kubeflow 全家桶、Karmada、自研 GPU scheduler、Agent AIOps、
多运行时同时上线、多集群调度。R4 只做**扩展点预留**，不改变当前领域模型：

- [x] **多运行时扩展点**：`ModelVersion.Runtime` 字段已存在（vLLM 校验）；renderer 按 runtime 分发（当前仅 vLLM）
- [x] **多集群扩展点**：`KubeClient` 接口抽象（Fake/Real 可插拔），多集群通过新增 adapter 实现不改变领域对象
- [x] **pipeline 服务骨架**：`services/pipeline/` 目录 + 发布流水线接口（artifact 校验/基准测试/灰度阶段）+ 默认执行器占位实现
- [ ] **Prefill/Decode 分离、训练 Notebook**：文档级预留（不实现）

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

## 3. 前端跟进

- [x] 新增"租户与配额"页：配额/已用/可用/利用率 + 管理员调整配额（QuotasView）
- [x] 新增"告警与审计"页：审计日志查询/搜索（AuditView）
- [x] 部署向导第 4 步：确认页（服务名/租户/命名空间/幂等键）
- [x] 服务详情页 6 tab：性能/GPU/日志/事件/配置/API 示例（对齐 INTERACTION-PROTOTYPE-V2 §3）
- [x] GPU 资源页三级展开：Pool（型号）→ 节点 → GPU 卡
- [x] API Key 授权模型功能（白名单设置 UI：checkbox 多选模型）
- [x] 异步操作"受理成功 + 轮询"反馈统一（部署向导跳详情页轮询状态）

---

## 4. 暂不做的决定（ROADMAP-V2 §4）

不引入 Kubeflow 全家桶、Karmada、自研 GPU scheduler、Agent AIOps。
多运行时、多集群、训练、Prefill/Decode 分离均属 R4，不在当前优先级。
