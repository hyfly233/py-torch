# kk-infra — Carrot AI Infra 分布式微服务平台 TODO

> 依据：`docs/PRD.md`、`docs/AI-INFRA-PLAN.md`、`docs/ARCHITECTURE.md`、`docs/INTERACTION-PROTOTYPE.md`
> 目标：从零搭建一个基于 Kubernetes 的 AI 推理基础设施平台，Go 多模块 workspace（`go.work`），本地可用 Fake K8s 跑通 MVP 闭环。
> 工具链：Go 1.26.5（PATH 默认 1.20.4，必须用 `/opt/homebrew/bin/go`）。

---

## 0. 当前状态（2026-08-01）

### 已完成 ✅

| 阶段 | 内容 | 提交 |
|---|---|---|
| P0 | lib 共享库：领域模型/状态机/错误码/中间件/API 类型 | `d2c6828` |
| P1 | modelregistry：模型/版本 CRUD/校验/唯一性 | `d4aa3dc` |
| P2 | k8sadapter：KubeClient 接口/Fake/vLLM renderer/真实 K8s 客户端 | `14bb800` `f9589aa` `7d65d04` |
| P3 | controlplane：部署生命周期/状态机/GPU 资源/Reconciler | `0d751de` |
| P4 | gateway + inference：OpenAI 兼容 API/API Key/路由/流式 | `b544181` |
| P5 部分 | observability 服务 + 指标链路打通（gateway 埋点/controlplane 转发/GPU 采集） | `788d7ce` |
| 前端 | 全部页面 + 角色权限 + 路由守卫 + 脚手架清理 | `5ac3845` |

### 本轮修复的 Bug

- **前端角色判定失效**：`useAuth()` 返回嵌套 computed ref，模板 `auth.isAdmin` 不自动解包恒为 truthy。修复：视图解构 `const { isAdmin } = useAuth()`；另发现 Dashboard 部署按钮硬编码无权限判断，一并修复。
- **部署幂等 + 同名重建**：`CreateDeployment` 按 Name 幂等，但删除仅标记 DELETED 不删记录 → 同名部署删除后无法重建。修复：biz 幂等查询跳过 DELETED；repo.Create 允许 DELETED 同名覆盖 byName。

### 剩余待办 ⬜

1. **e2e 流式调用验证**：脚本 `curl|head -c` 在 pipefail 下 SIGPIPE 退出（exit=23），需修复断言方式并确认 gateway 流式透传真实链路正常
2. **README 编写**（根目录：架构/启动/API/演示）
3. 后续轮骨架（pipeline 目录、Postgres repo、真实 Prometheus/DCGM 采集）——本轮不做

---

## 1. 目标与范围

### 本轮目标（MVP 闭环）

```text
发现 GPU → 注册模型 → 创建 vLLM 服务 → 获取 Endpoint
→ OpenAI API 调用（含流式）→ 查看 TTFT/GPU 指标 → 扩容 → 删除并释放资源
```

### 明确不做（后续轮次）

- 真实多集群调度、自研 GPU 调度器
- 多推理运行时（Triton/TensorRT-LLM，仅 vLLM）
- 复杂自动扩缩容、完整计费结算
- 生产化（RBAC/NetworkPolicy/高可用/备份）——留接口与骨架
- 模型训练

---

## 2. 总体架构

```text
用户 / Web Console / OpenAI API
                │
                ▼
┌─────────────────────────────┐
│ controlplane 控制面 :8080    │  REST API / 部署 UseCase / 状态机 / Reconciler
├─────────────────────────────┤
│ modelregistry :8081         │  模型 & 版本元数据 / 可部署性校验
├─────────────────────────────┤
│ k8sadapter :8082            │  K8s Client / vLLM Renderer / Discovery / Fake
├─────────────────────────────┤
│ gateway :8083               │  OpenAI 兼容 API / 路由 / 鉴权 / 流式
├─────────────────────────────┤
│ observability :8084         │  指标聚合 / 查询 API（MVP 内存，预留 Prometheus）
├─────────────────────────────┤
│ inference :8085             │  Mock vLLM 推理后端（本地闭环用；生产为真实 vLLM）
└─────────────────────────────┘
```

### 服务依赖

| 服务 | 依赖 | 端口 | 职责 |
|---|---|---|---|
| `lib` | 无 | - | 领域模型、状态机、错误码、API 类型、中间件（共享，非独立进程） |
| `controlplane` | lib, modelregistry, k8sadapter | 8080 | 部署/模型/资源 UseCase、状态机、Reconciler、REST API |
| `modelregistry` | lib | 8081 | 模型与版本 CRUD、可部署性校验、Benchmark 字段预留 |
| `k8sadapter` | lib | 8082 | K8s 资源转换（renderer）、Node/GPU 发现、状态同步、Fake 实现 |
| `gateway` | lib, modelregistry(可选) | 8083 | `/v1/models`、`/v1/chat/completions`、API Key、路由、限流、流式 |
| `observability` | lib | 8084 | GPU/推理指标聚合与查询（MVP 内存存储 + 接口预留 Prometheus） |
| `inference` | lib | 8085 | Mock OpenAI 兼容后端（流式 + TTFT 模拟），本地 E2E 用 |

### 通信方式

- 服务间通信：HTTP/JSON（MVP 简单可靠；后续可升级 gRPC）
- 每个服务独立进程、独立 go.mod，`go.work` 统一编排
- 控制面重启恢复：`data` 层落内存 + 启动时从 k8sadapter 重新对账（Fake 模式同样支持）

---

## 3. 目录结构（待创建）

```text
kk-infra/
├── go.work                          # 已存在，加入各模块
├── TODO.md                          # 本文件
├── lib/                             # 共享库模块
│   ├── go.mod                       # module kk-infra/lib
│   ├── domain/                      # 领域模型
│   │   ├── model.go                 # Model / ModelVersion
│   │   ├── deployment.go            # ModelDeployment / DeploymentStatus
│   │   ├── resource.go              # Resource / GPUResource
│   │   ├── quota.go                 # TenantQuota
│   │   └── statemachine.go          # 部署状态机 + 事件记录
│   ├── errcode/                     # 错误码定义
│   ├── apitypes/                    # REST 请求/响应类型
│   └── middleware/                  # RequestID / 日志 / 恢复
├── services/
│   ├── controlplane/
│   │   ├── go.mod                   # module kk-infra/services/controlplane
│   │   ├── cmd/main.go
│   │   └── internal/
│   │       ├── biz/                 # deployment.go / resource.go / model.go / quota.go
│   │       ├── data/                # repository 接口 + 内存实现
│   │       ├── server/              # HTTP handler / router
│   │       └── worker/              # reconciler 对账循环
│   ├── modelregistry/
│   │   ├── go.mod
│   │   ├── cmd/main.go
│   │   └── internal/
│   │       ├── biz/
│   │       ├── data/
│   │       └── server/
│   ├── k8sadapter/
│   │   ├── go.mod
│   │   ├── cmd/main.go
│   │   └── internal/
│   │       ├── client/              # KubeClient 接口 + fake_kube.go（真实 client-go 后续轮）
│   │       ├── renderer/            # vLLM Deployment/Service/Secret/ConfigMap 渲染
│   │       ├── discovery/           # Node/GPU 发现 → GPUResource
│   │       ├── informer/            # Deployment/Pod 状态同步
│   │       └── server/              # HTTP API（供 controlplane 调用）
│   ├── gateway/
│   │   ├── go.mod
│   │   ├── cmd/main.go
│   │   └── internal/
│   │       ├── auth/                # API Key 管理（哈希存储）
│   │       ├── router/              # model → 后端 endpoint 路由
│   │       ├── proxy/               # OpenAI 代理 + 流式转发 + 限流/超时
│   │       └── server/
│   ├── observability/
│   │   ├── go.mod
│   │   ├── cmd/main.go
│   │   └── internal/
│   │       ├── metrics/             # 内存指标存储 + 聚合
│   │       ├── collector/           # 从 gateway/k8sadapter 拉取指标（接口，MVP 直接注入）
│   │       └── server/
│   ├── inference/                   # Mock vLLM
│   │   ├── go.mod
│   │   ├── cmd/main.go
│   │   └── internal/server/         # OpenAI 兼容 chat completions（含流式 + TTFT）
│   └── pipeline/                    # 骨架（Phase 6，本轮只建目录与 README）
└── hack/                            # 本地一键启动脚本 / E2E 脚本
    ├── dev-up.sh                    # 依次启动全部服务
    └── e2e-test.sh                  # 跑 MVP 验收场景
```

---

## 4. 技术选型

| 项 | 选择 | 理由 |
|---|---|---|
| 语言/工具链 | Go 1.26.5（`/opt/homebrew/bin/go`） | 仓库已声明 go 1.26 |
| HTTP 框架 | 标准库 `net/http`（Go 1.22+ 方法路由） | 零依赖、够用；拒绝过度设计 |
| 存储（MVP） | 内存 repository（`sync.RWMutex`）+ 接口 | docs 允许内存/PostgreSQL；接口预留 Postgres 后续轮 |
| K8s 客户端 | 自定义 `KubeClient` 接口 + Fake 实现 | 本地无集群；docs 明确要求接口隔离 + Fake |
| vLLM 渲染 | 自写 renderer 输出 K8s manifest 结构体 | 不引 client-go，纯数据结构 + JSON/YAML |
| Mock 推理 | 自写 OpenAI 兼容 server | 本地闭环必须；流式 + TTFT 可验证 |
| 日志 | `log/slog`（标准库） | 零依赖；RequestID 中间件注入 |
| 测试 | 标准库 `testing` + httptest | 领域单测 + 服务集成 + E2E 脚本 |

---

## 5. 任务拆分（P0 → P5）

### P0 — 工程骨架与共享库

- [x] 初始化 `lib` 模块：`domain`（Model/ModelVersion/ModelDeployment/Resource/GPUResource/TenantQuota/DeploymentStatus）
- [x] `statemachine.go`：部署状态机（NEW→VALIDATING→SUBMITTING→STARTING→RUNNING / FAILED，RUNNING 可 SCALING/RESTARTING/DELETING→DELETED），状态变更事件记录（前状态/后状态/原因/RequestID/诊断）
- [x] `errcode`：统一错误码（资源不足/模型不存在/配额不足/非法状态转换等）
- [x] `middleware`：RequestID、访问日志（脱敏：不打印权重地址/API Key）、panic 恢复
- [x] 领域模型单元测试（状态机合法/非法转换、错误码）
- [x] `go.work` 加入 `./lib`

### P1 — modelregistry 模型注册中心

- [x] 模型/版本 CRUD API：`POST/GET /api/v1/models`、`POST /api/v1/models/{id}/versions`、详情/删除
- [x] 字段：ArtifactURI、Runtime(vLLM)、GPUType、GPUCount、MemoryMB、ContextLength、Status
- [x] 可部署性校验：版本状态 ∈ {REGISTERED, VALIDATED} 且必填字段完整才可部署
- [x] 内存 repository + 单元测试
- [x] 名称+版本唯一性约束、删除保护（被部署引用的版本不可删）

### P2 — k8sadapter 适配器（Fake 优先）

- [x] `KubeClient` 接口：ListNodes / GetGPUResources / CreateDeployment / GetDeployment / ScaleDeployment / DeleteDeployment / ListPods / ListEvents
- [x] Fake 实现：内置 2 节点 × 8 卡 A100 GPU 池；模拟 Pending→Starting→Running 生命周期、扩缩容、删除；支持注入失败场景（测试用）
- [x] renderer：把 ModelDeployment → vLLM Deployment/Service/Secret/ConfigMap manifest（`nvidia.com/gpu` limits、模型启动参数、健康检查、`carrot.ai/*` 标签）
- [x] discovery：Node/GPU → `GPUResource`（型号/总量/可用/已用/健康）
- [x] HTTP API：`GET /v1/resources/gpus`、部署/查询/扩容/删除、状态同步
- [x] renderer 单测（labels/resources/参数注入断言）

### P3 — controlplane 控制面（核心闭环）

- [x] `biz/deployment.go`：创建部署（校验模型版本→校验配额→选 GPU Pool→写状态→调 k8sadapter）、查询、扩缩容、重启、删除（幂等）
- [x] `biz/resource.go`：GPU 汇总（总量/可用/已用/异常）、按型号筛选
- [x] `biz/quota.go`：租户配额模型 + 校验（MVP 单租户 default）
- [x] `data`：部署 repository（内存，接口化）+ 启动时对账恢复
- [x] `server`：REST API `POST/GET /api/v1/deployments`、`GET /api/v1/deployments/{id}`、`POST .../scale`、`DELETE .../deployments/{id}`、`GET /api/v1/resources/gpus`、`GET /api/v1/deployments/{id}/metrics`
- [x] `worker/reconciler`：周期性对账（期望 vs 实际副本、状态收敛、失败诊断 + 事件采集）
- [x] 幂等性：同一 deployment 创建请求重复提交不重复建资源；删除不存在资源返回成功（幂等删除）

### P4 — gateway + inference mock 后端

- [x] `inference`：Mock OpenAI 兼容服务（`/v1/models`、`/v1/chat/completions` 非流式 + 流式 SSE，模拟 TTFT/Token 输出）
- [x] `gateway/auth`：API Key 创建（明文仅返回一次）、哈希存储、禁用/轮换、租户绑定
- [x] `gateway/router`：`model` 字段 → 后端 endpoint（从 controlplane 查询或配置注入）
- [x] `gateway/proxy`：`/v1/models`、`/v1/chat/completions`，流式透传，租户限流（简单令牌桶）、超时、错误转换
- [x] 网关请求埋点：请求数/延迟/Token/错误 → 上报 observability（接口注入，MVP 直连内存）
- [x] 集成测试：mock 后端 + gateway 全链路（含流式首 Token 断言）

### P5 — observability + E2E 验收

- [x] `observability/metrics`：内存指标存储（QPS、错误率、TTFT、Token/s、GPU 利用率），按服务/副本/时间过滤
- [x] 采集：gateway 请求指标 + k8sadapter GPU 指标（MVP 直接注入/拉取，预留 Prometheus adapter 接口）
- [x] `GET /api/v1/deployments/{id}/metrics?range=...` 查询 API
- [x] `hack/dev-up.sh`：一键启动 6 个服务（或 `go run` 各 cmd）
- [x] `hack/e2e-test.sh`：按 MVP 验收场景逐条断言（发现 GPU→注册模型→创建服务→Running→OpenAI 调用（含流式）→查指标→扩容→删除→资源释放）
- [x] README：架构图、启动方法、API 列表、演示步骤

### 后续轮（骨架/预留，不实现）

- [ ] `pipeline/` 目录 + README（Phase 6：校验/Benchmark/灰度发布）
- [ ] Postgres repository 实现
- [ ] 真实 client-go 适配器 + Prometheus/DCGM 采集
- [ ] 多租户隔离、生产化（RBAC/Quota/NetworkPolicy）

---

## 6. 验收标准（对应 PRD §7 + AI-INFRA-PLAN §7）

```text
1. Fake K8s 启动，发现 GPU 节点（2×8 A100）并展示资源
2. 注册 qwen-7b 模型版本
3. 创建 1 GPU vLLM Deployment
4. 服务进入 Running
5. /v1/chat/completions 调用成功（非流式 + 流式）
6. 指标 API 返回 TTFT/Token/GPU 利用率
7. 扩容到 2 副本成功
8. 删除服务后 GPU 配额释放
9. 控制面重启后服务状态从 k8sadapter 对账恢复
10. 无效 API Key 返回明确错误
```

---

## 8. 前端管理页面（frontend/，Vue 3 + Vite + TS）

> 对应 INTERACTION-PROTOTYPE.md 的信息架构。区分两种角色：管理员 / 普通用户。

### 角色权限矩阵

| 页面/操作 | 普通用户 | 管理员 |
|---|---|---|
| 总览 Dashboard | ✅ 查看 | ✅ 查看 |
| 模型列表/详情 | ✅ 只读 | ✅ 管理（注册/版本/校验/删除） |
| 模型服务列表/详情 | ✅ 只读 | ✅ 管理（创建/扩容/重启/删除） |
| GPU 资源 | ✅ 查看 | ✅ 查看+筛选 |
| API Key | ❌ | ✅ 创建/禁用/轮换 |
| 部署向导 | ❌ | ✅ |
| 系统设置 | ❌ | ✅（占位） |

### 页面清单

- [x] 登录页（角色选择：管理员/普通用户，MVP 本地模拟不接真实认证）
- [x] 总览 Dashboard：GPU 总量/可用/在线服务/异常 + 最近部署列表
- [x] 模型管理：列表 + 详情（版本 tab）+ 注册模型/版本向导 + 校验
- [x] 模型服务：列表 + 详情（状态/事件/API 示例）+ 创建部署向导（选择模型→配置 GPU→配置服务）+ 扩容/重启/删除
- [x] GPU 资源页：汇总卡片 + 节点表格 + 型号筛选
- [x] API Key 管理（管理员）：创建（明文展示一次）/禁用/轮换
- [x] 路由守卫：按角色控制页面访问

### 技术选型

- 路由：vue-router（npm 镜像安装）
- HTTP：原生 fetch 封装（不引 axios）
- 状态：轻量 composable（不引 Pinia）
- UI：手写 CSS 管理后台风格（深色主题），不引 UI 框架
- API 对接：controlplane :8080 / modelregistry :8081 / gateway :8083

### 后端对接补充

- [x] vite proxy：`/api` → controlplane，`/model-registry` → modelregistry，`/gateway` → gateway
- [x] 前端类型定义与后端 apitypes 对齐（手写 TS interface）

---

## 7. 实现顺序建议（后端）

```text
P0 (lib) → P1 (modelregistry) → P2 (k8sadapter) → P3 (controlplane)
→ P4 (gateway + inference mock) → P5 (observability + E2E)
```

每批完成即 `go build ./...` + `go test ./...` 验证，全部通过再进下一批。
