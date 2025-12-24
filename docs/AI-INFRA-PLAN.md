# Carrot AI Infra 项目计划

## 1. 项目定位

Carrot 演进为一个基于 Kubernetes 的 AI 推理基础设施平台，将底层 IaaS 的 GPU、CPU、存储和网络资源，转化为上层可以直接消费的模型服务能力。

核心目标不是让用户申请 GPU，而是让用户完成以下操作：

```text
选择模型 → 选择规格 → 部署服务 → 获得 OpenAI API → 查看性能与成本 → 扩缩容/下线
```

项目最终定位：

> 面向企业的 AI 推理控制平面，统一管理 GPU 资源、模型部署、推理路由、租户配额和 AI 可观测性。

## 2. 总体架构

```text
用户 / Web Console / OpenAI API
                │
                ▼
         Carrot AI Control Plane
 ┌────────────────────────────────┐
 │ Model Deployment API            │
 │ Model Registry                  │
 │ Resource Pool & Tenant Quota    │
 │ Inference Gateway & Router     │
 │ Metrics / Cost / Audit API      │
 └───────────────┬────────────────┘
                 │
                 ▼
          Kubernetes Adapter
                 │
 ┌───────────────┼────────────────┐
 │               │                │
 GPU 调度       模型运行时        可观测性
 Volcano/Kueue  vLLM/KServe       Prometheus/DCGM
                 │
                 ▼
          Kubernetes GPU Cluster
```

### 现有模块的演进方向

| 当前模块          | AI Infra 目标职责                             |
|-------------------|-----------------------------------------------|
| ResourceManager   | GPU 资源池、节点能力、租户配额、资源分配      |
| NodeManager       | K8s Node/GPU 状态采集；长期由 Kubernetes 替代 |
| ApplicationMaster | 模型服务 Deployment/Service 生命周期编排      |
| Pipeline          | 模型校验、Benchmark、灰度发布和发布流程       |
| ModelRegistry     | 模型版本、权重地址、运行时和指标管理          |
| Gateway           | OpenAI API、模型路由、鉴权、限流和灰度        |
| Dashboard         | 模型服务、GPU 资源、性能、日志和成本          |

## 3. MVP 范围

第一个版本只实现一条完整闭环：

```text
部署 Qwen/Llama 类模型
    → 创建 vLLM 服务
    → 申请 GPU
    → 暴露 OpenAI 兼容接口
    → 采集 GPU 和推理指标
    → 删除服务并释放资源
```

### MVP 必须支持

- 单 Kubernetes 集群
- 单租户或基础租户标识
- 单一推理运行时：vLLM
- 单一模型存储：S3/MinIO 或 NFS 二选一
- GPU 节点发现和资源展示
- 模型服务创建、查询、扩容和删除
- OpenAI Chat Completions 兼容接口
- Prometheus + NVIDIA DCGM 指标
- 基础 Web Dashboard
- 部署状态机和失败诊断

### MVP 暂不支持

- 自研 GPU 调度器
- 同时支持 vLLM、Triton、TensorRT-LLM
- 多集群调度
- 分布式训练
- Prefill/Decode 分离
- AIOps Agent
- 复杂模型血缘和自动重训练

## 4. 阶段计划

### Phase 0：架构基线与接口设计

目标：统一领域模型和 API，避免后续服务各自定义资源对象。

任务：

- [ ] 明确 Kubernetes 为生产执行底座，保留现有 YARN 实现作为实验适配器
- [ ] 定义 Model、ModelVersion、ModelDeployment、GPUResourcePool、TenantQuota
- [ ] 定义模型服务状态机
- [ ] 定义 REST API 和错误码
- [ ] 定义 Kubernetes 资源标签和注解规范
- [ ] 定义 MVP 的模型、GPU、存储和网关选型
- [ ] 补充架构决策记录（ADR）

交付物：

- API 草案
- 领域模型文档
- 状态机文档
- Kubernetes 资源模板规范

验收标准：

- 一个模型服务从创建到删除的状态和事件均有明确描述
- ResourceManager、ModelRegistry、Gateway 对同一对象使用一致的 ID 和状态

### Phase 1：GPU 资源中心

目标：让平台能够发现、展示和分配 Kubernetes GPU 资源。

任务：

- [ ] 扩展 `Resource`，加入 GPU 数量、GPU 类型和 GPU 显存
- [ ] 扩展 Node 模型，记录 GPU 能力、健康状态和使用量
- [ ] 实现 Kubernetes Node/GPU Discovery
- [ ] 实现 GPU Pool 查询接口
- [ ] 增加节点标签、GPU 型号和资源池过滤
- [ ] 增加租户 GPU 配额模型
- [ ] 接入 Volcano 或 Kueue 的队列/配额能力
- [ ] 增加 GPU 资源分配和释放测试

建议模型：

```go
type Resource struct {
MemoryMB    int64
VCores      int32
GPUCount    int32
GPUType     string
GPUMemoryMB int64
}
```

验收标准：

- API 能返回 GPU 节点、型号、显存和健康状态
- 指定 GPU 类型和数量的部署请求能够被调度
- 删除服务后 GPU 配额正确释放

### Phase 2：模型注册中心

目标：统一管理可部署模型及其版本。

任务：

- [ ] 完善 `services/modelregistry`
- [ ] 实现模型注册、查询和删除
- [ ] 实现模型版本管理
- [ ] 保存模型权重 URI、框架、运行时、显存需求和上下文长度
- [ ] 支持模型可部署性校验
- [ ] 增加模型访问权限
- [ ] 增加 S3/MinIO/NFS 存储适配接口
- [ ] 为模型版本增加 Benchmark 指标字段

核心对象：

```go
type ModelVersion struct {
ModelName     string
Version       string
ArtifactURI   string
Runtime       string
GPUType       string
GPUCount      int32
MemoryMB      int64
ContextLength int32
Status        string
}
```

验收标准：

- 可以注册一个模型版本并查询完整元数据
- 部署请求只能引用已注册且可部署的模型版本
- 模型存储地址不会直接暴露为未授权下载地址

### Phase 3：Kubernetes Adapter 与模型部署

目标：将平台部署请求转换为 Kubernetes 工作负载。

任务：

- [ ] 新增 Kubernetes Client 封装
- [ ] 实现 Namespace、Secret、ConfigMap、Deployment、Service 管理
- [ ] 实现 vLLM Deployment Renderer
- [ ] 注入模型路径、启动参数和环境变量
- [ ] 注入 GPU 资源限制和节点选择器
- [ ] 注入健康检查和优雅终止配置
- [ ] 实现 Deployment 状态同步
- [ ] 实现失败原因和 Pod 事件采集
- [ ] 实现创建、查询、扩容、重启和删除
- [ ] 增加幂等性和断线重试

核心 API：

```text
POST   /api/v1/deployments
GET    /api/v1/deployments
GET    /api/v1/deployments/{id}
POST   /api/v1/deployments/{id}/scale
DELETE /api/v1/deployments/{id}
```

验收标准：

- 创建请求能够生成可运行的 vLLM Deployment
- 平台能同步 Pending、Running、Failed、Deleting 等状态
- 删除请求能清理相关 Kubernetes 资源
- 控制面重启后能够从 Kubernetes 恢复服务状态

### Phase 4：AI Gateway 与推理路由

目标：为所有模型服务提供统一 OpenAI 兼容入口。

任务：

- [ ] 实现 `/v1/chat/completions`
- [ ] 实现 `/v1/models`
- [ ] 根据 `model` 字段进行模型路由
- [ ] 支持流式响应
- [ ] 支持 API Key 鉴权
- [ ] 增加租户级限流
- [ ] 增加请求超时、重试和错误转换
- [ ] 记录请求、模型、租户和 Token 元数据
- [ ] 预留灰度和 fallback 路由配置

验收标准：

- 用户无需感知后端 Pod 地址即可调用模型
- 同一模型多个副本可以负载均衡
- 未授权租户不能访问其他租户模型
- 流式请求的首 Token 和完整响应均能正确返回

### Phase 5：AI 可观测性与成本

目标：从 Kubernetes 监控升级为 GPU、推理和业务三层监控。

任务：

- [ ] 接入 Prometheus 或 VictoriaMetrics
- [ ] 接入 NVIDIA DCGM Exporter
- [ ] 采集 GPU 利用率、显存、温度、功耗和错误
- [ ] 采集请求量、错误率、TTFT、TPOT、Token/s
- [ ] 采集 KV Cache、队列长度和副本状态
- [ ] 采集输入/输出 Token 和租户用量
- [ ] 建立模型服务 Dashboard
- [ ] 建立 GPU 资源池 Dashboard
- [ ] 定义 GPU 分钟和 Token 成本计算规则
- [ ] 增加日志关联和请求 Trace ID

验收标准：

- 每个模型服务都能查看请求、延迟、吞吐和错误率
- 每个 GPU 都能查看利用率、显存和健康状态
- 指标能够按租户、模型和时间范围过滤
- 一次请求可以关联到网关日志、模型服务和 GPU 指标

### Phase 6：Pipeline、评测与模型发布

目标：把模型部署扩展为可控的发布流程。

任务：

- [ ] 完善 Pipeline DAG 定义和拓扑排序
- [ ] 增加模型下载和格式校验 Stage
- [ ] 增加启动 Smoke Test Stage
- [ ] 增加吞吐/延迟 Benchmark Stage
- [ ] 增加安全和配置检查 Stage
- [ ] 增加 Staging/Production 环境
- [ ] 增加灰度发布和回滚
- [ ] 将 Pipeline 结果写回 ModelRegistry
- [ ] 增加发布审批和审计事件

推荐流程：

```text
REGISTERED
    → VALIDATING
    → BENCHMARKING
    → STAGING
    → PRODUCTION
    → ARCHIVED
```

验收标准：

- 模型发布前必须通过基础校验和 Smoke Test
- Benchmark 结果可追溯到具体模型版本和 GPU 规格
- 生产发布失败可以自动回滚到上一版本

### Phase 7：生产化能力

目标：满足多租户和生产环境要求。

任务：

- [ ] Namespace、RBAC 和 ResourceQuota
- [ ] 租户级 GPU 配额和优先级
- [ ] Secret 管理和模型访问控制
- [ ] NetworkPolicy 和服务隔离
- [ ] 审计日志
- [ ] 控制面高可用
- [ ] 数据持久化和备份
- [ ] 限流、熔断和故障恢复
- [ ] OpenCost/Kubecost 成本接入
- [ ] 压测和容量评估

验收标准：

- 不同租户之间资源、模型和请求相互隔离
- 控制面重启不丢失部署状态
- 单个模型服务故障不会影响其他模型服务
- 能够给出单租户、单模型和单 Token 的成本估算

## 5. 当前迭代任务拆分

第一轮只做 Phase 0 和 Phase 1 的最小闭环，任务按以下顺序执行：

### Sprint 1：领域模型与接口

- [ ] 定义 `GPUResource`、`ModelDeployment`、`DeploymentStatus`
- [ ] 扩展共享 `Resource` 类型
- [ ] 定义部署 REST API 请求/响应
- [ ] 定义错误码和状态转换
- [ ] 增加领域模型单元测试

### Sprint 2：Kubernetes 资源发现

- [ ] 增加 Kubernetes Client 配置
- [ ] 查询 Node 和 GPU 资源
- [ ] 转换为 Carrot 资源模型
- [ ] 增加 GPU Pool API
- [ ] 增加本地 Fake Kubernetes Client
- [ ] 增加资源发现集成测试

### Sprint 3：第一个模型服务

- [ ] 注册一个模型版本
- [ ] 生成 vLLM Deployment
- [ ] 创建 Service
- [ ] 同步 Deployment 状态
- [ ] 返回服务 Endpoint
- [ ] 完成创建/查询/删除端到端测试

### Sprint 4：可观测性和 Dashboard

- [ ] 接入 DCGM/Prometheus 指标
- [ ] 增加模型服务状态页面
- [ ] 增加 GPU 资源页面
- [ ] 增加基础请求延迟和 Token 指标
- [ ] 完成 MVP 演示脚本

## 6. 非功能要求

### 可靠性

- 所有创建和删除操作必须幂等
- 控制面重启后必须能够重新同步 Kubernetes 状态
- 外部 API 调用必须有超时和重试上限

### 安全性

- 模型权重地址和 API Key 不得写入日志
- Kubernetes 凭证通过 Secret 或外部 Secret 管理
- 模型服务必须按租户隔离
- 网关请求必须带审计信息

### 可测试性

- 领域逻辑不依赖 Kubernetes Client
- Kubernetes Adapter 使用接口隔离并提供 Fake 实现
- 至少包含单元测试、Adapter 测试和端到端测试

### 可观测性

- 控制面所有请求带 Request ID
- Deployment 状态变化产生审计事件
- 关键操作记录耗时、结果和错误原因

## 7. MVP 最终验收场景

```text
1. 启动 Kubernetes GPU 集群
2. Carrot 发现 GPU 节点并展示资源
3. 注册 Qwen 模型版本
4. 创建一个 1 GPU 的 vLLM Deployment
5. 服务进入 Running 状态
6. 通过 /v1/chat/completions 发起请求
7. Dashboard 展示 TTFT、Token/s、GPU 利用率
8. 扩容到 2 个副本
9. 删除服务并确认 GPU 资源释放
10. 控制面重启后服务状态仍然正确
```

只有以上场景稳定完成后，再进入多租户、自动扩缩容、模型评测和多集群阶段。

## 8. 项目成功指标

- 10 分钟内完成一个模型服务部署
- 模型服务部署成功率达到 95% 以上
- 控制面重启后服务状态恢复正确率达到 100%
- 能够展示 GPU 利用率和推理延迟
- 模型服务支持标准 OpenAI API
- 删除服务后资源无泄漏
- 至少支持两个模型版本的灰度或回滚

