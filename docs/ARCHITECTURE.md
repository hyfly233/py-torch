# Carrot AI Infra 架构设计

## 1. 设计目标

- 以 Kubernetes 作为生产执行底座
- 控制面与执行面解耦
- 领域逻辑不依赖 Kubernetes SDK
- 模型服务生命周期可恢复、可审计、可重试
- 为多租户、多运行时和多集群预留扩展点

## 2. 逻辑分层

```text
┌─────────────────────────────────────────────┐
│ Presentation Layer                           │
│ REST API / OpenAI API / Dashboard            │
├─────────────────────────────────────────────┤
│ Application Layer                            │
│ Deployment UseCase / Model UseCase / Gateway │
├─────────────────────────────────────────────┤
│ Domain Layer                                 │
│ Model / Deployment / Resource / Quota        │
├─────────────────────────────────────────────┤
│ Adapter Layer                                │
│ Kubernetes / Prometheus / S3 / vLLM          │
├─────────────────────────────────────────────┤
│ Infrastructure                               │
│ Kubernetes Cluster / GPU / Storage / Network │
└─────────────────────────────────────────────┘
```

## 3. 服务边界

### Control Plane

负责：

- 接收模型注册和部署请求
- 校验资源和租户配额
- 创建 Deployment 任务
- 同步 Kubernetes 状态
- 暴露查询和审计 API

### Kubernetes Adapter

负责：

- 将领域对象转换为 Kubernetes 资源
- 创建、更新、删除和查询资源
- 监听 Pod、Deployment 和事件
- 屏蔽 Kubernetes SDK 细节

### Model Registry

负责：

- 模型和版本元数据
- 权重 URI 和运行时配置
- 模型可部署性状态
- Benchmark 和发布结果

### Inference Gateway

负责：

- OpenAI 兼容 API
- API Key 和租户鉴权
- 按模型路由到 Service
- 限流、超时、重试和访问日志

### Observability

负责：

- GPU、Kubernetes 和推理指标采集
- 服务指标聚合
- Dashboard 查询接口
- 用量和成本估算

## 4. 推荐目录结构

```text
services/
├── controlplane/
│   ├── cmd/
│   └── internal/
│       ├── biz/
│       │   ├── deployment.go
│       │   ├── resource.go
│       │   └── quota.go
│       ├── data/
│       ├── server/
│       └── worker/
├── k8sadapter/
│   ├── internal/
│   │   ├── client/
│   │   ├── renderer/
│   │   ├── informer/
│   │   └── discovery/
├── modelregistry/
├── gateway/
├── inference/
├── pipeline/
└── observability/
```

现阶段也可以复用已有 `resourcemanager` 和 `applicationmaster`，但新增代码应通过接口隔离，避免继续扩大核心服务的职责。

## 5. 核心领域对象

```go
type Model struct {
    ID          string
    Name        string
    Description string
}

type ModelVersion struct {
    ID            string
    ModelID       string
    Version       string
    ArtifactURI   string
    Runtime       string
    GPUType       string
    GPUCount      int32
    MemoryMB      int64
    ContextLength int32
    Status        string
}

type ModelDeployment struct {
    ID            string
    Name          string
    ModelVersionID string
    TenantID      string
    Namespace     string
    Replicas      int32
    Resource      Resource
    Endpoint      string
    Status        string
    Generation    int64
}

type GPUResource struct {
    NodeName     string
    GPUType      string
    Total        int32
    Allocatable  int32
    Used         int32
    MemoryMB     int64
    Health       string
}
```

## 6. 状态机设计

### ModelDeployment

```text
NEW
  ↓
VALIDATING ──→ FAILED
  ↓
SUBMITTING ──→ FAILED
  ↓
STARTING ────→ FAILED
  ↓
RUNNING
  ├── SCALING → RUNNING
  ├── RESTARTING → RUNNING
  └── DELETING → DELETED
```

状态变更必须记录：

- 变更前状态
- 变更后状态
- 触发原因
- Request ID
- Kubernetes 资源版本
- 错误诊断

## 7. 部署数据流

```text
Client
  │ POST /deployments
  ▼
Control Plane
  │ 1. 校验模型版本
  │ 2. 校验租户配额
  │ 3. 选择 GPU Pool
  │ 4. 写入 Deployment 状态
  ▼
Kubernetes Adapter
  │ 创建 Namespace/Secret/Deployment/Service
  ▼
Kubernetes
  │ 调度 GPU、启动 vLLM、创建 Pod
  ▼
Informer / Reconciler
  │ 同步 Pod 和 Deployment 状态
  ▼
Control Plane
  │ 更新状态、Endpoint 和诊断信息
  ▼
Gateway
  │ 根据 model 路由请求
  ▼
vLLM Service
```

## 8. Reconciler 设计

部署状态同步采用期望状态与实际状态对账，而不是只依赖创建请求的同步返回。

```text
期望状态：Deployment replicas=2
实际状态：Deployment replicas=1
            │
            ▼
       Reconciler
            │
     更新 Kubernetes 资源
            │
     等待实际状态收敛
```

要求：

- Reconcile 必须幂等
- 支持控制面重启后重新同步
- 对 Kubernetes 临时错误进行有限重试
- 对永久错误进入 Failed 并保留诊断
- 删除操作必须处理已不存在资源的情况

## 9. Kubernetes 资源约定

每个模型服务的资源统一使用以下标签：

```text
carrot.ai/deployment-id
carrot.ai/model-id
carrot.ai/model-version
carrot.ai/tenant-id
carrot.ai/managed-by=carrot
```

建议的 Deployment 资源：

- Namespace
- Secret
- ConfigMap
- Deployment
- Service
- PodDisruptionBudget
- ServiceMonitor（接入 Prometheus 时）

GPU 资源通过标准 Kubernetes `limits` 注入：

```yaml
resources:
  limits:
    nvidia.com/gpu: "1"
```

## 10. 存储设计

### MVP

- 模型元数据：内存或 PostgreSQL
- 模型权重：S3/MinIO 或 NFS
- 控制面状态：PostgreSQL
- 日志：容器标准输出 + Kubernetes 日志

### 后续

- Harbor OCI Artifact
- JuiceFS/Fluid 模型缓存
- 模型分片和预热
- 产物与模型版本关联

## 11. API 设计

### 控制面 API

```text
POST   /api/v1/models
GET    /api/v1/models
POST   /api/v1/models/{id}/versions
GET    /api/v1/deployments
POST   /api/v1/deployments
GET    /api/v1/deployments/{id}
POST   /api/v1/deployments/{id}/scale
DELETE /api/v1/deployments/{id}
GET    /api/v1/resources/gpus
GET    /api/v1/deployments/{id}/metrics
```

### 推理 API

```text
GET  /v1/models
POST /v1/chat/completions
```

## 12. 可靠性和安全设计

### 可靠性

- 创建、删除、扩缩容操作使用幂等键
- 所有外部调用设置超时
- 状态同步使用重试和退避
- 控制面重启后通过 Informer/列表查询恢复状态
- 关键状态落库，不能只保存在内存

### 安全

- API Key 哈希存储
- Kubernetes 凭证使用 Secret
- 按租户隔离 Namespace 和 ResourceQuota
- 模型权重访问使用临时凭证
- Gateway、控制面和模型服务之间使用最小权限
- 审计部署、删除、扩容、API Key 操作

## 13. 演进路线

```text
MVP 单集群 vLLM
    ↓
多租户与配额
    ↓
自动扩缩容与成本
    ↓
模型评测和灰度发布
    ↓
Triton/KServe 等多运行时
    ↓
多集群和跨地域调度
    ↓
AIOps Agent
```

