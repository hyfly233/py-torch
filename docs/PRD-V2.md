# Carrot AI Infra PRD（V2：单集群推理云）

## 1. 产品定义

Carrot 将 GPU、存储、网络等 IaaS 资源封装为可消费的模型服务。用户购买/使用的不是“几张 GPU”，而是一个带 OpenAI 兼容 Endpoint、配额、指标和生命周期管理的模型服务。

核心闭环：`选择已发布模型 → 预估资源 → 部署 → 调用 → 观察 → 扩缩/回滚/下线`。

## 2. 目标用户与关键任务

| 用户 | 首要任务 | 成功标准 |
|---|---|---|
| AI 工程师 | 快速发布模型 | 10 分钟内拿到可调用 Endpoint |
| 应用开发者 | 稳定调用模型 | 只需 API Key，不感知 Pod/GPU |
| 平台管理员 | 管理资源和租户 | 配额、隔离、利用率可解释 |
| SRE | 定位异常 | 从服务指标跳到事件、日志和资源 |

## 3. MVP 与 R2 范围

MVP：单集群、vLLM、模型注册/校验、GPU 发现、部署/扩缩/删除、OpenAI Chat Completions、流式、基础指标。

R2：Postgres、真实 K8s、Prometheus/DCGM、租户/RBAC/配额、审计、重启恢复、失败诊断和孤儿资源回收。

非目标：训练平台、多集群、复杂计费、自研调度器、AIOps Agent、多运行时同时上线。

## 4. 功能要求

### 模型与发布

- 模型版本必须包含 artifact URI、runtime、GPU 类型/数量、显存、上下文长度
- 只有 `VALIDATED/RELEASED` 版本可部署
- 发布记录校验结果、基准指标、操作者和时间

### 服务生命周期

- 创建请求支持幂等键；同名服务策略明确
- 状态：`NEW → VALIDATING → SUBMITTING → STARTING → RUNNING`，异常进入 `FAILED`，操作态支持 `SCALING/RESTARTING/DELETING`
- 每次状态变化记录 reason、request ID、诊断和 Kubernetes resource version

### 调用与观测

- `/v1/models`、`/v1/chat/completions`，支持 SSE
- API Key 只明文展示一次，按租户/模型授权
- 展示 QPS、错误率、TTFT、TPOT、Token/s、GPU 利用率/显存、队列长度
- 异常指标可跳转到部署事件和 Pod 诊断

## 5. 产品指标与验收

- 部署受理 P95 < 3 秒；启动耗时可观测
- 控制面重启后部署状态恢复率 100%
- 无效 Key、跨租户访问、超配额请求均被拒绝
- 删除后 GPU 配额最终释放，孤儿 K8s 资源可发现
- MVP E2E 全链路通过；R2 在真实 K8s 完成故障注入验收
