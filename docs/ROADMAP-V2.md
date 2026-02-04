# Carrot AI Infra 路线计划（V2）

> 依据当前代码与“AI 推理连接 IaaS/PaaS”的产品方向。执行基线：单集群、vLLM、OpenAI 兼容 API；Fake K8s 只用于开发验收，不能作为生产能力。

## 1. 当前判断

仓库已经完成一个可演示的 MVP 闭环：`模型注册 → GPU 资源检查 → 部署生命周期 → OpenAI API → 指标 → 扩缩容/删除`。现有实现的关键边界是：内存存储、Fake K8s/Mock 推理、单租户、基础路由和指标聚合。

因此下一阶段不应继续横向增加组件，而应补齐生产闭环的四个缺口：持久化与恢复、真实 Kubernetes/Prometheus、租户隔离与配额、可诊断的发布流程。

## 2. 分阶段路线

### R1：MVP 收口（当前）

目标：让本地演示和代码契约稳定。

- 修复并固化 E2E 流式断言
- 统一 API 错误、状态、标签和幂等语义
- 补齐服务详情中的事件、诊断、API 示例
- 用契约测试覆盖 controlplane、gateway、observability

验收：一条脚本稳定通过“部署→调用（流式/非流式）→指标→扩容→删除释放”。

### R2：单集群生产化（优先级最高）

目标：真实集群上可持续运行一个租户或多个基础租户。

- Postgres 替换内存 repository；迁移、索引、审计事件持久化
- 真实 client-go/informer；控制面重启后从 Kubernetes 对账恢复
- Prometheus + DCGM Exporter；指标按 deployment/tenant/model 维度查询
- API Key、RBAC、Namespace、ResourceQuota、NetworkPolicy
- 部署超时、重试、失败诊断、删除清理和孤儿资源回收
- 使用 Kueue 或 Volcano 实现队列/配额，不自研调度器

门槛：控制面重启不丢状态；跨租户不可见/不可调用；GPU 与部署状态最终一致；故障能定位到 Pod 事件或运行时日志。

### R3：推理平台能力

目标：从“能部署”升级为“可发布、可扩缩、可运营”。

- 模型版本发布门禁：artifact 校验、启动探针、基准测试、人工确认
- 灰度/回滚、模型路由、限流和 fallback
- 基于 QPS、队列长度、TTFT、KV Cache 的推理扩缩容
- 成本估算、Token 用量、租户账单维度
- 生产 Dashboard、告警和 SLO（可用性、TTFT、错误率）

门槛：一次发布可回滚；扩缩容不会破坏正在服务的请求；每个租户能解释资源和 Token 消耗。

### R4：平台扩展

在 R2/R3 稳定后再做：多运行时（Triton/TensorRT-LLM）、多集群调度、训练 Notebook/Pipeline、Prefill/Decode 分离、AIOps Agent。它们不是当前 MVP 的前置条件。

## 3. 近期迭代顺序

1. E2E/契约测试与文档一致性
2. Postgres repository + migration
3. 真实 K8s adapter + informer/reconcile
4. Prometheus/DCGM 指标链路
5. 租户/RBAC/配额/安全基线
6. 发布门禁与灰度回滚

## 4. 暂不做的决定

不在当前阶段引入 Kubeflow 全家桶、Karmada、自研 GPU scheduler 或 Agent AIOps。原因是它们不能直接解决当前代码的可靠性和产品闭环问题。
