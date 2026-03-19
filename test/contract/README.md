# 契约测试说明

跨服务契约测试由 `hack/contract-test.sh` 执行（真实服务 HTTP 对拍）。

## 为什么不用 Go 集成测试模块

各服务业务代码在 `internal/` 包下，Go 的 internal 规则禁止跨模块引用。
契约测试需要启动真实服务进程，因此采用 shell + HTTP 对拍方式，
与 e2e-test.sh 一致。

## 执行方式

```bash
./hack/dev-up.sh --storage=memory   # 或 postgres
./hack/contract-test.sh
```

## 覆盖范围（R1-1）

| 契约 | 断言 |
|---|---|
| 错误码 | 404（不存在资源）、400（缺必填）、409（重名）、401（无效 Key） |
| 状态机 | 部署初始非终态、可达 RUNNING |
| 幂等 | 同 IdempotencyKey 返回同一部署；幂等删除 |
| 响应格式 | code/message/requestId 统一包装；X-Request-Id 响应头 |
