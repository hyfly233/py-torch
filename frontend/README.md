# Carrot AI Infra 前端管理台

Carrot AI Infra Platform 的 Web 管理控制台（Vue 3 + TypeScript + Vite）。

## 功能

- 登录页（MVP 本地模拟：管理员 / 普通用户两种角色，不接真实认证）
- 总览 Dashboard：GPU 资源汇总、在线/异常服务、最近部署
- 模型管理：模型列表、详情、版本注册与校验
- 模型服务：服务列表、详情、部署向导（选模型 → 配 GPU → 配服务）、扩容/重启/删除
- GPU 资源页：汇总卡片 + 节点表格 + 型号筛选
- API Key 管理（仅管理员）：创建（明文展示一次）/ 禁用 / 轮换
- 角色权限：普通用户只读查看，管理员可管理；路由守卫拦截越权访问

## 开发

```bash
npm install        # 首次安装依赖（使用 npm，非 yarn）
npm run dev        # 本地开发，默认 http://localhost:5173
npm run build      # 生产构建，输出到 dist/
```

## 代理配置

开发环境通过 `vite.config.ts` 代理后端服务：

| 前端路径前缀 | 后端服务 | 端口 |
|---|---|---|
| `/api` | controlplane 控制面 | 8080 |
| `/model-registry` | modelregistry 模型注册中心 | 8081 |
| `/gateway` | gateway 推理网关 | 8083 |

## 技术选型

- 路由：vue-router
- HTTP：原生 fetch 封装（src/api/）
- 状态：轻量 composable（useAuth，不引 Pinia）
- UI：手写 CSS 深色管理后台风格，不引 UI 框架
