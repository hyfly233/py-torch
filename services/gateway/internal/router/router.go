// Package router 模型路由：model 字段 → 后端服务 endpoint。
package router

import (
	"errors"
	"sync"
)

// ErrRouteNotFound 模型无路由
var ErrRouteNotFound = errors.New("model route not found")

// Route 路由条目
type Route struct {
	Model     string `json:"model"`     // 模型名（= 部署服务名）
	Endpoint  string `json:"endpoint"`  // 后端 base URL（如 http://qwen-demo:80）
	TenantID  string `json:"tenantId"`  // 归属租户
	DeploymentID string `json:"deploymentId"`
}

// Table 路由表（线程安全）
type Table struct {
	mu     sync.RWMutex
	routes map[string]*Route // model → route
}

// NewTable 创建路由表
func NewTable() *Table {
	return &Table{routes: make(map[string]*Route)}
}

// Register 注册/更新路由
func (t *Table) Register(r *Route) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.routes[r.Model] = r
}

// Unregister 注销路由
func (t *Table) Unregister(model string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.routes, model)
}

// Resolve 解析模型路由
func (t *Table) Resolve(model string) (*Route, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	r, ok := t.routes[model]
	if !ok {
		return nil, ErrRouteNotFound
	}
	return r, nil
}

// List 列出全部路由
func (t *Table) List() []*Route {
	t.mu.RLock()
	defer t.mu.RUnlock()
	out := make([]*Route, 0, len(t.routes))
	for _, r := range t.routes {
		out = append(out, r)
	}
	return out
}
