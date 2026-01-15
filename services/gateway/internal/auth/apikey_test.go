package auth

import "testing"

// API Key 创建 → 鉴权 → 禁用 → 轮换 全流程
func TestAPIKeyLifecycle(t *testing.T) {
	m := NewManager()

	// 创建
	res, err := m.Issue("default")
	if err != nil {
		t.Fatalf("创建失败: %v", err)
	}
	if res.Key == "" || res.KeyID == "" || res.Tenant != "default" {
		t.Fatalf("创建结果异常: %+v", res)
	}

	// 鉴权
	tenant, err := m.Authenticate(res.Key)
	if err != nil || tenant != "default" {
		t.Fatalf("鉴权失败: %v %s", err, tenant)
	}

	// 无效 Key
	if _, err := m.Authenticate("sk-carrot-wrong"); err == nil {
		t.Fatal("无效 Key 应鉴权失败")
	}

	// 禁用
	if err := m.Disable(res.KeyID); err != nil {
		t.Fatalf("禁用失败: %v", err)
	}
	if _, err := m.Authenticate(res.Key); err != ErrKeyDisabled {
		t.Fatalf("禁用后鉴权应返回 ErrKeyDisabled: %v", err)
	}

	// 轮换
	newRes, err := m.Rotate(res.KeyID, "default")
	if err != nil {
		t.Fatalf("轮换失败: %v", err)
	}
	if newRes.Key == res.Key {
		t.Fatal("轮换后 Key 不应相同")
	}
	if _, err := m.Authenticate(newRes.Key); err != nil {
		t.Fatalf("新 Key 鉴权失败: %v", err)
	}

	// 明文只出现一次：存储中不应有明文
	for _, k := range m.List() {
		if k.KeyHash == res.Key {
			t.Fatal("存储不应包含明文 Key")
		}
	}
}

// 最近调用时间更新
func TestLastUsedAt(t *testing.T) {
	m := NewManager()
	res, _ := m.Issue("t1")
	if _, err := m.Authenticate(res.Key); err != nil {
		t.Fatalf("鉴权失败: %v", err)
	}
	for _, k := range m.List() {
		if k.ID == res.KeyID && k.LastUsedAt.IsZero() {
			t.Fatal("鉴权后应更新 LastUsedAt")
		}
	}
}
