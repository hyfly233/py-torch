// Package auth API Key 管理。
// 安全要求：Key 明文只在创建时返回一次，存储只保留 SHA-256 哈希。
package auth

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"sync"
	"time"
)

// ErrInvalidKey API Key 无效
var ErrInvalidKey = errors.New("invalid api key")

// ErrKeyDisabled Key 已禁用
var ErrKeyDisabled = errors.New("api key disabled")

// APIKey 存储模型
type APIKey struct {
	ID         string    `json:"id"`
	KeyHash    string    `json:"keyHash"` // SHA-256，不存明文
	TenantID   string    `json:"tenantId"`
	CreatedAt  time.Time `json:"createdAt"`
	LastUsedAt time.Time `json:"lastUsedAt,omitempty"`
	Disabled   bool      `json:"disabled"`
}

// Manager API Key 管理器
type Manager struct {
	mu   sync.RWMutex
	keys map[string]*APIKey // keyHash → key
	seq  int
}

// NewManager 创建管理器
func NewManager() *Manager {
	return &Manager{keys: make(map[string]*APIKey)}
}

// IssueResult 创建结果（明文只出现一次）
type IssueResult struct {
	Key    string `json:"key"`    // 明文 Key，仅此一次
	KeyID  string `json:"keyId"`
	Tenant string `json:"tenant"`
}

// Issue 创建新 API Key
func (m *Manager) Issue(tenantID string) (*IssueResult, error) {
	plain, err := generateKey()
	if err != nil {
		return nil, err
	}
	hash := hashKey(plain)
	m.mu.Lock()
	defer m.mu.Unlock()
	m.seq++
	k := &APIKey{
		ID:        "k" + itoa(m.seq),
		KeyHash:   hash,
		TenantID:  tenantID,
		CreatedAt: time.Now(),
	}
	m.keys[hash] = k
	return &IssueResult{Key: plain, KeyID: k.ID, Tenant: tenantID}, nil
}

// Authenticate 校验 Key，返回租户
func (m *Manager) Authenticate(plain string) (string, error) {
	if plain == "" {
		return "", ErrInvalidKey
	}
	hash := hashKey(plain)
	m.mu.RLock()
	k, ok := m.keys[hash]
	m.mu.RUnlock()
	if !ok {
		return "", ErrInvalidKey
	}
	if k.Disabled {
		return "", ErrKeyDisabled
	}
	// 更新最近调用时间（不影响主路径）
	m.mu.Lock()
	k.LastUsedAt = time.Now()
	m.mu.Unlock()
	return k.TenantID, nil
}

// Disable 禁用 Key
func (m *Manager) Disable(keyID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, k := range m.keys {
		if k.ID == keyID {
			k.Disabled = true
			return nil
		}
	}
	return ErrInvalidKey
}

// Rotate 轮换 Key：禁用旧的并签发新的
func (m *Manager) Rotate(oldKeyID, tenantID string) (*IssueResult, error) {
	if err := m.Disable(oldKeyID); err != nil {
		return nil, err
	}
	return m.Issue(tenantID)
}

// List 列出 Key（不含哈希明文，仅元数据）
func (m *Manager) List() []APIKey {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make([]APIKey, 0, len(m.keys))
	for _, k := range m.keys {
		out = append(out, *k)
	}
	return out
}

// generateKey 生成 sk-carrot-<hex> 格式 Key
func generateKey() (string, error) {
	b := make([]byte, 24)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return "sk-carrot-" + hex.EncodeToString(b), nil
}

// hashKey SHA-256 哈希
func hashKey(plain string) string {
	sum := sha256.Sum256([]byte(plain))
	return hex.EncodeToString(sum[:])
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}
