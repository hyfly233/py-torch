package data

import (
	"database/sql"
	"time"
)

// AuditEntry 审计日志条目
type AuditEntry struct {
	ID        int64     `json:"id"`
	Action    string    `json:"action"`    // 如 deployment.create / deployment.delete
	Actor     string    `json:"actor"`     // 操作者
	TenantID  string    `json:"tenantId"`  // 租户
	Resource  string    `json:"resource"`  // 资源标识（部署 ID / Key ID 等）
	RequestID string    `json:"requestId"` // 请求 ID
	Detail    string    `json:"detail"`    // 详情
	CreatedAt time.Time `json:"createdAt"`
}

// AuditStore 审计日志存储（内存 / Postgres）
type AuditStore interface {
	Write(e *AuditEntry) error
	List(limit int) ([]AuditEntry, error)
}

// MemoryAuditStore 内存实现
type MemoryAuditStore struct {
	entries []AuditEntry
	seq     int64
}

// NewMemoryAuditStore 创建内存审计存储
func NewMemoryAuditStore() *MemoryAuditStore {
	return &MemoryAuditStore{}
}

func (s *MemoryAuditStore) Write(e *AuditEntry) error {
	s.seq++
	e.ID = s.seq
	s.entries = append(s.entries, *e)
	if len(s.entries) > 500 {
		s.entries = s.entries[len(s.entries)-500:]
	}
	return nil
}

func (s *MemoryAuditStore) List(limit int) ([]AuditEntry, error) {
	if limit <= 0 || limit > len(s.entries) {
		limit = len(s.entries)
	}
	out := make([]AuditEntry, limit)
	copy(out, s.entries[len(s.entries)-limit:])
	return out, nil
}

// PostgresAuditStore Postgres 实现
type PostgresAuditStore struct {
	db *sql.DB
}

// NewPostgresAuditStore 创建 Postgres 审计存储
func NewPostgresAuditStore(db *sql.DB) *PostgresAuditStore {
	return &PostgresAuditStore{db: db}
}

func (s *PostgresAuditStore) Write(e *AuditEntry) error {
	err := s.db.QueryRow(
		`INSERT INTO audit_logs (action, actor, tenant_id, resource, request_id, detail, created_at)
		 VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING id`,
		e.Action, e.Actor, e.TenantID, e.Resource, e.RequestID, e.Detail, e.CreatedAt,
	).Scan(&e.ID)
	return err
}

func (s *PostgresAuditStore) List(limit int) ([]AuditEntry, error) {
	if limit <= 0 || limit > 200 {
		limit = 200
	}
	rows, err := s.db.Query(
		`SELECT id, action, actor, tenant_id, resource, request_id, detail, created_at
		 FROM audit_logs ORDER BY id DESC LIMIT $1`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]AuditEntry, 0)
	for rows.Next() {
		var e AuditEntry
		if err := rows.Scan(&e.ID, &e.Action, &e.Actor, &e.TenantID, &e.Resource, &e.RequestID, &e.Detail, &e.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, e)
	}
	return out, nil
}
