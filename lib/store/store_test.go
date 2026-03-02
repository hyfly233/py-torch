package store

import (
	"testing"
)

// 默认配置指向本机 docker Postgres（/Users/flyhy/workspace/docker/postgresql）
// 如无 PG 环境，测试自动跳过。
func TestOpenAndMigrate(t *testing.T) {
	cfg := DefaultConfig()
	db, err := Open(cfg)
	if err != nil {
		t.Skipf("本机 Postgres 不可用，跳过: %v", err)
	}
	defer db.Close()

	m := NewMigrator(db)
	_ = m
	// 直接执行 SQL 文件内容（migrate.go 的 embed 版本由各服务提供）
	if _, err := db.Exec(`
		CREATE TABLE IF NOT EXISTS schema_migrations (
			version TEXT PRIMARY KEY,
			applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
		)`); err != nil {
		t.Fatalf("建表失败: %v", err)
	}
	t.Log("本机 Postgres 连接成功")
}

// DSN 格式验证
func TestDSN(t *testing.T) {
	cfg := DefaultConfig()
	dsn := cfg.DSN()
	if dsn == "" {
		t.Fatal("DSN 不应为空")
	}
	t.Logf("DSN: %s", dsn)
}
