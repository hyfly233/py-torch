package store

import (
	"database/sql"
	"embed"
	"fmt"
	"sort"
	"strings"
)

// Migrator 轻量 migration 执行器。
// SQL 文件按文件名排序执行，已执行的记录在 schema_migrations 表，幂等可重入。
type Migrator struct {
	db *sql.DB
}

// NewMigrator 创建执行器
func NewMigrator(db *sql.DB) *Migrator {
	return &Migrator{db: db}
}

// Migrate 执行内嵌 SQL 文件。
// files 参数为 embed.FS 中文件名列表（如 "001_init.sql"）。
func (m *Migrator) Migrate(fs embed.FS, files []string) error {
	// 建 schema_migrations 表
	if _, err := m.db.Exec(`
		CREATE TABLE IF NOT EXISTS schema_migrations (
			version    TEXT PRIMARY KEY,
			applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
		)`); err != nil {
		return fmt.Errorf("创建 schema_migrations 失败: %w", err)
	}

	// 已执行版本
	applied := map[string]bool{}
	rows, err := m.db.Query(`SELECT version FROM schema_migrations`)
	if err != nil {
		return fmt.Errorf("查询已执行 migration 失败: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var v string
		if err := rows.Scan(&v); err != nil {
			return err
		}
		applied[v] = true
	}

	// 按文件名排序执行
	sort.Strings(files)
	for _, name := range files {
		if applied[name] {
			continue
		}
		data, err := fs.ReadFile(name)
		if err != nil {
			return fmt.Errorf("读取 migration %s 失败: %w", name, err)
		}
		// 每个文件一个事务执行
		tx, err := m.db.Begin()
		if err != nil {
			return err
		}
		if _, err := tx.Exec(string(data)); err != nil {
			_ = tx.Rollback()
			return fmt.Errorf("执行 migration %s 失败: %w", name, err)
		}
		if _, err := tx.Exec(`INSERT INTO schema_migrations (version) VALUES ($1)`, name); err != nil {
			_ = tx.Rollback()
			return fmt.Errorf("记录 migration %s 失败: %w", name, err)
		}
		if err := tx.Commit(); err != nil {
			return err
		}
	}
	return nil
}

// splitStatements 拆分多语句 SQL（按分号，忽略注释内的分号——简单场景够用）
func splitStatements(sql string) []string {
	var out []string
	var cur strings.Builder
	for _, line := range strings.Split(sql, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "--") || trimmed == "" {
			continue
		}
		cur.WriteString(line)
		cur.WriteString("\n")
		if strings.HasSuffix(strings.TrimRight(line, " \t"), ";") {
			out = append(out, cur.String())
			cur.Reset()
		}
	}
	if cur.Len() > 0 {
		out = append(out, cur.String())
	}
	return out
}
