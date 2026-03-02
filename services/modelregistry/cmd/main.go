// modelregistry 模型注册中心服务入口。
// 存储支持内存（--storage=memory，默认）或 Postgres（--storage=postgres）。
package main

import (
	"context"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"kk-infra/lib/store"
	"kk-infra/services/modelregistry/internal/biz"
	"kk-infra/services/modelregistry/internal/data"
	"kk-infra/services/modelregistry/internal/server"
)

func main() {
	addr := flag.String("addr", ":8081", "监听地址")
	storage := flag.String("storage", "memory", "存储后端: memory | postgres")
	flag.Parse()

	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	logger.Info("modelregistry 启动", "addr", *addr, "storage", *storage)

	var repo data.Repository
	if *storage == "postgres" {
		db, err := store.Open(store.DefaultConfig())
		if err != nil {
			logger.Error("连接 Postgres 失败", "err", err)
			os.Exit(1)
		}
		if err := store.MigrateAll(db); err != nil {
			logger.Error("执行 migration 失败", "err", err)
			os.Exit(1)
		}
		repo = data.NewPostgresRepository(db)
		logger.Info("使用 Postgres 存储", "db", store.DefaultConfig().DBName)
	} else {
		repo = data.NewMemoryRepository()
		logger.Info("使用内存存储")
	}

	registry := biz.NewRegistry(repo)
	srv := server.NewServer(registry, logger)

	httpSrv := &http.Server{
		Addr:              *addr,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	go func() {
		logger.Info("HTTP 服务监听", "addr", *addr)
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Error("HTTP 服务异常退出", "err", err)
			os.Exit(1)
		}
	}()

	<-ctx.Done()
	logger.Info("收到退出信号，开始优雅关闭")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = httpSrv.Shutdown(shutdownCtx)
	logger.Info("modelregistry 已退出")
}
