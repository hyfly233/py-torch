// inference Mock vLLM 推理后端入口（本地 MVP 闭环用）。
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

	"kk-infra/services/inference/internal/server"
)

func main() {
	addr := flag.String("addr", ":8085", "监听地址")
	model := flag.String("model", "qwen-demo", "模拟的模型名（与部署服务名一致）")
	flag.Parse()

	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	srv := server.NewServer(logger, *model)
	httpSrv := &http.Server{
		Addr:              *addr,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	go func() {
		logger.Info("HTTP 服务监听", "addr", *addr, "model", *model)
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
	logger.Info("inference 已退出")
}
