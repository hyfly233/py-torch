// gateway AI 网关服务入口。
// 提供 OpenAI 兼容 API：/v1/models、/v1/chat/completions（含流式）。
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

	"kk-infra/services/gateway/internal/auth"
	"kk-infra/services/gateway/internal/proxy"
	"kk-infra/services/gateway/internal/router"
	"kk-infra/services/gateway/internal/server"
	"kk-infra/services/gateway/internal/sink"
)

func main() {
	addr := flag.String("addr", ":8083", "监听地址")
	ratePerMinute := flag.Int("rate-per-minute", 0, "每租户每分钟限流（0=不限）")
	obsURL := flag.String("observability-url", "", "observability 服务地址（如 http://localhost:8084），为空则不上报指标")
	flag.Parse()

	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))

	keys := auth.NewManager()
	routes := router.NewTable()

	// 指标上报：配置 observability 地址时启用
	var m proxy.MetricsSink
	if *obsURL != "" {
		m = sink.NewHTTPSink(*obsURL, logger)
		logger.Info("指标上报已启用", "observability", *obsURL)
	}
	p := proxy.NewProxy(routes, logger, m)
	if *ratePerMinute > 0 {
		p.RatePerMinute = *ratePerMinute
	}
	srv := server.NewServer(keys, routes, p, logger)

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
	logger.Info("gateway 已退出")
}
