// controlplane 控制面服务入口。
// 编排模型注册、GPU 资源、部署生命周期，提供 REST API。
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

	"kk-infra/services/controlplane/internal/biz"
	"kk-infra/services/controlplane/internal/clients"
	"kk-infra/services/controlplane/internal/data"
	"kk-infra/services/controlplane/internal/server"
	"kk-infra/services/controlplane/internal/worker"
)

func main() {
	addr := flag.String("addr", ":8080", "监听地址")
	modelRegistry := flag.String("model-registry", "http://127.0.0.1:8081", "modelregistry 地址")
	k8sAdapter := flag.String("k8s-adapter", "http://127.0.0.1:8082", "k8sadapter 地址")
	deploymentImage := flag.String("deployment-image", "", "部署使用的模型镜像（默认 k8sadapter 决定）")
	obsURL := flag.String("observability-url", "", "observability 服务地址（如 http://localhost:8084），为空则指标返回占位")
	flag.Parse()

	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))

	repo := data.NewMemoryDeploymentRepository()
	modelClient := clients.NewModelRegistryClient(*modelRegistry)
	kubeClient := clients.NewK8sAdapterClient(*k8sAdapter)

	deployUse := biz.NewDeploymentUseCase(repo, modelClient, kubeClient)
	deployUse.SetDeploymentImage(*deploymentImage)
	resUse := biz.NewResourceUseCase(kubeClient)
	srv := server.NewServer(deployUse, resUse, repo, logger)
	if *obsURL != "" {
		srv.SetObservabilityClient(clients.NewObservabilityClient(*obsURL))
		logger.Info("可观测性接入", "observability", *obsURL)
	}

	// Reconciler：每 3 秒对账一次
	reconciler := worker.NewReconciler(repo, deployUse, logger, 3*time.Second)

	httpSrv := &http.Server{
		Addr:              *addr,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	// 启动 Reconciler
	go reconciler.Run(ctx)

	go func() {
		logger.Info("HTTP 服务监听", "addr", *addr,
			"modelRegistry", *modelRegistry, "k8sAdapter", *k8sAdapter)
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
	logger.Info("controlplane 已退出")
}
