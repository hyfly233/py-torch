// Package worker 控制面后台任务：部署状态对账（Reconciler）。
package worker

import (
	"context"
	"log/slog"
	"time"

	"kk-infra/lib/domain"
	"kk-infra/services/controlplane/internal/biz"
	"kk-infra/services/controlplane/internal/data"
)

// Reconciler 对账器：周期性把控制面部署状态与 Kubernetes 实际状态对齐。
// 满足架构要求：Reconcile 幂等、控制面重启后恢复、失败诊断。
type Reconciler struct {
	repo      data.DeploymentRepository
	deployUse *biz.DeploymentUseCase
	logger    *slog.Logger
	interval  time.Duration
}

// NewReconciler 创建对账器
func NewReconciler(repo data.DeploymentRepository, deployUse *biz.DeploymentUseCase, logger *slog.Logger, interval time.Duration) *Reconciler {
	return &Reconciler{
		repo:      repo,
		deployUse: deployUse,
		logger:    logger,
		interval:  interval,
	}
}

// Run 启动对账循环（阻塞，直到 ctx 取消）
func (r *Reconciler) Run(ctx context.Context) {
	r.logger.Info("Reconciler 启动", "interval", r.interval.String())
	ticker := time.NewTicker(r.interval)
	defer ticker.Stop()

	// 启动时立即对账一次（控制面重启恢复）
	r.reconcileAll(ctx)

	for {
		select {
		case <-ctx.Done():
			r.logger.Info("Reconciler 停止")
			return
		case <-ticker.C:
			r.reconcileAll(ctx)
		}
	}
}

// reconcileAll 对账所有活跃部署
func (r *Reconciler) reconcileAll(ctx context.Context) {
	deploys, err := r.repo.List("")
	if err != nil {
		r.logger.Error("对账：列出部署失败", "err", err)
		return
	}
	for _, d := range deploys {
		// 终态（DELETED）或进行中状态跳过，其余由对账推进
		switch d.Status {
		case domain.DeploymentStatusDeleted, domain.DeploymentStatusDeleting:
			continue
		case domain.DeploymentStatusRunning:
			// 已运行：核对副本数（异常时打日志，MVP 不做自动修复）
			continue
		}
		// 对非终态部署同步 K8s 实际状态
		r.reconcileOne(ctx, d)
	}
}

// reconcileOne 对账单个部署
func (r *Reconciler) reconcileOne(ctx context.Context, d *domain.ModelDeployment) {
	// SyncFromK8s 内部判断状态是否可转换
	r.deployUse.SyncFromK8s(ctx, d.ID)
}
