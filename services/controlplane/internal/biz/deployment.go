// Package biz 控制面业务逻辑
package biz

import (
	"context"
	"time"

	"kk-infra/lib/apitypes"
	"kk-infra/lib/domain"
	"kk-infra/lib/errcode"
	"kk-infra/services/controlplane/internal/clients"
	"kk-infra/services/controlplane/internal/data"
)

// ModelRegistryClient 模型注册接口（便于测试替换）
type ModelRegistryClient interface {
	GetVersion(ctx context.Context, versionID string) (*domain.ModelVersion, error)
}

// K8sClient K8s 适配器接口（便于测试替换）
type K8sClient interface {
	ListGPUs(ctx context.Context) ([]domain.GPUResource, error)
	CreateDeployment(ctx context.Context, spec *clients.CreateDeploymentSpec) (*clients.K8sDeploymentResult, error)
	GetDeployment(ctx context.Context, name, namespace string) (*clients.K8sDeploymentResult, error)
	ScaleDeployment(ctx context.Context, name, namespace string, replicas int32) (*clients.K8sDeploymentResult, error)
	DeleteDeployment(ctx context.Context, name, namespace string) error
}

// DeploymentUseCase 部署业务用例
type DeploymentUseCase struct {
	repo    data.DeploymentRepository
	models  ModelRegistryClient
	kube    K8sClient
	sm      *domain.DeploymentStateMachine
	now     func() time.Time
	// 默认租户（MVP 单租户）
	defaultTenant string
	// 部署镜像（空则使用 k8sadapter 默认；本机验证用 mock 镜像）
	deploymentImage string
}

// NewDeploymentUseCase 创建用例
func NewDeploymentUseCase(repo data.DeploymentRepository, models ModelRegistryClient, kube K8sClient) *DeploymentUseCase {
	return &DeploymentUseCase{
		repo:          repo,
		models:        models,
		kube:          kube,
		sm:            domain.NewDeploymentStateMachine(),
		now:           time.Now,
		defaultTenant: "default",
	}
}

// SetDeploymentImage 设置部署镜像（验证环境注入 mock 镜像）
func (uc *DeploymentUseCase) SetDeploymentImage(image string) {
	uc.deploymentImage = image
}

// CreateDeployment 创建部署（幂等：同名且未删除时返回同一部署；已删除同名允许重建）
func (uc *DeploymentUseCase) CreateDeployment(ctx context.Context, req *apitypes.CreateDeploymentRequest) (*domain.ModelDeployment, error) {
	// 幂等：按名称查已存在部署（跳过已删除的，允许同名重建）
	if existing, err := uc.repo.GetByName(req.Name); err == nil && existing.Status != domain.DeploymentStatusDeleted {
		return existing, nil
	}

	// 1. 校验模型版本
	version, err := uc.models.GetVersion(ctx, req.ModelVersionID)
	if err != nil {
		return nil, errcode.Wrap(errcode.ErrModelVersionFound, "模型版本校验失败", err)
	}
	if !version.Deployable() {
		return nil, errcode.New(errcode.ErrModelNotDeployable,
			"模型版本不可部署（状态="+version.Status+"），请先完成校验")
	}
	// 版本对应模型 ID（MVP：版本接口返回 ModelID）
	modelID := version.ModelID

	// 2. 校验资源配额（GPU 足够）
	tenantID := req.TenantID
	if tenantID == "" {
		tenantID = uc.defaultTenant
	}
	if err := uc.checkGPUQuota(ctx, version.GPUType, version.GPUCount, req.Replicas); err != nil {
		return nil, err
	}

	// 3. 构建部署对象
	now := uc.now()
	namespace := req.Namespace
	if namespace == "" {
		namespace = "tenant-" + tenantID
	}
	d := &domain.ModelDeployment{
		ID:             req.IdempotencyKey,
		Name:           req.Name,
		ModelID:        modelID,
		ModelVersionID: version.ID,
		ModelName:      version.ModelName,
		ModelVersion:   version.Version,
		TenantID:       tenantID,
		Namespace:      namespace,
		Replicas:       req.Replicas,
		Resource: domain.Resource{
			GPUType:  version.GPUType,
			GPUCount: version.GPUCount,
			MemoryMB: version.MemoryMB,
		},
		Runtime:     version.Runtime,
		StartupArgs: req.StartupArgs,
		Status:      domain.DeploymentStatusNew,
		CreatedAt:   now,
		UpdatedAt:   now,
	}
	if d.Replicas <= 0 {
		d.Replicas = 1
	}

	// 4. 写入状态
	if err := uc.repo.Create(d); err != nil {
		if err == data.ErrConflict {
			return uc.repo.GetByName(req.Name)
		}
		return nil, errcode.Wrap(errcode.ErrInternal, "写入部署失败", err)
	}
	uc.recordEvent(d.ID, "", domain.DeploymentStatusNew, "创建部署请求", getRequestID(ctx), "")

	// 5. 异步提交（VALIDATING → SUBMITTING → K8s）
	// 注意：不能复用请求 ctx（请求返回后即取消），使用独立后台 ctx
	go uc.submit(context.Background(), d)
	return d, nil
}

// submit 异步提交流程：校验 → 提交 K8s → 等待启动
func (uc *DeploymentUseCase) submit(ctx context.Context, d *domain.ModelDeployment) {
	// 校验阶段
	if err := uc.transition(d.ID, domain.DeploymentStatusNew, domain.DeploymentStatusValidating, "开始校验"); err != nil {
		uc.failDeployment(d.ID, "状态流转失败: "+err.Error())
		return
	}
	time.Sleep(200 * time.Millisecond) // 模拟校验耗时
	uc.transition(d.ID, domain.DeploymentStatusValidating, domain.DeploymentStatusSubmitting, "校验通过，提交 Kubernetes")

	// 提交 K8s
	spec := &clients.CreateDeploymentSpec{
		DeploymentID: d.ID,
		Name:         d.Name,
		Namespace:    d.Namespace,
		Replicas:     d.Replicas,
		Resource:     d.Resource,
		Image:        uc.deploymentImage,
		Args:         d.StartupArgs,
		Labels: map[string]string{
			"carrot.ai/deployment-id": d.ID,
			"carrot.ai/model-id":      d.ModelID,
			"carrot.ai/model-version": d.ModelVersion,
			"carrot.ai/tenant-id":     d.TenantID,
			"carrot.ai/managed-by":    "carrot",
		},
		ModelPath: deploymentModelPath(d),
	}
	res, err := uc.kube.CreateDeployment(ctx, spec)
	if err != nil {
		uc.failDeployment(d.ID, "提交 Kubernetes 失败: "+err.Error())
		return
	}
	uc.transition(d.ID, domain.DeploymentStatusSubmitting, domain.DeploymentStatusStarting, "已创建 Kubernetes 资源")

	// 保存 Endpoint
	if res.Endpoint != "" {
		uc.updateDeployment(d.ID, func(dd *domain.ModelDeployment) {
			dd.Endpoint = "http://" + res.Endpoint
		})
	}

	// 等待 Running（Reconciler 也会推进，这里只做首次同步）
	uc.SyncFromK8s(ctx, d.ID)
}

// transition 执行状态转换并记录事件
func (uc *DeploymentUseCase) transition(id, from, to, reason string) error {
	if err := uc.sm.Transition(from, to); err != nil {
		uc.recordEvent(id, from, to, "非法转换: "+err.Error(), getRequestID(context.Background()), "")
		return err
	}
	uc.recordEvent(id, from, to, reason, getRequestID(context.Background()), "")
	return uc.updateDeployment(id, func(d *domain.ModelDeployment) {
		d.Status = to
		d.UpdatedAt = uc.now()
	})
}

// failDeployment 置为失败
func (uc *DeploymentUseCase) failDeployment(id, diag string) {
	uc.updateDeployment(id, func(d *domain.ModelDeployment) {
		from := d.Status
		if uc.sm.CanTransition(from, domain.DeploymentStatusFailed) {
			d.Status = domain.DeploymentStatusFailed
			d.Diagnostics = diag
			d.UpdatedAt = uc.now()
			uc.recordEvent(id, from, domain.DeploymentStatusFailed, "部署失败", getRequestID(context.Background()), diag)
		}
	})
}

// updateDeployment 更新部署（乐观并发）
func (uc *DeploymentUseCase) updateDeployment(id string, fn func(*domain.ModelDeployment)) error {
	d, err := uc.repo.Get(id)
	if err != nil {
		return err
	}
	fn(d)
	d.UpdatedAt = uc.now()
	return uc.repo.Update(d)
}

// recordEvent 记录状态事件
func (uc *DeploymentUseCase) recordEvent(id, from, to, reason, requestID, diag string) {
	uc.repo.AddEvent(&domain.StatusEvent{
		DeploymentID: id,
		From:         from,
		To:           to,
		Reason:       reason,
		RequestID:    requestID,
		Diagnostics:  diag,
		At:           uc.now(),
	})
}

// GetDeployment 查询部署
func (uc *DeploymentUseCase) GetDeployment(id string) (*domain.ModelDeployment, error) {
	d, err := uc.repo.Get(id)
	if err != nil {
		return nil, errcode.Wrap(errcode.ErrNotFound, "部署不存在: "+id, err)
	}
	return d, nil
}

// ListDeployments 部署列表
func (uc *DeploymentUseCase) ListDeployments(tenantID string) ([]*domain.ModelDeployment, error) {
	return uc.repo.List(tenantID)
}

// ScaleDeployment 扩缩容（幂等）
func (uc *DeploymentUseCase) ScaleDeployment(ctx context.Context, id string, replicas int32) (*domain.ModelDeployment, error) {
	d, err := uc.repo.Get(id)
	if err != nil {
		return nil, errcode.Wrap(errcode.ErrNotFound, "部署不存在: "+id, err)
	}
	if d.Status != domain.DeploymentStatusRunning && d.Status != domain.DeploymentStatusFailed {
		return nil, errcode.New(errcode.ErrIllegalState, "当前状态 "+d.Status+" 不允许扩缩容")
	}
	// 配额校验
	if err := uc.checkGPUQuota(ctx, d.Resource.GPUType, d.Resource.GPUCount, replicas); err != nil {
		return nil, err
	}

	// 状态机：Running → SCALING
	if d.Status == domain.DeploymentStatusRunning {
		if err := uc.transition(id, d.Status, domain.DeploymentStatusScaling, "发起扩缩容到 "+itoa32(replicas)); err != nil {
			return nil, err
		}
	}
	d.Replicas = replicas
	d.Generation++
	uc.repo.Update(d)

	// 调 K8s
	res, err := uc.kube.ScaleDeployment(ctx, d.Name, d.Namespace, replicas)
	if err != nil {
		uc.failDeployment(id, "扩缩容失败: "+err.Error())
		return nil, errcode.Wrap(errcode.ErrInternal, "扩缩容失败", err)
	}
	_ = res

	// SCALING → RUNNING
	if d.Status == domain.DeploymentStatusScaling {
		uc.transition(id, domain.DeploymentStatusScaling, domain.DeploymentStatusRunning, "扩缩容完成")
	}
	return uc.repo.Get(id)
}

// RestartDeployment 重启（MVP：调 K8s 重建 Pod 简化）
func (uc *DeploymentUseCase) RestartDeployment(ctx context.Context, id string) (*domain.ModelDeployment, error) {
	d, err := uc.repo.Get(id)
	if err != nil {
		return nil, errcode.Wrap(errcode.ErrNotFound, "部署不存在: "+id, err)
	}
	if d.Status != domain.DeploymentStatusRunning {
		return nil, errcode.New(errcode.ErrIllegalState, "当前状态 "+d.Status+" 不允许重启")
	}
	uc.transition(id, d.Status, domain.DeploymentStatusRestarting, "发起重启")
	time.Sleep(300 * time.Millisecond) // 模拟重启
	uc.transition(id, domain.DeploymentStatusRestarting, domain.DeploymentStatusRunning, "重启完成")
	return uc.repo.Get(id)
}

// DeleteDeployment 删除部署（幂等：不存在返回成功）
func (uc *DeploymentUseCase) DeleteDeployment(ctx context.Context, id string) error {
	d, err := uc.repo.Get(id)
	if err != nil {
		if err == data.ErrNotFound {
			return nil // 幂等删除
		}
		return errcode.Wrap(errcode.ErrInternal, "查询部署失败", err)
	}
	// 记录删除事件（任何状态均可删除）
	uc.recordEvent(id, d.Status, domain.DeploymentStatusDeleting, "发起删除", getRequestID(ctx), "")
	uc.updateDeployment(id, func(dd *domain.ModelDeployment) {
		dd.Status = domain.DeploymentStatusDeleting
	})

	// 调 K8s 删除（幂等）
	if err := uc.kube.DeleteDeployment(ctx, d.Name, d.Namespace); err != nil {
		uc.failDeployment(id, "删除失败: "+err.Error())
		return errcode.Wrap(errcode.ErrInternal, "删除失败", err)
	}
	uc.recordEvent(id, domain.DeploymentStatusDeleting, domain.DeploymentStatusDeleted, "删除完成", getRequestID(ctx), "")
	uc.updateDeployment(id, func(dd *domain.ModelDeployment) {
		dd.Status = domain.DeploymentStatusDeleted
	})
	return nil
}

// SyncFromK8s 从 K8s 同步部署状态（Reconciler 与提交后首次同步共用）
func (uc *DeploymentUseCase) SyncFromK8s(ctx context.Context, id string) {
	d, err := uc.repo.Get(id)
	if err != nil || d.Status == domain.DeploymentStatusDeleted || d.Status == domain.DeploymentStatusDeleting {
		return
	}
	res, err := uc.kube.GetDeployment(ctx, d.Name, d.Namespace)
	if err != nil {
		// 不存在说明可能还没创建完，忽略
		return
	}
	if res.Status == nil {
		return
	}
	switch res.Status.Condition {
	case "Available":
		if uc.sm.CanTransition(d.Status, domain.DeploymentStatusRunning) {
			uc.transition(id, d.Status, domain.DeploymentStatusRunning, "Pod 就绪")
		}
	case "ReplicaFailure":
		diag := res.Message
		if diag == "" {
			diag = "Pod 运行失败"
		}
		uc.failDeployment(id, diag)
	}
}

// checkGPUQuota 校验 GPU 资源与租户配额。
// MVP 单租户：检查集群该 GPU 型号是否有足够可用量。
func (uc *DeploymentUseCase) checkGPUQuota(ctx context.Context, gpuType string, gpuCount, replicas int32) error {
	if replicas <= 0 {
		replicas = 1
	}
	need := gpuCount * replicas
	nodes, err := uc.kube.ListGPUs(ctx)
	if err != nil {
		return errcode.Wrap(errcode.ErrInternal, "查询 GPU 资源失败", err)
	}
	var available int32
	for _, n := range nodes {
		if n.GPUType == gpuType {
			available += n.Available()
		}
	}
	if available < need {
		return errcode.New(errcode.ErrInsufficientGPU,
			"GPU 资源不足：需要 "+itoa32(need)+" 张 "+gpuType+"，当前可用 "+itoa32(available))
	}
	return nil
}

// deploymentModelPath 计算模型权重路径。
// MVP：模型路径来自版本 ArtifactURI 的简化解析（真实场景由模型存储适配器注入）。
func deploymentModelPath(d *domain.ModelDeployment) string {
	return "/models/" + d.ModelName
}

// getRequestID 从 context 取 RequestID（biz 层不依赖 middleware 包）
func getRequestID(ctx context.Context) string {
	if v, ok := ctx.Value(requestIDKey{}).(string); ok {
		return v
	}
	return ""
}

type requestIDKey struct{}

// itoa32 int32 转字符串
func itoa32(n int32) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
