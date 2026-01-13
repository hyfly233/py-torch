package biz

import (
	"context"

	"kk-infra/lib/domain"
	"kk-infra/lib/errcode"
)

// ResourceUseCase GPU 资源查询用例
type ResourceUseCase struct {
	kube K8sClient
}

// NewResourceUseCase 创建用例
func NewResourceUseCase(kube K8sClient) *ResourceUseCase {
	return &ResourceUseCase{kube: kube}
}

// ListGPUResources 返回 GPU 节点列表与汇总
func (uc *ResourceUseCase) ListGPUResources(ctx context.Context, gpuType string) (*domain.GPUSummary, []domain.GPUResource, error) {
	nodes, err := uc.kube.ListGPUs(ctx)
	if err != nil {
		return nil, nil, errcode.Wrap(errcode.ErrInternal, "查询 GPU 资源失败", err)
	}

	var filtered []domain.GPUResource
	if gpuType != "" {
		for _, n := range nodes {
			if n.GPUType == gpuType {
				filtered = append(filtered, n)
			}
		}
	} else {
		filtered = nodes
	}

	summary := summarizeGPUs(filtered)
	return &summary, filtered, nil
}

// summarizeGPUs 汇总 GPU 资源
func summarizeGPUs(nodes []domain.GPUResource) domain.GPUSummary {
	s := domain.GPUSummary{
		ByType: make(map[string]*domain.GPUTypeSummary),
	}
	for _, n := range nodes {
		s.TotalGPU += n.Total
		s.UsedGPU += n.Used
		if n.Health != domain.GPUHealthHealthy {
			s.ErrorGPU += n.Total
		}
		ts, ok := s.ByType[n.GPUType]
		if !ok {
			ts = &domain.GPUTypeSummary{GPUType: n.GPUType}
			s.ByType[n.GPUType] = ts
		}
		ts.Total += n.Total
		ts.Used += n.Used
		ts.MemoryMB = n.MemoryMB
	}
	s.AvailableGPU = s.TotalGPU - s.UsedGPU
	if s.AvailableGPU < 0 {
		s.AvailableGPU = 0
	}
	s.NodeCount = len(nodes)
	// 计算各型号可用与利用率
	for _, n := range nodes {
		if ts, ok := s.ByType[n.GPUType]; ok {
			ts.Available = ts.Total - ts.Used
			if ts.Available < 0 {
				ts.Available = 0
			}
			ts.Utilization = n.Utilization
		}
	}
	return s
}
