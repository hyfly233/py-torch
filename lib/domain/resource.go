package domain

// Resource 资源规格：CPU/内存/GPU 的统一描述
type Resource struct {
	MemoryMB    int64  `json:"memoryMB"`    // 内存
	VCores      int32  `json:"vcores"`      // CPU 核数
	GPUCount    int32  `json:"gpuCount"`    // GPU 数量
	GPUType     string `json:"gpuType"`     // GPU 型号
	GPUMemoryMB int64  `json:"gpuMemoryMB"` // 单卡显存
}

// GPUHealth GPU 健康状态
const (
	GPUHealthHealthy = "Healthy"
	GPUHealthUnknown = "Unknown"
	GPUHealthError   = "Error"
)

// GPUResource GPU 资源快照：以节点 × 型号为粒度
type GPUResource struct {
	NodeName    string  `json:"nodeName"`    // K8s 节点名
	GPUType     string  `json:"gpuType"`     // GPU 型号，如 A100
	Total       int32   `json:"total"`       // 节点 GPU 总数
	Allocatable int32   `json:"allocatable"` // 可分配数（=Total）
	Used        int32   `json:"used"`        // 已用数
	MemoryMB    int64   `json:"memoryMB"`    // 单卡显存
	Utilization float64 `json:"utilization"` // 平均利用率 0-100
	Health      string  `json:"health"`      // Healthy/Unknown/Error
}

// Available 返回该节点可用 GPU 数
func (g *GPUResource) Available() int32 {
	if g == nil {
		return 0
	}
	a := g.Allocatable - g.Used
	if a < 0 {
		return 0
	}
	return a
}

// GPUSummary GPU 资源汇总
type GPUSummary struct {
	TotalGPU     int32   `json:"totalGpu"`     // GPU 总数
	AvailableGPU int32   `json:"availableGpu"` // 可用 GPU 数
	UsedGPU      int32   `json:"usedGpu"`      // 已用 GPU 数
	ErrorGPU     int32   `json:"errorGpu"`     // 异常 GPU 数
	NodeCount    int     `json:"nodeCount"`    // 节点数
	ByType       map[string]*GPUTypeSummary `json:"byType"` // 按型号分组
}

// GPUTypeSummary 按型号的 GPU 汇总
type GPUTypeSummary struct {
	GPUType     string  `json:"gpuType"`
	Total       int32   `json:"total"`
	Available   int32   `json:"available"`
	Used        int32   `json:"used"`
	MemoryMB    int64   `json:"memoryMB"`
	Utilization float64 `json:"utilization"`
}

// TenantQuota 租户 GPU 配额
type TenantQuota struct {
	TenantID string `json:"tenantId"`
	GPUType  string `json:"gpuType"`
	Quota    int32  `json:"quota"` // 配额上限
	Used     int32  `json:"used"`  // 已用
}

// Available 返回剩余配额
func (q *TenantQuota) Available() int32 {
	if q == nil {
		return 0
	}
	a := q.Quota - q.Used
	if a < 0 {
		return 0
	}
	return a
}
